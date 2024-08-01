import json
import logging
import math
import os
import sys
import torch 
import random  
from time import time
from tqdm import tqdm
import pandas as pd
from typing import Dict, Iterable, List, Optional, Tuple, Union
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    set_seed,
)

#from molecule_generation import load_model_from_directory
from terminator.args import CustomTrainingArguments, EvalArguments
from terminator.collators import (
    ConditionalGenerationEvaluationCollator,
    PropertyCollator,
)
from terminator.datasets import get_dataset
from terminator.evaluator import Evaluator
from terminator.property_predictors import PREDICT_FACTORY
from terminator.tokenization import ExpressionBertTokenizer
from terminator.trainer import get_trainer_dict
from terminator.utils import (
    disable_rdkit_logging,
    find_safe_path,
    get_latest_checkpoint,
    get_equispaced_ranges,
)
from terminator.search import BeamSearch, GreedySearch, SamplingSearch

def generate(
        num_samples: int,
        save_path: str = None,
        denormalize_params: Optional[List[float]] = None,
        return_sequences: bool = False,
    ) -> Union[List[str], None]:
    """
    Function to generate sequences using beam search.

    Args:
        collator (): PyTorch collator object
        num_samples (int): Number of samples to generate.
        save_path (str): Path where results are saved, defaults to None (no saving).
        denormalize_params: The min and max values of the property to denormalize the results.
        return_sequences (bool): Whether to return the generated sequences.
        
    Returns:
        List of generated sequences if return_sequences is True, otherwise None.
    """

    parser = HfArgumentParser((CustomTrainingArguments, EvalArguments))
    training_args, eval_args = parser.parse_args_into_dataclasses()
    with open(eval_args.param_path, "r") as f:
        eval_params = json.load(f)

    param_filename = eval_args.param_path.split("/")[-1].split(".json")[0]

    # Wrap into args to be safe
    eval_args.__dict__.update(eval_params)

    model_dir = training_args.output_dir
    # Generate dataloader with custom collator
    config_name = os.path.join(model_dir, "config.json")
    with open(config_name, "r") as f:
        model_params = json.load(f)

    config = AutoConfig.from_pretrained(
        config_name, mem_len=model_params.get("mem_len", 1024)
    )
    tokenizer = ExpressionBertTokenizer.from_pretrained(model_dir)
    sep = tokenizer.expression_separator
    #dataloader = evaluator.get_custom_dataloader(collator, bs=num_samples)
    
    model = AutoModelWithLMHead.from_pretrained('./jointformer/models/regression-transformer/trained_models/qed/',config=config)
    model.resize_token_embeddings(len(tokenizer))

    model.eval()
    if eval_params.get("block_size", -1) <= 0:
        eval_params["block_size"] = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        eval_params["block_size"] = min(training_args.block_size, tokenizer.max_len)
    eval_dataset = get_dataset(
        eval_args.eval_file,
        block_size=eval_params["block_size"],
        tokenizer=tokenizer,
        line_by_line=eval_params.get("line_by_line", True),
    )
    vanilla_collator = DataCollatorForPermutationLanguageModeling(
        tokenizer=tokenizer,
        plm_probability=0.2,
        max_span_length=eval_params["max_span_length"],
    )
    custom_trainer_params = get_trainer_dict(model_params)
    evaluator = Evaluator(
        model=model,
        args=training_args,
        eval_params=eval_params,
        data_collator=vanilla_collator,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        prediction_loss_only=False,
        **custom_trainer_params,
    )
    properties = eval_params["property_tokens"]
    for prop in properties:
        print(prop)
    #prop = '<qed>'

    #conditioning_ranges = eval_params.get(
     #   "conditioning_range",
     #   get_equispaced_ranges(
      #      eval_args.eval_file,
       #     prop,
        #    precisions=eval_params.get("property_precisions", [2] * len(prop)),
        #),
    #)
        
    #conditioning_range = [random.uniform(0, 1)]
    conditioning_range=[
            0.051,
            0.151,
            0.251,
            0.351,
            0.451,
            0.551,
            0.651,
            0.751,
            0.851,
            0.951
        ]
    collator = ConditionalGenerationEvaluationCollator(
            tokenizer=tokenizer,
            property_token=prop,
            conditioning_range=conditioning_range,
            plm_probability=0.5,
            max_span_length=eval_params["max_span_length"],
            entity_to_mask=eval_params.get("entity_to_mask", None),
            entity_separator_token=eval_params.get("entity_separator_token", None)
        )
    dataloader = evaluator.get_custom_dataloader(collator, bs=num_samples)

    # Prepare denormalization function

    
    if denormalize_params:
        denormalize = lambda x: x * (denormalize_params[1] - denormalize_params[0]) + denormalize_params[0]
    else:
        denormalize = lambda x: x

    seq_to_prop = {}

    beam_search = BeamSearch(
            temperature=eval_params.get("temperature", 1.0),
            beam_width=eval_params.get("beam_width", 1),
            top_tokens=eval_params.get("beam_top_tokens", 5),
        )
    # Run the prediction loop
    logits, label_ids, metrics, input_ids, returned = evaluator.prediction_loop(
        dataloader=dataloader,
        description=f"Conditional generation {prop}",
        prediction_loss_only=False,
        return_inputs=True,
        pop_and_return_keys=["real_property", "sample_weights"],
        pad_idx=tokenizer.vocab["[PAD]"],
    )
    
    logits = torch.Tensor(logits).cpu()
    input_ids = torch.Tensor(input_ids)

    # Beam search restricted to affected logits
    bw = beam_search.beam_width
    beam_preds = torch.zeros(bw, *logits.shape[:2]).long()
    for sample_idx in tqdm(range(logits.shape[0]), desc="Beam search"):
        keep_pos = torch.nonzero(input_ids[sample_idx, :] == tokenizer.vocab["[MASK]"]).squeeze(1)
        relevant_logits = logits[sample_idx, keep_pos, :].unsqueeze(0)
        if len(relevant_logits.shape) == 2:
            relevant_logits = relevant_logits.unsqueeze(0)
        beams, scores = beam_search(relevant_logits)
        beam_preds[:, sample_idx, keep_pos] = beams.squeeze(dim=0).permute(1, 0).long()

    # Decode and evaluate sequences
    decoded_sequences = []
    for yhat in tqdm(beam_preds, desc="Evaluating search"):
        for x, yhat_seq in zip(input_ids, yhat):
            x_tokens = tokenizer.decode(x, clean_up_tokenization_spaces=False).split(" ")
            yhat_tokens = tokenizer.decode(yhat_seq, clean_up_tokenization_spaces=False).split(" ")
            genseq, _ = tokenizer.aggregate_tokens(tokenizer.get_sample_prediction(yhat_tokens, x_tokens), label_mode=False)
            decoded_sequences.append(genseq.split("|")[-1])

    # Save generated sequences if save_path is provided
    if save_path is not None:
        sequences_df = pd.DataFrame({"GeneratedSequence": decoded_sequences})
        sequences_df.to_csv(save_path, index=False)
        print(f"Generated sequences saved in {save_path}")

    if return_sequences:
        return decoded_sequences
    return None

# Example usage
generated_sequences = generate(num_samples=10, save_path='generated_sequences.csv', return_sequences=True)
print(generated_sequences)