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
import csv

from regression_transformer.terminator.factories import MODEL_TO_EMBEDDING_FN, NUM_ENCODING_FACTORY
#from molecule_generation import load_model_from_directory
from regression_transformer.terminator.args import CustomTrainingArguments, EvalArguments
from regression_transformer.terminator.collators import (
    ConditionalGenerationEvaluationCollator,
    PropertyCollator,
)
from regression_transformer.terminator.datasets import get_dataset
from regression_transformer.terminator.evaluator import Evaluator
from regression_transformer.terminator.property_predictors import PREDICT_FACTORY
from regression_transformer.terminator.tokenization import ExpressionBertTokenizer
from regression_transformer.terminator.trainer import get_trainer_dict
from regression_transformer.terminator.utils import (
    disable_rdkit_logging,
    find_safe_path,
    get_latest_checkpoint,
    get_equispaced_ranges,
)
from regression_transformer.terminator.search import BeamSearch, GreedySearch, SamplingSearch

def generate(
        num_samples: int,
        save_path: str = None,
        denormalize_params: Optional[List[float]] = None,
        return_sequences: bool = False,
        beam_width: int = 5,
        max_seq_length: int = 45
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
    
    model = AutoModelWithLMHead.from_pretrained('trained_models/qed/',config=config)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model.eval()
    #if eval_params.get("block_size", -1) <= 0:
    #    eval_params["block_size"] = tokenizer.max_len
        # Our input block size will be the max possible for the model
   # else:
     #   eval_params["block_size"] = min(training_args.block_size, tokenizer.max_len)
    #eval_dataset = get_dataset(
    #    eval_args.eval_file,
    #    block_size=eval_params["block_size"],
    #    tokenizer=tokenizer,
     #   line_by_line=eval_params.get("line_by_line", True),
    #)
    #vanilla_collator = DataCollatorForPermutationLanguageModeling(
     #   tokenizer=tokenizer,
     #   plm_probability=0.2,
     #   max_span_length=eval_params["max_span_length"],
    #)
    #custom_trainer_params = get_trainer_dict(model_params)
    #evaluator = Evaluator(
     #   model=model,
     #   args=training_args,
      #  eval_params=eval_params,
       # data_collator=vanilla_collator,
       # eval_dataset=eval_dataset,
       # tokenizer=tokenizer,
       # prediction_loss_only=False,
       # **custom_trainer_params,
    #)
    #properties = eval_params["property_tokens"]
    #for prop in properties:
    #    print(prop)
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
    #conditioning_range=[
    #        0.051,
    #        0.151,
     #       0.251,
     #       0.351,
     #       0.451,
     #       0.551,
     #       0.651,
     #       0.751,
     #       0.851,
     #       0.951
       # ]
    #collator = ConditionalGenerationEvaluationCollator(
     #       tokenizer=tokenizer,
     #       property_token=prop,
     #       conditioning_range=conditioning_range,
     #       plm_probability=0.5,
     #       max_span_length=eval_params["max_span_length"],
     #       entity_to_mask=eval_params.get("entity_to_mask", None),
     #       entity_separator_token=eval_params.get("entity_separator_token", None)
     #   )
    #dataloader = evaluator.get_custom_dataloader(collator, bs=num_samples)

    # Prepare denormalization function

    
    if denormalize_params:
        denormalize = lambda x: x * (denormalize_params[1] - denormalize_params[0]) + denormalize_params[0]
    else:
        denormalize = lambda x: x

    greedy_search = GreedySearch()
    beam_search = BeamSearch(
            temperature=eval_params.get("temperature", 1.0),
            beam_width=eval_params.get("beam_width", 1),
            top_tokens=eval_params.get("beam_top_tokens", 5),
        )
    # Run the prediction loop
    #logits, label_ids, metrics, input_ids, returned = evaluator.prediction_loop(
    #    dataloader=dataloader,
    #    description=f"Conditional generation {prop}",
     #   prediction_loss_only=False,
     #   return_inputs=True,
     #   pop_and_return_keys=["real_property", "sample_weights"],
     #   pad_idx=tokenizer.vocab["[PAD]"],
    #)
    
    #logits = torch.Tensor(logits).cpu()
    #input_ids = torch.Tensor(input_ids)

    #bos_token_id = tokenizer.bos_token
    #prefix = "<qed>0.931|<[MASK]>"
    #generated_logits = []
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #input_ids =torch.tensor(tokenizer.encode(prefix,add_special_tokens=False)).unsqueeze(0)
    #labels = torch.tensor(tokenizer.encode("[C][C][Branch1_1][C][C][Branch1_1][C][C][C][Branch2_1][Ring1][Branch1_1][C][=C][C][C][=C][Branch1_1][Branch2_2][C][C][C][Expl=Ring1][Branch1_2][S][Ring1][Branch2_2][C][Branch1_1][C][O][=O][N][C][=C][N][=C][Ring1][Branch1_1]", add_special_tokens=False)).unsqueeze(0)
    all_generated_sequences = []
    #model_inputs = input_ids.copy()
    for _ in range(num_samples):
        # Prepare input IDs
        #bos_token_id = tokenizer.bos_token_id
        qed_value = random.uniform(0, 1)
        prefix = f"<qed>{qed_value:.3f}|<[MASK]>"
        #prefix = "<qed>0.931|<[MASK]>"
        input_ids = torch.tensor(tokenizer.encode(prefix, add_special_tokens=False)).unsqueeze(0)
        
        # Prepare beam search predictions
        #beam_preds = torch.zeros(beam_width=3, max_seq_length=50, dtype=torch.long)
        #beam_preds = torch.zeros((beam_width, max_seq_length), dtype=torch.long)
        current_input_ids = input_ids

        # Prepare permutation mask for autoregressive generation
        #perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        #perm_mask[:, :, -1] = 1.0  # Ensure the last token is predicted

        # Generate sequences
        #current_input_ids = input_ids
        for _ in range(50):
            # Prepare target mapping for prediction
            #target_mapping = torch.zeros((1, 1, current_input_ids.shape[1]), dtype=torch.float)
            #target_mapping[0, 0, -1] = 1.0
            
            with torch.no_grad():
                outputs = model(input_ids=current_input_ids)
                if isinstance(outputs, tuple):
                    next_token_logits = outputs[0]  # logits are usually the first element of the tuple
                else:
                    next_token_logits = outputs.logits  
                

            logits = next_token_logits[:, -1, :]  # Extract logits for the last token
            logits = logits.unsqueeze(1)

            # Sample from the logits to get the next token
            #logits = next_token_logits[:, -1, :]
            #next_token = torch.argmax(next_token_logits[:, -1, :], dim=-1)
            greedy_preds = greedy_search(logits).unsqueeze(1)
            #print(beam_preds.shape)
            #print(greedy_preds.shape)
            bw = beam_search.beam_width
            #beam_preds = torch.cat([beam_preds, greedy_preds], dim=1)
            beam_preds = torch.cat([greedy_preds] * bw, dim=0).squeeze(1)
            #beam_preds = torch.cat([beam_preds, logits.unsqueeze(1)], dim=1)

            #perorm beam search
            relevant_logits = logits
            beams, _ = beam_search(relevant_logits)
            beam_preds[:, -1] = beams.long()
            next_token = beams[:, -1]
            
            # Append the predicted token to the sequence
            #current_input_ids = torch.cat([current_input_ids, beam_preds[:, -1].unsqueeze(0)], dim=-1)
            current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
            

            #current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Update permutation mask to include the new token
            #perm_mask = torch.zeros((1, current_input_ids.shape[1], current_input_ids.shape[1]), dtype=torch.float)
            #perm_mask[:, :, -1] = 1.0

            #bw = beam_search.beam_width
            #beam_preds = torch.zeros(bw, *logits.shape[:2]).long()
            #for sample_idx in tqdm(range(logits.shape[0]), desc="Beam search"):
             #   keep_pos = torch.nonzero(input_ids[sample_idx, :] == tokenizer.vocab["[MASK]"]).squeeze(1)
             #   relevant_logits = logits[sample_idx, keep_pos, :].unsqueeze(0)
             #   if len(relevant_logits.shape) == 2:
              #      relevant_logits = relevant_logits.unsqueeze(0)
              #      beams, scores = beam_search(relevant_logits)
              #      beam_preds[:, sample_idx, keep_pos] = beams.squeeze(dim=0).permute(1, 0).long()
            
            # Decode and evaluate sequences
        decoded_sequences = []

        decoded_sequences = [tokenizer.decode(current_input_ids[i], skip_special_tokens=True) for i in range(current_input_ids.size(0))]
        
        all_generated_sequences.extend(decoded_sequences)
        for seq in all_generated_sequences:
            seq.split("|")[-1]
        #for yhat in beam_preds:
         #   decoded_seq = tokenizer.decode(yhat, skip_special_tokens=True)
          #  decoded_sequences.append(decoded_seq)

        #all_generated_sequences.extend(decoded_sequences)

    if save_path:
        save_path = find_safe_path(save_path)
        df = pd.DataFrame({
            "Sample_ID": range(1, len(all_generated_sequences) + 1),
            "Generated_Sequence": all_generated_sequences
        })
        df.to_csv(save_path, index=False)
        print(f"Generated sequences saved in {save_path}")

    if return_sequences:
        return all_generated_sequences
    return None

            

        # Decode the generated sequence
        #generated_text = tokenizer.decode(current_input_ids[0], skip_special_tokens=True)
        #all_generated_sequences.append(generated_text)

# Example usage
generated_sequences = generate(
    num_samples=20,
    save_path='generated_sequences.csv',
    return_sequences=True
)