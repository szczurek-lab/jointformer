import torch

from torch.utils.data._utils.collate import default_collate
from typing import List, Tuple, Any, Optional, Union
from jointformer.models.utils import ModelInput


TOKEN_DICT = {
    'cls': '[BOS]',
    'pad': '[PAD]',
    'mask': '[MASK]',
    'sep': '[EOS]',
    'unknown': '[UNK]',
    'ignore': -100
}


class BaseTokenizer:

    def __init__(
        self,
        path_to_vocabulary: str,
        max_molecule_length: int,
        mlm_probability: Optional[float] = 0.15,
        ignore_index: Optional[int] = TOKEN_DICT['ignore']
    ):

        self.tokenizer = None        
        self.max_molecule_length = max_molecule_length
        self.mlm_probability = mlm_probability
        self.ignore_index = ignore_index
        self.generation_prefix = None
        self._init_tokenizer(path_to_vocabulary)
        self._set_generation_prefix()

    @property
    def cls_token_id(self):
        raise NotImplementedError
    
    @property
    def mask_token_id(self):
        raise NotImplementedError
    
    @property
    def cls_token_id(self):
        raise NotImplementedError
    
    @property
    def pad_token_id(self):
        raise NotImplementedError
    
    @property
    def eos_token_id(self):
        raise NotImplementedError
    
    @property
    def sep_token_id(self):
        raise NotImplementedError
    
    def _set_generation_prefix(self):
        raise NotImplementedError
        
    def _init_tokenizer(self, path_to_vocabulary: str):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.tokenizer)

    def _tokenize(self, data: Union[str, List[str]]):
        raise NotImplementedError

    def __call__(self, x: Union[str, List[str], Tuple[str, torch.Tensor], List[Tuple[str, torch.Tensor]]], task: str) -> ModelInput:
        if isinstance(x[0], tuple):
            data, properties = zip(*x)
        elif isinstance(x, tuple):
            data, properties = x[0], [x[1]]
        elif isinstance(x, str) or (isinstance(x, list) and isinstance(x[0], str)):
            data = x
            properties = None
        else:
            raise ValueError('Variable `data` must be either a string, a list of strings or a tuple of strings and tensors.')

        batch = self._tokenize(data)
        batch["attention_mask"] = batch["attention_mask"].bool()
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["task"] = task

        if task == 'generation':
            labels = batch["input_ids"].clone()
            if self.pad_token_id is not None:
                labels[labels == self.pad_token_id] = self.ignore_index
            if self.cls_token_id is not None:
                labels[labels == self.cls_token_id] = self.ignore_index
            batch["input_labels"] = labels

        elif task == 'mlm':
            batch["input_ids"], batch["input_labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask)

        elif task == 'prediction' or task == 'physchem':
            pass  # do nothing

        else:
            raise ValueError('Variable `task` must be either `generation`, `mlm` or `prediction`.')

        if properties is not None:
            batch["properties"] = default_collate(properties)
            assert len(batch["properties"].shape) == 2 and isinstance(batch["properties"], torch.Tensor)

        return ModelInput(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            task=task,
            input_labels=batch.get("input_labels", None),
            properties=batch.get("properties", None)
        )
        
    def mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Reimplements preparing masked token inputs / labels for masked language modeling: 80% MASK, 20% original.

        No random token replacement is performed, as this would correspond to invalid SMILES strings.

        Source: https://github.com/huggingface/transformers/blob/745bbfe4bb2b61491dedd56e1e8ee4af8ef1a9ec/src/transformers/data/data_collator.py#L782
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = TOKEN_DICT['ignore']  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.convert_tokens_to_ids(self.mask_token)

        # We do not replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (20% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def decode(self, x: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        smiles_data = []
        for _, row in enumerate(x):
            smiles = self.tokenizer.decode(row, skip_special_tokens=skip_special_tokens).replace(' ', '')
            smiles_data.append(smiles)
        return smiles_data
