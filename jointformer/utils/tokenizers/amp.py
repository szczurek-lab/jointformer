import torch

from typing import List, Optional, Union, Tuple

from jointformer.utils.tokenizers.base import TOKEN_DICT
from jointformer.utils.tokenizers.hf import HFTokenizer
from jointformer.utils.tokenizers.smiles_separate_task_token import SmilesTokenizerSeparateTaskToken
from jointformer.utils.tokenizers.smiles_with_prefix import SmilesTokenizerWithPrefix
from jointformer.models.utils import ModelInput
from transformers import AutoTokenizer
TASK_TOKEN_DICT = {
    'generation': '[GEN]',
    'prediction': '[PRED]'
}


class AMPTokenizerWithPrefix(SmilesTokenizerSeparateTaskToken):

    def __init__(
        self,
        path_to_vocabulary: str,
        max_molecule_length: int,
        mlm_probability: Optional[float] = 0.15,
        ignore_index: Optional[int] = TOKEN_DICT['ignore']
    ):

        super().__init__(
            path_to_vocabulary=path_to_vocabulary,
            max_molecule_length=max_molecule_length,
            mlm_probability=mlm_probability,
            ignore_index=ignore_index,
        )
    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id
    
    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id
    
    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id
    
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def sep_token_id(self):
        return self.tokenizer.sep_token_id

    def _set_generation_prefix(self):
        self.generation_prefix = [self.tokenizer.convert_tokens_to_ids(TASK_TOKEN_DICT['generation']), self.tokenizer.cls_token_id]

    def _init_tokenizer(self, path_to_vocabulary: str):
        #super()._init_tokenizer(path_to_vocabulary)
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_vocabulary)
        self.tokenizer.add_special_tokens({'additional_special_tokens': list(TASK_TOKEN_DICT.values())})

    def __call__(self, x: Union[str, List[str], Tuple[str, torch.Tensor], List[Tuple[str, torch.Tensor]]], task: str) -> ModelInput:
        inputs = super().__call__(x, task)
        inputs['input_ids'][:, 0] = self.task_token_id(task)  # Add task token to the beginning of the sequence
        inputs['input_ids'] = inputs['input_ids'].contiguous()
        return inputs

    @classmethod
    def from_config(cls, config):
        return cls(
            path_to_vocabulary=config.path_to_vocabulary,
            max_molecule_length=config.max_molecule_length
        )

    def task_token_id(self, task: str) -> int:
        if task in ['prediction', 'physchem', 'mlm']:
            task_token = TASK_TOKEN_DICT['prediction']
        elif task == 'generation':
            task_token = TASK_TOKEN_DICT['generation']
        else:
            raise ValueError('Variable `task` must be either `generation`, `mlm` or `prediction`.')
        return self.tokenizer.convert_tokens_to_ids(task_token)
    