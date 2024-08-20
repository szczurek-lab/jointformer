from deepchem.feat.smiles_tokenizer import SmilesTokenizer as DeepChemSmilesTokenizer
from typing import List, Optional, Union

from jointformer.utils.tokenizers.base import BaseTokenizer, TOKEN_DICT

from transformers import AutoTokenizer


class HFTokenizer(BaseTokenizer):

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
            ignore_index=ignore_index
        )

    def _init_tokenizer(self, path_to_vocabulary: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_vocabulary)

    def __len__(self):
        return len(self.tokenizer)

    def _tokenize(self, data: Union[str, List[str]]):
        return self.tokenizer(
            data,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=self.max_molecule_length,
            return_special_tokens_mask=False
        )
        
    @classmethod
    def from_config(cls, config):
        return cls(
            path_to_vocabulary=config.path_to_vocabulary,
            max_molecule_length=config.max_molecule_length
        )
