
from typing import List, Optional, Union, Callable
from jointformer.configs.base import Config


class TaskConfig(Config):

    def __init__(
            self,
            dataset_name: str,
            target_label: str,
            validate: bool,
            num_samples: int,
            split: str,
            transform: Optional[Union[Callable, List]],
            target_transform: Optional[Union[Callable, List]],
            tokenizer: Optional[Union[Callable, List]],
            path_to_vocabulary: str,
            max_molecule_length: int,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.target_label = target_label
        self.validate = validate
        self.num_samples = num_samples
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = tokenizer
        self.path_to_vocabulary = path_to_vocabulary
        self.max_molecule_length = max_molecule_length