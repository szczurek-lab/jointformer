import os
import torch

from urllib.request import urlretrieve
from typing import Optional

from jointformer.utils.properties.auto import AutoTarget
from jointformer.configs.dataset import DatasetConfig
from jointformer.utils.datasets.sequence import SequentialDataset
from jointformer.utils.data import read_strings_from_file


DEFAULT_DATA_SUBDIR = 'datasets/guacamol'
DATA_FILENAME = 'smiles.txt'

GUACAMOL_URL = {
    'train': "https://ndownloader.figshare.com/files/13612760",
    'val': "https://ndownloader.figshare.com/files/13612766",
    'test': "https://ndownloader.figshare.com/files/13612757",
    'all': "https://ndownloader.figshare.com/files/13612745"
}


class GuacamolDataset(SequentialDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = self.target.squeeze(1) # TODO: delete __init__ - hotfix

    @staticmethod
    def _download_data(data_filepath: str, split: str) -> None:
        print(f"Downloading data into {data_filepath}")
        os.makedirs(os.path.dirname(data_filepath), exist_ok=True)
        urlretrieve(GUACAMOL_URL[split], data_filepath)
    
    @staticmethod
    def _download_target(data_filepath: str, target_filepath: str, target_label: str) -> None:
        print(f"Downloading target into {target_filepath}")
        data = read_strings_from_file(data_filepath)
        oracle = AutoTarget.from_target_label(target_label)
        target = oracle(data)
        torch.save(target, target_filepath)
        
    @classmethod
    def from_config(cls, config: DatasetConfig, split: Optional[str] = None, seed: Optional[int] = None, root: str = None):
        
        # Set split
        if split is None:
            split = config.split
        else:
            raise ValueError("Provide a correct split value.")

        # Download data and target
        data_filepath = os.path.join(root, DEFAULT_DATA_SUBDIR, split, DATA_FILENAME)
        if not os.path.exists(data_filepath):
            cls._download_data(data_filepath, split)
        if config.target_label is not None:
            target_filepath = os.path.join(root, DEFAULT_DATA_SUBDIR, split, f"{config.target_label}.pt")
            if not os.path.exists(target_filepath):
                cls._download_target(data_filepath, target_filepath, config.target_label)
        else:
            target_filepath = None
        
        # Init
        return cls._from_filepath(
            root=None,
            data_filepath=data_filepath,
            target_filepath=target_filepath,
            transform=config.transform,
            target_transform=config.target_transform,
            seed=seed,
            num_samples=config.num_samples,
            task_type=config.task_type
        )
