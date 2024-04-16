import os
import torch
import logging
import random
import moses

from tqdm import tqdm
from typing import Tuple, List, Any
from torch.utils.data.dataset import Dataset
from urllib.request import urlretrieve
from guacamol.utils.chemistry import is_valid
from rdkit import Chem

from hybrid_transformer.utils.datasets.utils import load_txt_into_list, save_list_into_txt
from hybrid_transformer.utils.objectives.guacamol.objective import get_objective
from hybrid_transformer.utils.objectives.guacamol.utils import GUACAMOL_TASK_NAMES
from hybrid_transformer.utils.utils import select_random_indices_from_length
from hybrid_transformer.utils.transforms.augment import AugmentSMILES

DATASET_SEED = 0
DATA_FOLDER = './data/moses'
FILE_NAME = "smiles.txt"

TARGET_LABELS = {}


class MOSESSMILESDataset(Dataset):

    def __init__(
            self, split: str = None, transforms: List[Any] = None,
            validate: bool = True, num_samples: int = None) -> None:

        super().__init__()
        self.data = None
        self.target = None
        self.target_label = None
        self.split = split
        self.transforms = transforms
        self.target_transforms = None
        self.path_to_data_dir = None
        self.num_samples = num_samples if self.split == 'train' else None  # subset only training data

        self._load_data()
        if validate:
            self._validate()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:

        x = self.data[idx]

        if self.transforms is not None:
            for transform in self.transforms:
                x = transform(x)

        return x

    def _load_data(self) -> None:
        self.data = moses.get_dataset(split=self.split).tolist()
        return None

    def _validate(self):
        data_safe = []
        for idx, x in enumerate(tqdm(self.data)):
            if is_valid(x):
                data_safe.append(x)
        self.data = data_safe
        return None

    def _subset_dataset(self):
        idx = select_random_indices_from_length(len(self.data), self.num_samples)
        self.data = [self.data[i] for i in idx]
        return None

    def undo_target_transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.target_transforms is None:
            return y
        y_undone = y.numpy()
        for transform in reversed(self.target_transforms):
            y_undone = transform.untransform(y_undone)
        return torch.from_numpy(y_undone).view(-1, 1)

    @classmethod
    def from_config(cls, config):
        transforms = []
        if config.augment_molecular_representation:
            transforms.append(AugmentSMILES(augmentation_prob=config.augmentation_prob))

        return cls(
            split=config.split, transforms=transforms,
            validate=config.validate, num_samples=config.num_samples)
