""" AutoDataset class for automatic dataset selection based on config.

This module contains the AutoDataset class, which is used to automatically select the
 appropriate dataset class based on the dataset name specified in the config.

"""

import importlib

from jointformer.configs.dataset import DatasetConfig
from jointformer.utils.datasets.base import BaseDataset


class AutoDataset:

    @classmethod
    def from_config(
            cls,
            config: DatasetConfig,
            split: 'str' = None,
            num_samples: int = None,
            target_label: str = None,
            validate: bool = None,
            standardize: bool = None,
            data_dir: str = None,
            seed: int = None
    ) -> BaseDataset:

        # Override config values if available
        if split is not None:
            config.split = split
        if num_samples is not None:
            config.num_samples = num_samples
        if target_label is not None:
            config.target_label = target_label
        if validate is not None:
            config.validate = validate
        if standardize is not None:
            config.standardize = standardize

        # Disable sampling for validation and test splits
        if config.split in ['val', 'test', 'test_scaffolds']:
            config.num_samples = None

        if config.dataset_name == 'moses':
            return getattr(importlib.import_module(
                "jointformer.utils.datasets.smiles.moses"),
                "MosesDataset").from_config(config)
        elif config.dataset_name == 'guacamol':
            return getattr(importlib.import_module(
                "jointformer.utils.datasets.guacamol"),
                "GuacamolDataset").from_config(config, root=data_dir, seed=seed)
        elif config.dataset_name == 'molecule_net':
            return getattr(importlib.import_module(
                "jointformer.utils.datasets.smiles.molecule_net"),
                "MoleculeNetDataset").from_config(config, data_dir=data_dir, seed=seed)
        elif config.dataset_name == 'sequence_dataset':
            return getattr(importlib.import_module(
                "jointformer.utils.datasets.sequence"),
                "SequentialDataset").from_config(config, data_dir=data_dir, seed=seed)
        else:
            raise ValueError(f"Dataset {config.dataset_name} not available.")
