import os
import torch
import random

from tqdm import tqdm
from typing import Tuple, List, Any
from torch.utils.data.dataset import Dataset
from urllib.request import urlretrieve
from guacamol.utils.chemistry import is_valid

from rdkit import Chem

import deepchem as dc
import numpy as np
import pandas as pd

from deepchem.feat.molecule_featurizers.raw_featurizer import RawFeaturizer

from hybrid_transformer.utils.datasets.utils import load_txt_into_list, save_list_into_txt
from hybrid_transformer.utils.objectives.molecule_net.objective import MOLECULE_NET_TASKS, MOLECULE_NET_CLASSIFICATION_TASKS
from hybrid_transformer.utils.utils import select_random_indices_from_length
from hybrid_transformer.utils.transforms.augment import AugmentSMILES
from rdkit.Chem.AllChem import MolToSmiles, MolFromSmiles

from hybrid_transformer.utils.objectives.molecule_net.objective import DTYPE_OBJECTIVE

from hybrid_transformer.utils.runtime import set_seed

DATASET_SEED = 0
MAX_MOLECULES_LENGTH = 126

DATA_FOLDER = './data/molecule_net'
FILE_NAME = "smiles.txt"

TARGET_LABELS = {}


class MoleculeNetSMILESDataset(Dataset):

    def __init__(self, split: str, target_label: str, transforms: List[Any]) -> None:

        super().__init__()
        self.data = None
        self.target = None
        self.split = split
        self.target_label = target_label
        self.transforms = transforms
        self.target_transforms = None
        self.path_to_data_dir = None

        self._check_args()
        self._get_path_to_data_dir()
        self._load_data()
        self._load_target()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:

        x = self.data[idx]

        if self.target is not None:
            for transform in self.transforms:
                x = transform(x)

        if self.target_label is None:
            return x
        else:
            y = self.target[idx]
            return x, y

    def _check_args(self):
        if self.target_label not in MOLECULE_NET_TASKS:
            raise ValueError('Variable `target_label` must be one of "%s"' % MOLECULE_NET_TASKS)

    def _get_path_to_data_dir(self) -> None:
        self.path_to_data_dir = os.path.join(DATA_FOLDER, self.target_label)
        return None

    def _load_data(self) -> None:
        filename = os.path.join(self.path_to_data_dir, self.split, FILE_NAME)
        if not os.path.exists(filename):
            self._download()
        self.data = load_txt_into_list(filename)

        featurizer = RawFeaturizer(smiles=True)
        splitter = 'scaffold' if self.target_label in MOLECULE_NET_CLASSIFICATION_TASKS else 'random'
        target_transforms = None

        if self.target_label == 'esol':
            _, _, target_transforms = dc.molnet.load_delaney(
                featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)
        if self.target_label == 'freesolv':
            _, _, target_transforms = dc.molnet.load_sampl(featurizer=featurizer, splitter=splitter)
        if self.target_label == 'lipo':
            _, _, target_transforms = dc.molnet.load_lipo(featurizer=featurizer, splitter=splitter)
        if self.target_label == 'bace':
            _, _, target_transforms = dc.molnet.load_bace_classification(
                featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)
        if self.target_label == 'hiv':
            _, _, target_transforms = dc.molnet.load_hiv(featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)
        if self.target_label == 'tox21':
            _, _, target_transforms = dc.molnet.load_tox21(featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)
        if self.target_label == 'toxcast':
            _, _, target_transforms = dc.molnet.load_toxcast(featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)
        if self.target_label == 'sider':
            _, _, target_transforms = dc.molnet.load_sider(featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)
        if self.target_label == 'muv':
            _, _, target_transforms = dc.molnet.load_muv(featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)
        if self.target_label == 'clintox':
            _, _, target_transforms = dc.molnet.load_clintox(featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)

        self.target_transforms = target_transforms if target_transforms is not None else None

        return None

    def _load_target(self) -> None:
        filename = os.path.join(self.path_to_data_dir, self.split, f'{self.target_label}.pt')
        if not os.path.exists(filename):
            self._download()
        self.target = torch.load(filename)
        return None

    def _download(self) -> None:
        print(f"Downloading {self.target_label} MoleculeNet task...")

        set_seed(DATASET_SEED)
        featurizer = RawFeaturizer(smiles=True)
        splitter = 'scaffold' if self.target_label in MOLECULE_NET_CLASSIFICATION_TASKS else 'random'

        if self.target_label == 'esol':
            _, datasets, _ = dc.molnet.load_delaney(
                featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)
        if self.target_label == 'freesolv':
            _, datasets, _ = dc.molnet.load_sampl(
                featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)
        if self.target_label == 'lipo':
            _, datasets, _ = dc.molnet.load_lipo(
                featurizer=featurizer, splitter=splitter, data_dir=self.path_to_data_dir)
        if self.target_label == 'bace':
            _, datasets, _ = dc.molnet.load_bace_classification(
                featurizer=featurizer, splitter=splitter)
        if self.target_label == 'bbbp':
            datasets = self._load_bbbp_dataset()
        if self.target_label == 'hiv':
            _, datasets, _ = dc.molnet.load_hiv(
                featurizer=featurizer, splitter=splitter)
        if self.target_label == 'tox21':
            _, datasets, _ = dc.molnet.load_tox21(
                featurizer=featurizer, splitter=splitter)
        if self.target_label == 'toxcast':
            _, datasets, _ = dc.molnet.load_toxcast(
                featurizer=featurizer, splitter=splitter)
        if self.target_label == 'sider':
            _, datasets, _ = dc.molnet.load_sider(
                featurizer=featurizer, splitter=splitter)
        if self.target_label == 'muv':
            _, datasets, _ = dc.molnet.load_muv(
                featurizer=featurizer, splitter=splitter)
        if self.target_label == 'clintox':
            _, datasets, _ = dc.molnet.load_clintox(
                featurizer=featurizer, splitter=splitter)

        split_names = ['train', 'val', 'test']
        for idx, split in enumerate(datasets):
            data = split.ids.tolist() if self.target_label == 'bbbp' else split.X.tolist()
            target = split.X if self.target_label == 'bbbp' else split.y
            target = torch.Tensor(target).to(DTYPE_OBJECTIVE)
            data, target = self._validate(data, target)

            split_name = split_names[idx]
            data_dir = os.path.join(self.path_to_data_dir, split_name)
            os.makedirs(data_dir, exist_ok=True)
            save_list_into_txt(os.path.join(data_dir, FILE_NAME), data)
            torch.save(target, os.path.join(data_dir, f"{self.target_label}.pt"))

        print(f"Downloaded into {self.path_to_data_dir}")

    def undo_target_transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.target_transforms is None or self.target_label not in ['esol', 'freesolv', 'lipo']:
            return y
        y_undone = y.numpy()
        for transform in reversed(self.target_transforms):
            y_undone = transform.untransform(y_undone)
        return torch.from_numpy(y_undone).view(-1, 1)

    @staticmethod
    def _validate(data, target):

        data_validated = []
        target_validated = []

        for idx, (x, y) in enumerate(zip(data, target)):
            if is_valid(x) and not torch.isnan(y):
                data_validated.append(Chem.MolToSmiles(Chem.MolFromSmiles(x)))
                target_validated.append(y)

        return data_validated, torch.Tensor(target_validated).view(-1, 1)

    @classmethod
    def from_config(cls, config):
        transforms = []
        if config.augment_molecular_representation:
            transforms.append(AugmentSMILES(augmentation_prob=config.augmentation_prob))

        return cls(split=config.split, target_label=config.target_label, transforms=transforms)

    def _create_dir_from_path(path: str, overwrite: bool = True) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=overwrite)

    @staticmethod
    def _load_bbbp_dataset(seed: int = 0):

        # define path
        data_path = DATA_FOLDER + 'bbbp' + '/bbbp.csv'

        # download
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        bbbp_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv'
        urlretrieve(bbbp_url, data_path)

        # load
        input = pd.read_csv(data_path, sep=',')
        input_smiles = input['smiles']
        input_labels = input['p_np']
        input_labels.replace(-1, 0, inplace=True)

        assert len(input_smiles) == len(input_labels)
        mols = [MolFromSmiles(smiles) for smiles in input_smiles]
        output_smiles = [MolToSmiles(mol) for mol in mols if mol]
        output_labels = [label for mol, label in zip(mols, input_labels) if mol]

        dataset = dc.data.DiskDataset.from_numpy(ids=output_smiles, X=np.array(output_labels))
        scaffoldsplitter = dc.splits.ScaffoldSplitter()
        train_dataset, val_dataset, test_dataset = scaffoldsplitter.train_valid_test_split(dataset, seed=seed)

        return train_dataset, val_dataset, test_dataset
