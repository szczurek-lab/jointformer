import numpy as np

from rdkit import Chem
from rdkit.Chem.QED import qed
from jointformer.utils.properties.smiles.molbert.featurizer import PhysChemFeaturizer

from jointformer.utils.properties.smiles.base import BaseTarget

NORMALIZE = True


class PhysChem(BaseTarget):
    """ PhysChem target. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.descriptor_set = 'all'
        self.num_physchem = 200
        self.descriptor_list = PhysChemFeaturizer.get_descriptor_subset(self.descriptor_set, self.num_physchem)
        self.physchem_featurizer = PhysChemFeaturizer(descriptors=self.descriptor_list, normalise=True)

    def _get_target(self, example: str) -> float:
        physchem, valid = self.physchem_featurizer.transform_single(example)
        assert bool(valid), f'Cannot compute the physchem props for {example}'
        return physchem[:self.num_physchem].tolist()

    @property
    def target_names(self):
        return self.descriptor_list

    def __repr__(self):
        return f"Physicochemical properties: {self.descriptor_list}"

    def __str__(self):
        return f"Physicochemical properties: {self.descriptor_list}"

    def __len__(self):
        return 200
