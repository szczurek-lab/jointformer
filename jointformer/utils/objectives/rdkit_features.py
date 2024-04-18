import json
import logging
import os
from abc import abstractmethod, ABC
from typing import List, Tuple, Dict, Sequence, Optional

import numpy as np
import pandas as pd
import scipy.stats as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MolFeaturizer(ABC):
    """
    Interface for the featurization of molecules, given as SMILES strings, to some vectorized representation.
    """

    def __call__(self, molecules: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        return self.transform(molecules)

    def transform(self, molecules: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Featurizes a sequence of molecules.

        Args:
            molecules: molecules, given as a sequence of SMILES strings

        Returns:
            Tuple: 2D array for the feature vectors, 1D array for the validity masks
        """
        single_results = [self.transform_single(m) for m in molecules]
        features_list, mask_list = zip(*single_results)

        return np.vstack(features_list), np.hstack(mask_list)

    @abstractmethod
    def transform_single(self, molecule: str) -> Tuple[np.ndarray, bool]:
        """
        Featurizes one molecule.

        Args:
            molecule: molecule, given as a SMILES string

        Returns:
            Tuple: feature vector (1D array), boolean for successful featurization
        """

    def invalid_mol_features(self) -> np.ndarray:
        """
        Features to return for invalid molecules.
        """
        return np.zeros(self.output_size)

    @property
    @abstractmethod
    def output_size(self) -> int:
        """
        Get the dimension after featurization
        """

    def is_valid(self, molecules: Sequence[str]) -> Sequence[bool]:
        return np.array([self.is_valid_single(mol) for mol in molecules])

    def is_valid_single(self, molecule: str) -> bool:
        mol = Chem.MolFromSmiles(molecule, True, {})

        if mol is None or len(molecule) == 0:
            return False

        return True


class RDKitFeaturizer(MolFeaturizer, ABC):
    """
    Base class for MolFeaturizers relying on RDKit.Mols during featurization
    """

    def __init__(self, sanitize: bool = True, replacements: Optional[dict] = None):
        """
        Args:
            sanitize: toggles sanitization of the molecule.
            replacements: a dictionary of replacement strings. Defaults to {}
                (@see http://www.rdkit.org/Python_Docs/rdkit.Chem.rdmolfiles-module.html#MolFromSmiles)
        """
        if replacements is None:
            replacements = {}

        self.sanitize = sanitize
        self.replacements = replacements

    def transform_single(self, molecule: str) -> Tuple[np.ndarray, bool]:
        mol = Chem.MolFromSmiles(molecule, self.sanitize, self.replacements)

        if mol is None or len(molecule) == 0:
            return self.invalid_mol_features(), False

        return self.transform_mol(mol)

    @abstractmethod
    def transform_mol(self, molecule: Chem.rdchem.Mol) -> Tuple[np.ndarray, bool]:
        """
        Featurizes one molecule given as a RDKit.Mol
        """


class PhysChemFeaturizer(RDKitFeaturizer):
    """
    MolFeaturizer that featurizes a molecule with an array of phys-chem properties.

    @see http://www.rdkit.org/Python_Docs/rdkit.ML.Descriptors.MoleculeDescriptors-module.html
    For available descriptors @see http://rdkit.org/docs/source/rdkit.ML.Descriptors.MoleculeDescriptors.html
    """

    def __init__(
        self,
        descriptors: List[str] = [],
        named_descriptor_set: str = 'all',
        fingerprint_extra_args: Optional[dict] = None,
        normalise: bool = False,
        subset_size: int = 200,
    ):
        """
        Args:
            descriptors: list of descriptor names -
                the subset given is validated to make sure they exist and will be used.
            named_descriptor_set: 'all' or 'simple' to use preset subsets
            fingerprint_extra_args: optional kwargs for `MolecularDescriptorCalculator`
            subset_size: number of descriptors to return (or the size of the subset if that's smaller)
        """
        super().__init__()

        if fingerprint_extra_args is None:
            fingerprint_extra_args = {}

        self.descriptors = self._get_descriptor_list(
            named_descriptor_set=named_descriptor_set, descriptor_list=descriptors, subset_size=subset_size
        )

        self.fingerprint_extra_args = fingerprint_extra_args
        self.calc = MolecularDescriptorCalculator(self.descriptors, **self.fingerprint_extra_args)
        self.normalise = normalise

        distributions_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '../data/physchem_distributions.json'
        )

        with open(distributions_path) as fp:
            self.distributions = json.load(fp)

        if self.normalise:
            self.scaler = PhyschemScaler(descriptor_list=self.descriptors, dists=self.distributions)

    @staticmethod
    def get_descriptor_subset(subset: str, subset_size: int) -> List[str]:
        if subset == 'all':
            return PhysChemFeaturizer.get_all_descriptor_names()[:subset_size]
        elif subset == 'simple':
            return PhysChemFeaturizer.get_simple_descriptor_subset()[:subset_size]
        elif subset == 'uncorrelated':
            return PhysChemFeaturizer.get_uncorrelated_descriptor_subset(subset_size)
        elif subset == 'fragment':
            return PhysChemFeaturizer.get_fragment_descriptor_subset()[:subset_size]
        elif subset == 'graph':
            return PhysChemFeaturizer.get_graph_descriptor_subset()[:subset_size]
        elif subset == 'surface':
            return PhysChemFeaturizer.get_surface_descriptor_subset()[:subset_size]
        elif subset == 'druglikeness':
            return PhysChemFeaturizer.get_druglikeness_descriptor_subset()[:subset_size]
        elif subset == 'logp':
            return PhysChemFeaturizer.get_logp_descriptor_subset()[:subset_size]
        elif subset == 'refractivity':
            return PhysChemFeaturizer.get_refractivity_descriptor_subset()[:subset_size]
        elif subset == 'estate':
            return PhysChemFeaturizer.get_estate_descriptor_subset()[:subset_size]
        elif subset == 'charge':
            return PhysChemFeaturizer.get_charge_descriptor_subset()[:subset_size]
        elif subset == 'general':
            return PhysChemFeaturizer.get_general_descriptor_subset()[:subset_size]
        else:
            raise ValueError(
                f'Unrecognised descriptor subset: {subset} (should be "all", "simple",'
                f'"uncorrelated", "fragment", "graph", "logp", "refractivity",'
                f'"estate", "druglikeness", "surface", "charge", "general").'
            )

    @property
    def output_size(self):
        return len(self.descriptors)

    def transform(self, molecules: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        features, valids = super().transform(molecules)

        return features, valids

    def transform_single(self, molecule: str) -> Tuple[np.ndarray, bool]:
        features, valid = super().transform_single(molecule)

        return features, valid

    def transform_mol(self, molecule: Chem.rdchem.Mol) -> Tuple[np.ndarray, bool]:
        fp = self.calc.CalcDescriptors(molecule)
        fp = np.array(fp)
        mask = np.isfinite(fp)
        fp[~mask] = 0
        fp = rdkit_dense_array_to_np(fp, dtype=float)
        if self.normalise:
            fp = self.scaler.transform_single(fp)
        return fp, True

    def is_valid_single(self, molecule: str) -> bool:
        _, valid = self.transform_single(molecule)
        return valid

    # control pickling / unpickling
    def __getstate__(self):
        return {
            'descriptors': self.descriptors,
            'fingerprint_extra_args': self.fingerprint_extra_args,
            'normalise': self.normalise,
        }

    def __setstate__(self, saved_dict):
        # ignore mypy check: calling __init__ directly as a form of reflection during unpickling (not called by default)
        self.__init__(  # type: ignore
            descriptors=saved_dict['descriptors'],
            fingerprint_extra_args=saved_dict['fingerprint_extra_args'],
            normalise=saved_dict['normalise'],
        )

    @staticmethod
    def get_all_descriptor_names() -> List[str]:
        """
        Get available descriptor names for RDKit physchem features. Custom subset can be used as list of descriptors.
        """
        return sorted([x[0] for x in Descriptors._descList])

    @staticmethod
    def get_simple_descriptor_subset() -> List[str]:
        return [
            'FpDensityMorgan2',
            'FractionCSP3',
            'MolLogP',
            'MolWt',
            'NumHAcceptors',
            'NumHDonors',
            'NumRotatableBonds',
            'TPSA',
        ]

    @staticmethod
    def get_refractivity_descriptor_subset() -> List[str]:
        return [
            'MolMR',
            'SMR_VSA1',
            'SMR_VSA10',
            'SMR_VSA2',
            'SMR_VSA3',
            'SMR_VSA4',
            'SMR_VSA5',
            'SMR_VSA6',
            'SMR_VSA7',
            'SMR_VSA8',
            'SMR_VSA9',
        ]

    @staticmethod
    def get_logp_descriptor_subset() -> List[str]:
        """LogP descriptors and VSA/LogP descriptors
        SlogP_VSA: VSA of atoms contributing to a specified bin of SlogP
        """

        return [
            'MolLogP',
            'SlogP_VSA1',
            'SlogP_VSA10',
            'SlogP_VSA11',
            'SlogP_VSA12',
            'SlogP_VSA2',
            'SlogP_VSA3',
            'SlogP_VSA4',
            'SlogP_VSA5',
            'SlogP_VSA6',
            'SlogP_VSA7',
            'SlogP_VSA8',
            'SlogP_VSA9',
        ]

    @staticmethod
    def get_graph_descriptor_subset() -> List[str]:
        """ Graph descriptors (https://www.rdkit.org/docs/source/rdkit.Chem.GraphDescriptors.html) """
        return [
            'BalabanJ',
            'BertzCT',
            'Chi0',
            'Chi0n',
            'Chi0v',
            'Chi1',
            'Chi1n',
            'Chi1v',
            'Chi2n',
            'Chi2v',
            'Chi3n',
            'Chi3v',
            'Chi4n',
            'Chi4v',
            'HallKierAlpha',
            'Ipc',
            'Kappa1',
            'Kappa2',
            'Kappa3',
        ]

    @staticmethod
    def get_surface_descriptor_subset() -> List[str]:
        """MOE-like surface descriptors
        EState_VSA: VSA (van der Waals surface area) of atoms contributing to a specified bin of e-state
        SlogP_VSA: VSA of atoms contributing to a specified bin of SlogP
        SMR_VSA: VSA of atoms contributing to a specified bin of molar refractivity
        PEOE_VSA: VSA of atoms contributing to a specified bin of partial charge (Gasteiger)
        LabuteASA: Labute's approximate surface area descriptor
        """
        return [
            'SlogP_VSA1',
            'SlogP_VSA10',
            'SlogP_VSA11',
            'SlogP_VSA12',
            'SlogP_VSA2',
            'SlogP_VSA3',
            'SlogP_VSA4',
            'SlogP_VSA5',
            'SlogP_VSA6',
            'SlogP_VSA7',
            'SlogP_VSA8',
            'SlogP_VSA9',
            'SMR_VSA1',
            'SMR_VSA10',
            'SMR_VSA2',
            'SMR_VSA3',
            'SMR_VSA4',
            'SMR_VSA5',
            'SMR_VSA6',
            'SMR_VSA7',
            'SMR_VSA8',
            'SMR_VSA9',
            'EState_VSA1',
            'EState_VSA10',
            'EState_VSA11',
            'EState_VSA2',
            'EState_VSA3',
            'EState_VSA4',
            'EState_VSA5',
            'EState_VSA6',
            'EState_VSA7',
            'EState_VSA8',
            'EState_VSA9',
            'LabuteASA',
            'PEOE_VSA1',
            'PEOE_VSA10',
            'PEOE_VSA11',
            'PEOE_VSA12',
            'PEOE_VSA13',
            'PEOE_VSA14',
            'PEOE_VSA2',
            'PEOE_VSA3',
            'PEOE_VSA4',
            'PEOE_VSA5',
            'PEOE_VSA6',
            'PEOE_VSA7',
            'PEOE_VSA8',
            'PEOE_VSA9',
            'TPSA',
        ]

    @staticmethod
    def get_druglikeness_descriptor_subset() -> List[str]:
        """ Descriptors commonly used to assess druglikeness"""
        return [
            'TPSA',
            'MolLogP',
            'MolMR',
            'ExactMolWt',
            'FractionCSP3',
            'HeavyAtomCount',
            'MolWt',
            'NHOHCount',
            'NOCount',
            'NumAliphaticCarbocycles',
            'NumAliphaticHeterocycles',
            'NumAliphaticRings',
            'NumAromaticCarbocycles',
            'NumAromaticHeterocycles',
            'NumAromaticRings',
            'NumHAcceptors',
            'NumHDonors',
            'NumHeteroatoms',
            'NumRotatableBonds',
            'NumSaturatedCarbocycles',
            'NumSaturatedHeterocycles',
            'NumSaturatedRings',
            'RingCount',
            'qed',
        ]

    @staticmethod
    def get_fragment_descriptor_subset() -> List[str]:
        return [
            'NHOHCount',
            'NOCount',
            'NumAliphaticCarbocycles',
            'NumAliphaticHeterocycles',
            'NumAliphaticRings',
            'NumAromaticCarbocycles',
            'NumAromaticHeterocycles',
            'NumAromaticRings',
            'NumHAcceptors',
            'NumHDonors',
            'NumHeteroatoms',
            'NumRotatableBonds',
            'NumSaturatedCarbocycles',
            'NumSaturatedHeterocycles',
            'NumSaturatedRings',
            'RingCount',
            'fr_Al_COO',
            'fr_Al_OH',
            'fr_Al_OH_noTert',
            'fr_ArN',
            'fr_Ar_COO',
            'fr_Ar_N',
            'fr_Ar_NH',
            'fr_Ar_OH',
            'fr_COO',
            'fr_COO2',
            'fr_C_O',
            'fr_C_O_noCOO',
            'fr_C_S',
            'fr_HOCCN',
            'fr_Imine',
            'fr_NH0',
            'fr_NH1',
            'fr_NH2',
            'fr_N_O',
            'fr_Ndealkylation1',
            'fr_Ndealkylation2',
            'fr_Nhpyrrole',
            'fr_SH',
            'fr_aldehyde',
            'fr_alkyl_carbamate',
            'fr_alkyl_halide',
            'fr_allylic_oxid',
            'fr_amide',
            'fr_amidine',
            'fr_aniline',
            'fr_aryl_methyl',
            'fr_azide',
            'fr_azo',
            'fr_barbitur',
            'fr_benzene',
            'fr_benzodiazepine',
            'fr_bicyclic',
            'fr_diazo',
            'fr_dihydropyridine',
            'fr_epoxide',
            'fr_ester',
            'fr_ether',
            'fr_furan',
            'fr_guanido',
            'fr_halogen',
            'fr_hdrzine',
            'fr_hdrzone',
            'fr_imidazole',
            'fr_imide',
            'fr_isocyan',
            'fr_isothiocyan',
            'fr_ketone',
            'fr_ketone_Topliss',
            'fr_lactam',
            'fr_lactone',
            'fr_methoxy',
            'fr_morpholine',
            'fr_nitrile',
            'fr_nitro',
            'fr_nitro_arom',
            'fr_nitro_arom_nonortho',
            'fr_nitroso',
            'fr_oxazole',
            'fr_oxime',
            'fr_para_hydroxylation',
            'fr_phenol',
            'fr_phenol_noOrthoHbond',
            'fr_phos_acid',
            'fr_phos_ester',
            'fr_piperdine',
            'fr_piperzine',
            'fr_priamide',
            'fr_prisulfonamd',
            'fr_pyridine',
            'fr_quatN',
            'fr_sulfide',
            'fr_sulfonamd',
            'fr_sulfone',
            'fr_term_acetylene',
            'fr_tetrazole',
            'fr_thiazole',
            'fr_thiocyan',
            'fr_thiophene',
            'fr_unbrch_alkane',
            'fr_urea',
        ]

    @staticmethod
    def get_estate_descriptor_subset() -> List[str]:
        """Electrotopological state (e-state) and VSA/e-state descriptors
        EState_VSA: VSA (van der Waals surface area) of atoms contributing to a specified bin of e-state
        VSA_EState: e-state values of atoms contributing to a specific bin of VSA
        """
        return [
            'EState_VSA1',
            'EState_VSA10',
            'EState_VSA11',
            'EState_VSA2',
            'EState_VSA3',
            'EState_VSA4',
            'EState_VSA5',
            'EState_VSA6',
            'EState_VSA7',
            'EState_VSA8',
            'EState_VSA9',
            'VSA_EState1',
            'VSA_EState10',
            'VSA_EState2',
            'VSA_EState3',
            'VSA_EState4',
            'VSA_EState5',
            'VSA_EState6',
            'VSA_EState7',
            'VSA_EState8',
            'VSA_EState9',
            'MaxAbsEStateIndex',
            'MaxEStateIndex',
            'MinAbsEStateIndex',
            'MinEStateIndex',
        ]

    @staticmethod
    def get_charge_descriptor_subset() -> List[str]:
        """
        Partial charge and VSA/charge descriptors
        PEOE: Partial Equalization of Orbital Electronegativities (Gasteiger partial atomic charges)
        PEOE_VSA: VSA of atoms contributing to a specific bin of partial charge
        """
        return [
            'PEOE_VSA1',
            'PEOE_VSA10',
            'PEOE_VSA11',
            'PEOE_VSA12',
            'PEOE_VSA13',
            'PEOE_VSA14',
            'PEOE_VSA2',
            'PEOE_VSA3',
            'PEOE_VSA4',
            'PEOE_VSA5',
            'PEOE_VSA6',
            'PEOE_VSA7',
            'PEOE_VSA8',
            'PEOE_VSA9',
            'MaxAbsPartialCharge',
            'MaxPartialCharge',
            'MinAbsPartialCharge',
            'MinPartialCharge',
        ]

    @staticmethod
    def get_general_descriptor_subset() -> List[str]:
        """ Descriptors from https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html """
        return [
            'MaxAbsPartialCharge',
            'MaxPartialCharge',
            'MinAbsPartialCharge',
            'MinPartialCharge',
            'ExactMolWt',
            'MolWt',
            'FpDensityMorgan1',
            'FpDensityMorgan2',
            'FpDensityMorgan3',
            'HeavyAtomMolWt',
            'NumRadicalElectrons',
            'NumValenceElectrons',
        ]

    @staticmethod
    def get_uncorrelated_descriptor_subset(subset_size: int) -> List[str]:
        """
        Column names are sorted starting with the non-informative descriptors, then the rest are ordered
        from most correlated to least correlated. This will return the n least correlated descriptors.

        Args:
            subset_size: how many to return

        Returns:
            List of descriptors
        """
        columns_sorted_by_correlation = [
            'fr_sulfone',
            'MinPartialCharge',
            'fr_C_O_noCOO',
            'fr_hdrzine',
            'fr_Ndealkylation2',
            'NumAromaticHeterocycles',
            'fr_N_O',
            'fr_piperdine',
            'fr_HOCCN',
            'fr_Nhpyrrole',
            'NumHAcceptors',
            'NumHeteroatoms',
            'fr_C_O',
            'VSA_EState5',
            'fr_Al_OH',
            'SlogP_VSA9',
            'fr_benzodiazepine',
            'VSA_EState6',
            'fr_Ar_N',
            'VSA_EState7',
            'fr_COO2',
            'VSA_EState3',
            'fr_Imine',
            'fr_sulfide',
            'FractionCSP3',
            'fr_imidazole',
            'fr_azo',
            'NumHDonors',
            'fr_COO',
            'fr_ether',
            'fr_nitro',
            'NumSaturatedHeterocycles',
            'fr_lactam',
            'fr_aniline',
            'NumAliphaticCarbocycles',
            'fr_para_hydroxylation',
            'SMR_VSA2',
            'MaxAbsPartialCharge',
            'fr_thiocyan',
            'NHOHCount',
            'fr_ester',
            'fr_aldehyde',
            'SMR_VSA8',
            'fr_halogen',
            'fr_NH0',
            'fr_furan',
            'fr_tetrazole',
            'HeavyAtomCount',
            'NumRotatableBonds',
            'NumSaturatedCarbocycles',
            'fr_SH',
            'fr_Ar_NH',
            'SlogP_VSA7',
            'fr_ketone',
            'fr_alkyl_halide',
            'fr_NH1',
            'NumRadicalElectrons',
            'MaxPartialCharge',
            'fr_ArN',
            'fr_imide',
            'fr_priamide',
            'fr_hdrzone',
            'fr_azide',
            'NumAromaticCarbocycles',
            'NOCount',
            'fr_isocyan',
            'RingCount',
            'fr_nitroso',
            'EState_VSA11',
            'MinAbsPartialCharge',
            'fr_Ar_COO',
            'fr_prisulfonamd',
            'fr_sulfonamd',
            'VSA_EState4',
            'fr_quatN',
            'fr_NH2',
            'fr_epoxide',
            'fr_allylic_oxid',
            'fr_piperzine',
            'VSA_EState1',
            'NumAliphaticHeterocycles',
            'fr_Ndealkylation1',
            'fr_Al_OH_noTert',
            'fr_aryl_methyl',
            'NumAromaticRings',
            'fr_bicyclic',
            'fr_methoxy',
            'fr_oxazole',
            'fr_barbitur',
            'NumAliphaticRings',
            'fr_Ar_OH',
            'fr_phos_ester',
            'fr_thiophene',
            'fr_nitrile',
            'fr_dihydropyridine',
            'VSA_EState2',
            'fr_nitro_arom',
            'SlogP_VSA11',
            'fr_thiazole',
            'fr_ketone_Topliss',
            'fr_term_acetylene',
            'fr_isothiocyan',
            'fr_urea',
            'fr_nitro_arom_nonortho',
            'fr_lactone',
            'fr_diazo',
            'fr_amide',
            'fr_alkyl_carbamate',
            'fr_Al_COO',
            'fr_amidine',
            'fr_phos_acid',
            'fr_oxime',
            'fr_guanido',
            'fr_C_S',
            'NumSaturatedRings',
            'fr_benzene',
            'fr_phenol',
            'fr_unbrch_alkane',
            'fr_phenol_noOrthoHbond',
            'fr_pyridine',
            'fr_morpholine',
            'MaxAbsEStateIndex',
            'ExactMolWt',
            'MolWt',
            'Chi0',
            'LabuteASA',
            'Chi0n',
            'NumValenceElectrons',
            'Chi3n',
            'Chi0v',
            'Chi3v',
            'Chi1',
            'Chi1n',
            'Chi1v',
            'FpDensityMorgan2',
            'HeavyAtomMolWt',
            'Kappa1',
            'SMR_VSA7',
            'Chi2n',
            'Chi2v',
            'Kappa2',
            'Chi4n',
            'SMR_VSA5',
            'MolMR',
            'EState_VSA10',
            'BertzCT',
            'MinEStateIndex',
            'SMR_VSA1',
            'FpDensityMorgan1',
            'VSA_EState10',
            'SlogP_VSA2',
            'SMR_VSA10',
            'HallKierAlpha',
            'VSA_EState9',
            'TPSA',
            'MaxEStateIndex',
            'Chi4v',
            'SMR_VSA4',
            'MolLogP',
            'qed',
            'VSA_EState8',
            'EState_VSA2',
            'SMR_VSA6',
            'PEOE_VSA1',
            'EState_VSA1',
            'SlogP_VSA8',
            'SlogP_VSA6',
            'SlogP_VSA5',
            'SlogP_VSA10',
            'BalabanJ',
            'Kappa3',
            'EState_VSA4',
            'PEOE_VSA6',
            'EState_VSA9',
            'PEOE_VSA2',
            'PEOE_VSA5',
            'SMR_VSA3',
            'SlogP_VSA3',
            'EState_VSA7',
            'EState_VSA3',
            'PEOE_VSA7',
            'SlogP_VSA1',
            'SMR_VSA9',
            'EState_VSA8',
            'EState_VSA6',
            'PEOE_VSA3',
            'MinAbsEStateIndex',
            'PEOE_VSA14',
            'FpDensityMorgan3',
            'PEOE_VSA12',
            'SlogP_VSA4',
            'PEOE_VSA9',
            'PEOE_VSA13',
            'PEOE_VSA10',
            'PEOE_VSA8',
            'EState_VSA5',
            'SlogP_VSA12',
            'PEOE_VSA4',
            'Ipc',
            'PEOE_VSA11',
        ]

        return columns_sorted_by_correlation[-subset_size:]

    @staticmethod
    def _get_descriptor_list(
        named_descriptor_set: str = 'all', descriptor_list: List[str] = [], subset_size: int = 200
    ):
        if len(descriptor_list) == 0:
            descriptor_list = PhysChemFeaturizer.get_descriptor_subset(named_descriptor_set, subset_size)
        else:  # else use the named_descriptor_set given by the user
            assert isinstance(descriptor_list, list)

            all_descriptors = set(PhysChemFeaturizer.get_all_descriptor_names())
            assert set(descriptor_list).issubset(all_descriptors)

        descriptor_list.sort()

        return descriptor_list
