
import numpy as np

from typing import Union
from rdkit import Chem
from rdkit.Chem.QED import qed


def predict_qed(smiles: str) -> Union[float, None]:
    try:
        return qed(Chem.MolFromSmiles(smiles, sanitize=False))
    except:
        return np.nan
