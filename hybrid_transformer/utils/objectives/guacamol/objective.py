"""Get objective values. """

import torch

from typing import List
from tqdm import tqdm

from hybrid_transformer.utils.objectives.guacamol.utils import smiles_to_desired_scores
from hybrid_transformer.utils.objectives.objective import predict_qed

DTYPE_OBJECTIVE = torch.float32

GUACAMOL_TASKS = [
    # 'zaleplon',
    # 'osimertinib',
    # 'fexofenadine',
    # 'ranolazine',
    # 'perindopril',
    # 'amlodipine',
    # 'sitagliptin',
    # 'qed',
    'qed_classification',
]


def get_objective(smiles_list: List[str], objective: str, verbose=False) -> torch.Tensor:
    if objective == 'qed':
        return torch.Tensor([predict_qed(smile) for smile in tqdm(smiles_list)]).to(DTYPE_OBJECTIVE).unsqueeze(1)
    elif objective == 'qed_classification':
        objective_values = torch.Tensor([predict_qed(smile) for smile in tqdm(smiles_list)]).to(DTYPE_OBJECTIVE).unsqueeze(1)
        return torch.where(objective_values >= 0.9, 1, 0)
    else:
        return torch.Tensor(smiles_to_desired_scores(smiles_list, objective, verbose)).to(DTYPE_OBJECTIVE).unsqueeze(1)
