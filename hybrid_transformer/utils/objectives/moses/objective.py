"""Get objective values. """

import torch
import moses

from typing import List

from hybrid_transformer.utils.objectives.guacamol.utils import smiles_to_desired_scores

DTYPE_OBJECTIVE = torch.float32


def get_objective(smiles: List[str], objective: str = None) -> torch.Tensor:
    objectives = moses.get_all_metrics(smiles)
    if objective:
        return objectives[objective]
    else:
        return objectives
