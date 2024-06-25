""" Jointformer without Hybrid Attention and alternating gradient updates.
 """

import torch
import torch.nn.functional as F

from typing import Optional

from jointformer.utils.tokenizers.smiles.smiles import IGNORE_INDEX
from jointformer.models.jointformer import Jointformer, DEFAULT_NUM_PHYCHEM_TASKS


class JointTransformer(Jointformer):

    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            embedding_dim: int,
            dropout: float,
            num_layers: int,
            bias: int,
            num_heads: int,
            num_prediction_tasks: int,
            layer_norm_eps: float,
            num_physchem_tasks: Optional[int] = DEFAULT_NUM_PHYCHEM_TASKS,
            init_weights: bool = True,
            tie_weights: bool = True
    ):

        super().__init__(
            vocab_size=vocab_size, max_seq_len=max_seq_len, embedding_dim=embedding_dim,
            dropout=dropout, num_layers=num_layers, bias=bias, num_heads=num_heads, layer_norm_eps=layer_norm_eps, num_prediction_tasks=num_prediction_tasks,
            num_physchem_tasks=num_physchem_tasks, init_weights=init_weights, tie_weights=tie_weights)

    def get_loss(
                self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                properties: Optional[torch.Tensor] = None,
                task: Optional[str] = None,
                alpha: Optional[float] = 0.33):

        outputs = super().forward(input_ids=input_ids, attention_mask=None, is_causal=True)
        outputs["loss"] = None
        outputs["logits"] = self.lm_head(outputs['embeddings'])
        if labels is not None:
            input = outputs['logits'][:, :-1, :].contiguous()
            target = labels[:, 1:].contiguous()
            outputs["loss"] = F.cross_entropy(
                input.view(-1, input.size(-1)), target.view(-1), ignore_index=IGNORE_INDEX, reduction='mean')

        if properties is not None:
            if task == 'lm' or task == 'physchem':         
                y_pred = self.physchem_head(outputs['embeddings'].flatten(start_dim=1, end_dim=-1))    
            elif task == 'prediction' and properties is not None:
                y_pred = self.prediction_head(outputs['embeddings'].flatten(start_dim=1, end_dim=-1))
            else:
                raise ValueError('Variable `task` must be either `lm`, `prediction` or `physchem`.')
            outputs["loss"] += alpha * F.mse_loss(y_pred.flatten(), properties.flatten(), reduction='mean')

        return outputs

    def predict(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs):
        """ Perform a forward pass through the discriminative part of the model. """
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, is_causal=True)
        outputs["loss"] = None
        outputs["logits"] = self.prediction_head(outputs['embeddings'].flatten(start_dim=1, end_dim=-1))
        outputs["y_pred"] = outputs["logits"]
        return outputs
    