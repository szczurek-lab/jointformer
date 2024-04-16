import torch

from typing import List
from guacamol.assess_distribution_learning import DistributionMatchingGenerator


class GuacamolModelWrapper(DistributionMatchingGenerator):

    def __init__(self, model, tokenizer, batch_size, device, temperature=1.0, top_k=0):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.temperature = temperature
        self.top_k = None if top_k is None else top_k if top_k > 0 else None
        self._eval()

    def _eval(self):
        self.model.eval()

    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        generated = []

        while len(generated) < number_samples:
            idx = torch.ones(size=(self.batch_size, 1), device=self.device) * self.tokenizer.generate_token_id
            idx = idx.long()
            samples = self.model.generate(
                idx=idx, max_new_tokens=self.tokenizer.max_molecule_length,
                temperature=self.temperature, top_k=self.top_k)
            generated += self.tokenizer.decode(samples)
        return generated[:number_samples]

    @classmethod
    def init_from_trainer(cls, trainer, temperature=1.0, top_k=0):
        return cls(
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            batch_size=trainer.batch_size,
            device=trainer.device,
            temperature=temperature,
            top_k=top_k)
