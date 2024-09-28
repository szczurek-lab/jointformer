import torch, re
import numpy as np

from guacamol.assess_distribution_learning import DistributionMatchingGenerator
from jointformer.models.base import SmilesEncoder, DistributionMatchingGenerator
from jointformer.utils.tokenizers.auto import SmilesTokenizer
from tqdm import tqdm
from typing import List

from jointformer.models.utils import ModelOutput


class DefaultSmilesGeneratorWrapper(DistributionMatchingGenerator):
    def __init__(self, model, tokenizer, batch_size, temperature, top_k, device):
        self._model = model
        self._tokenizer: SmilesTokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        self._temperature = temperature
        self._top_k = top_k

    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        generated = []
        self._model.eval()
        model = self._model.to(self._device)
        for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
            samples: list[str] = model.generate(self._tokenizer.cls_token_id,
                                        self._tokenizer.sep_token_id,
                                        self._tokenizer.pad_token_id,
                                        self._tokenizer.max_molecule_length,
                                        self._batch_size,
                                        self._temperature,
                                        self._top_k,
                                        self._device)
            generated.extend(self._tokenizer.decode(samples))
        return generated[:number_samples]


class JointformerSmilesGeneratorWrapper(DefaultSmilesGeneratorWrapper):
    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        generated = []
        self._model.eval()
        model = self._model.to(self._device)
        for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
            samples: list[str] = model.generate(self._tokenizer, self._batch_size, self._temperature, self._top_k, self._device)
            generated.extend(self._tokenizer.decode(samples))
        return generated[:number_samples]
    

class MolGPTSmilesGeneratorWrapper(DefaultSmilesGeneratorWrapper):
    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        # https://github.com/devalab/molgpt/blob/main/generate/generate.py#L109
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        stoi = {"#": 0, "%10": 1, "%11": 2, "%12": 3, "(": 4, ")": 5, "-": 6, "1": 7, "2": 8, "3": 9, "4": 10, "5": 11, "6": 12, "7": 13, "8": 14, "9": 15, "<": 16, "=": 17, "B": 18, "Br": 19, "C": 20, "Cl": 21, "F": 22, "I": 23, "N": 24, "O": 25, "P": 26, "S": 27, "[B-]": 28, "[BH-]": 29, "[BH2-]": 30, "[BH3-]": 31, "[B]": 32, "[C+]": 33, "[C-]": 34, "[CH+]": 35, "[CH-]": 36, "[CH2+]": 37, "[CH2]": 38, "[CH]": 39, "[F+]": 40, "[H]": 41, "[I+]": 42, "[IH2]": 43, "[IH]": 44, "[N+]": 45, "[N-]": 46, "[NH+]": 47, "[NH-]": 48, "[NH2+]": 49, "[NH3+]": 50, "[N]": 51, "[O+]": 52, "[O-]": 53, "[OH+]": 54, "[O]": 55, "[P+]": 56, "[PH+]": 57, "[PH2+]": 58, "[PH]": 59, "[S+]": 60, "[S-]": 61, "[SH+]": 62, "[SH]": 63, "[Se+]": 64, "[SeH+]": 65, "[SeH]": 66, "[Se]": 67, "[Si-]": 68, "[SiH-]": 69, "[SiH2]": 70, "[SiH]": 71, "[Si]": 72, "[b-]": 73, "[bH-]": 74, "[c+]": 75, "[c-]": 76, "[cH+]": 77, "[cH-]": 78, "[n+]": 79, "[n-]": 80, "[nH+]": 81, "[nH]": 82, "[o+]": 83, "[s+]": 84, "[sH+]": 85, "[se+]": 86, "[se]": 87, "b": 88, "c": 89, "n": 90, "o": 91, "p": 92, "s": 93}
        context = "C"
        generated = []
        self._model.eval()
        model = self._model.to(self._device)
        for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
            x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(self._batch_size, 1).to('cuda')
            block_size = model.get_block_size() 
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            samples: list[dict] = model.forward(x_cond)
            generated.extend(samples)
        return generated[:number_samples]    

    
class DefaultSmilesEncoderWrapper(SmilesEncoder):
    def __init__(self, model, tokenizer, batch_size, device):
        self._model = model
        self._tokenizer: SmilesTokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device

    @torch.no_grad()
    def encode(self, smiles: list[str]) -> np.ndarray:
        self._model.eval()
        model = self._model.to(self._device)
        embeddings = []
        for i in tqdm(range(0, len(smiles), self._batch_size), "Encoding samples"):
            batch = smiles[i:i+self._batch_size]
            batch_input = self._tokenizer(batch, task="prediction")
            for k,v in batch_input.items():
                if isinstance(v, torch.Tensor):
                    batch_input[k] = v.to(self._device)
            output: ModelOutput = model(**batch_input, is_causal=False)
            embeddings.append(output["global_embeddings"].cpu().numpy())
        ret = np.concatenate(embeddings, axis=0)
        return ret


class JointformerSmilesEncoderWrapper(DefaultSmilesEncoderWrapper):

    @torch.no_grad()
    def encode(self, smiles: list[str]) -> np.ndarray:
        self._model.eval()
        model = self._model.to(self._device)
        embeddings = np.zeros((len(smiles), model.embedding_dim))
        for i in tqdm(range(0, len(smiles), self._batch_size), "Encoding samples"):
            batch = smiles[i:i+self._batch_size]
            model_input = self._tokenizer(batch, task="prediction")
            model_input.to(self._device)
            output: ModelOutput = model(**model_input)
            embeddings[i:i+self._batch_size] = output.global_embeddings.cpu().numpy()
        return embeddings
    