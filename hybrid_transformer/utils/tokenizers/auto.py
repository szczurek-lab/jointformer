import importlib

from hybrid_transformer.configs.tasks.distribution_learning import DistributionLearningConfig


class AutoTokenizer:
    """Base Tokenizer. """

    @classmethod
    def from_config(cls, config: DistributionLearningConfig) -> "AutoTokenizer":

        if config.tokenizer not in ["SMILESTokenizer"]:
            raise ValueError(f"`config.tokenizer` must be 'SMILESTokenizer', got {config.tokenizer}.")

        if config.tokenizer == 'SMILESTokenizer':
            return getattr(importlib.import_module(
                "hybrid_transformer.utils.tokenizers.smiles"),
                "SMILESTokenizer").from_config(config)
        else:
            raise ValueError(f"Invalid `tokenizer`.")
