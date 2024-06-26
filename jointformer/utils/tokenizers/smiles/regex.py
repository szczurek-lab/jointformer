""" Reimplements the BasicSmilesTokenizer class from deepchem.feat.smiles_tokenizer.
 Source: https://deepchem.readthedocs.io/en/2.4.0/api_reference/tokenizers.html

The tokenizer encodes SMILES strings using the tokenization SMILES regex developed by [1].

    References
    ----------
    [1]  Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and
     Alpha A. Lee. ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated
      Chemical Reaction Prediction 1572-1583 DOI: 10.1021/acscentsci.9b00576.
"""

import re
from typing import List

REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


class RegexSmilesTokenizer:
    """ Tokenization of SMILES strings based on a regex pattern developed by [1].
    """

    def __init__(self, regex_pattern: str = REGEX_PATTERN):
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text):
        tokens = list(self.regex.findall(text))
        reconstructed = "".join(tokens)
        if reconstructed != text:
            raise ValueError(f"Tokenization failed. Original: {tokens}, Reconstructed: {reconstructed}")
        return tokens
