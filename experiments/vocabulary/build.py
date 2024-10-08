""" This script builds a vocabulary given a dataset and a tokenizer. """

import os
import argparse

from tqdm import tqdm

from jointformer.configs.dataset import DatasetConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.smiles.regex import RegexSmilesTokenizer
from jointformer.utils.runtime import save_strings_to_file

VOCABULARY_DIR = './data/vocabularies'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_task_config", type=str, required=True)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)

    dataset = AutoDataset.from_config(dataset_config, split='all')
    print(f"Number of SMILES strings: {len(dataset)}")
    tokenizer = RegexSmilesTokenizer()

    vocabulary = set()
    for idx, x in enumerate(tqdm(dataset, desc='Extracting all tokens')):
        tokens = tokenizer.tokenize(x)
        vocabulary.update(set(tokens))

    out_dir = os.path.join(VOCABULARY_DIR, f'{task_config.dataset_name.lower()}.txt')
    save_strings_to_file(list(vocabulary), out_dir)
    print(f'Vocabulary saved to {out_dir}')


if __name__ == "__main__":
    main()
