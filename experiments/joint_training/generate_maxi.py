import os
import torch
import sys
import time

top_level_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(top_level_folder)
print(f"Appended {top_level_folder} to PATH.")

from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig

from jointformer.utils.tokenizers.auto import AutoTokenizer

from jointformer.models.auto import AutoModel
from jointformer.models.jointformer import Jointformer


REPOSITORY_DIR = "/home/maxi/code/jointformer"
OUT_DIR_BASE = "/home/maxi/code/jf-data"
os.chdir(REPOSITORY_DIR)

PRETRAINED_JOINTFORMER_FILE = f"{OUT_DIR_BASE}/ckpt.pt"
DATA_FILE = f"{OUT_DIR_BASE}/data/guacamol/val/smiles.txt"
OUTPUT_DIR = f"{OUT_DIR_BASE}/out-generate"

PATH_TO_TOKENIZER_CONFIG = f"{REPOSITORY_DIR}/configs/tokenizers/smiles_separate_task_token"
PATH_TO_MODEL_CONFIG = f"{REPOSITORY_DIR}/configs/models/jointformer_separate_task_token"

smiles_data = open(DATA_FILE, "r").readlines()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

tokenizer_config = TokenizerConfig.from_config_file(PATH_TO_TOKENIZER_CONFIG)
tokenizer = AutoTokenizer.from_config(tokenizer_config)

model_config = ModelConfig.from_config_file(PATH_TO_MODEL_CONFIG)
model: Jointformer = AutoModel.from_config(model_config)

# Cheating a bit here
batch_size = 128
model.update_batch_size(128)
model.load_pretrained(PRETRAINED_JOINTFORMER_FILE)
model.update_training_mode(False)

temperature = 0.8
top_k = 10

generator = model.to_guacamole_generator(tokenizer=tokenizer, batch_size=batch_size, temperature=temperature, top_k=top_k, device=device)
out_smiles = generator.generate(number_samples=10)

filename = f"gen-output_{time.strftime('%d_%m-%H_%M_%S')}"
OUTFILE = os.path.join(OUTPUT_DIR, filename)
open(OUTFILE, "w").writelines(out_smiles)
print(f"Wrote {len(out_smiles)} lines to '{OUTFILE}'.")
