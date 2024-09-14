import os, logging, sys

import torch.distributed as dist

from socket import gethostname
from torch.distributed.elastic.multiprocessing.errors import record

top_level_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(top_level_folder)
print(f"Appended {top_level_folder} to PATH.")

from jointformer.configs.dataset import DatasetConfig
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.utils.loggers.auto import AutoLogger
from jointformer.utils.loggers.wandb import WandbLogger
from jointformer.utils.runtime import set_seed, create_output_dir, dump_configs
from jointformer.utils.ddp import init_ddp, end_ddp

from jointformer.models.auto import AutoModel
from jointformer.models.jointformer import Jointformer

from jointformer.trainers.trainer import Trainer


REPOSITORY_DIR = "/home/maxi/code/jointformer"
OUT_DIR_BASE = "/home/maxi/code/jf-data"

MODEL_CHECKPOINT = f"{OUT_DIR_BASE}/ckpt.pt"
DATA_DIR = f"{OUT_DIR_BASE}/data"
OUTPUT_DIR = f"{OUT_DIR_BASE}/out-train"

PATH_TO_DATASET_CONFIG = f"{REPOSITORY_DIR}/configs/datasets/guacamol/unsupervised"
PATH_TO_TOKENIZER_CONFIG = f"{REPOSITORY_DIR}/configs/tokenizers/smiles_separate_task_token"
PATH_TO_MODEL_CONFIG = f"{REPOSITORY_DIR}/configs/models/jointformer_separate_task_token"
PATH_TO_TRAINER_CONFIG = f"{REPOSITORY_DIR}/configs/trainers/maxi_test"
PATH_TO_LOGGER_CONFIG = f"{REPOSITORY_DIR}/configs/loggers/maxi"

DEFAULT_MODEL_SEED_ARRAY = 1337

def setup_default_logging():
    console = logging.getLogger(__file__)
    logging.basicConfig(
        level=logging.INFO,
        filename=f'{os.environ.get("SLURM_JOB_NAME", "run")}.log',
        filemode='a',
        format=f'{gethostname()}, rank {int(os.environ.get("SLURM_PROCID", "0"))}: %(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.captureWarnings(True)
    return console

def get_add_logger() -> WandbLogger: 
    additional_logger = WandbLogger(enable_logging=True, user="maximilian-armuss-tum", project="jointformer-training", resume="allow", watch=True, watch_freq=100, display_name="Observe Loss")
    additional_logger.set_run_id()
    return additional_logger

@record
def main():
    console = setup_default_logging()
    additional_logger = get_add_logger()

    dataset_config = DatasetConfig.from_config_file(PATH_TO_DATASET_CONFIG)
    tokenizer_config = TokenizerConfig.from_config_file(PATH_TO_TOKENIZER_CONFIG)
    model_config = ModelConfig.from_config_file(PATH_TO_MODEL_CONFIG)
    trainer_config = TrainerConfig.from_config_file(PATH_TO_TRAINER_CONFIG)
    logger_config = LoggerConfig.from_config_file(PATH_TO_LOGGER_CONFIG) if PATH_TO_LOGGER_CONFIG else None

    train_dataset = AutoDataset.from_config(dataset_config, split='train', data_dir=DATA_DIR)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', data_dir=DATA_DIR)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    model: Jointformer = AutoModel.from_config(model_config)
    logger = AutoLogger.from_config(logger_config) if logger_config else None

    dump_configs(OUTPUT_DIR, dataset_config, tokenizer_config, model_config, trainer_config, logger_config) 
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config) 

    init_ddp(trainer_config.enable_ddp)
    model.update_batch_size(trainer_config.batch_size)
    model.update_training_mode(True)
    trainer = Trainer(
        out_dir=OUTPUT_DIR,
        seed=DEFAULT_MODEL_SEED_ARRAY,
        config=trainer_config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        logger=None)
    trainer.add_logger(additional_logger)
    
    
    try:
        trainer.resume_snapshot()
        console.info("Resumed Snapshot")
    except FileNotFoundError:
        if MODEL_CHECKPOINT:
            try:
                trainer.resume_from_file(MODEL_CHECKPOINT)
                console.info(f"Resuming pre-trained model from '{MODEL_CHECKPOINT}'")
            except FileNotFoundError:
                console.info(f"No model checkpoint at '{MODEL_CHECKPOINT}'")
        else:
            console.info("Training from scratch")
    if trainer.is_ddp:
        dist.barrier() # Ensure all processes are ready before training
        
    trainer.train()
    trainer._save_ckpt(MODEL_CHECKPOINT)
    end_ddp(trainer_config.enable_ddp)
    additional_logger.finish()


if __name__ == "__main__":
    os.chdir(REPOSITORY_DIR)
    create_output_dir(os.path.join(OUTPUT_DIR, f"seed_{DEFAULT_MODEL_SEED_ARRAY}"))
    set_seed(DEFAULT_MODEL_SEED_ARRAY)
    try:
        main()
        logging.info(f"Completed seed {DEFAULT_MODEL_SEED_ARRAY}")
    except Exception as e:
        logging.critical(e, exc_info=True)

    print(open(os.path.join(REPOSITORY_DIR, "run.log"), "r").read())
    