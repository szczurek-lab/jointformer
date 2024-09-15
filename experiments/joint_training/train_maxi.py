import os, logging, sys, argparse, time

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_repo_dir", "-r", type=str, nargs='?', required=True)
    parser.add_argument("--path_to_out_dir", "-o", type=str, nargs='?', required=True)
    args = parser.parse_args()
    if not os.path.exists(args.path_to_out_dir):
        os.makedirs(args.path_to_out_dir)
    return args


def setup_default_logging():
    console = logging.getLogger(__file__)
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join("logs", f'{os.environ.get("SLURM_JOB_NAME", "run")}.log'),
        filemode='a',
        format=f'{gethostname()}, rank {int(os.environ.get("SLURM_PROCID", "0"))}: %(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.captureWarnings(True)
    return console


@record
def main(seed, repo_dir, out_dir):
    model_ckpt = f"{out_dir}/ckpt.pt"
    data_dir = f"{out_dir}/data"

    dataset_config = DatasetConfig.from_config_file(f"{repo_dir}/configs/datasets/guacamol/unsupervised")
    tokenizer_config = TokenizerConfig.from_config_file(f"{repo_dir}/configs/tokenizers/smiles_separate_task_token")
    model_config = ModelConfig.from_config_file(f"{repo_dir}/configs/models/jointformer_separate_task_token")
    trainer_config = TrainerConfig.from_config_file(f"{repo_dir}/configs/trainers/maxi_test")
    logger_config = LoggerConfig.from_config_file(f"{repo_dir}/configs/loggers/maxi")

    train_dataset = AutoDataset.from_config(dataset_config, split='train', data_dir=data_dir)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', data_dir=data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    model: Jointformer = AutoModel.from_config(model_config)
    logger = AutoLogger.from_config(logger_config) if logger_config else None

    dump_configs(out_dir, dataset_config, tokenizer_config, model_config, trainer_config, logger_config) 
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config) 

    init_ddp(trainer_config.enable_ddp)
    
    model.update_batch_size(trainer_config.batch_size)
    model.update_training_mode(True)
    
    trainer = Trainer(
        out_dir=out_dir,
        seed=seed,
        config=trainer_config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        logger=logger)
    
    console = setup_default_logging()
    try:
        trainer.resume_snapshot()
        console.info("Resumed Snapshot")
    except FileNotFoundError:
        if model_ckpt:
            try:
                trainer.resume_from_file(model_ckpt)
                console.info(f"Resuming pre-trained model from '{model_ckpt}'")
            except FileNotFoundError:
                console.info(f"No model checkpoint at '{model_ckpt}'")
        else:
            console.info("Training from scratch")
    if trainer.is_ddp:
        dist.barrier() # Ensure all processes are ready before training
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    trainer._save_ckpt(model_ckpt)
    end_ddp(trainer_config.enable_ddp)
    logger.log({"Execution Time (h)": (end_time - start_time)/3600})
    logger.finish()


if __name__ == "__main__":
    args = parse_args()
    repo_dir = args.path_to_repo_dir
    out_dir = args.path_to_out_dir
    os.chdir(repo_dir)
    seed = 1337
    set_seed(seed)
    create_output_dir(os.path.join(out_dir, f"seed_{seed}"))
    try:
        main(seed, repo_dir, out_dir)
        logging.info(f"Completed seed {seed}")
    except Exception as e:
        logging.critical(e, exc_info=True)
