import os, sys
import logging
import argparse

import torch.distributed as dist
import numpy as np

from socket import gethostname
from torch.distributed.elastic.multiprocessing.errors import record

from jointformer.configs.dataset import DatasetConfig
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from jointformer.utils.loggers.auto import AutoLogger

from jointformer.trainers.trainer import Trainer

from jointformer.utils.runtime import set_seed, create_output_dir, set_to_dev_mode, log_args, dump_configs
from jointformer.utils.ddp import init_ddp, end_ddp
from jointformer.utils.data import write_dict_to_file

# from experiments.joint_training.train import main as joint_training_main

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    filename=f"{os.environ.get('SLURM_JOB_NAME', 'run')}.log",
    filemode='a',
    format=f'{gethostname()}, rank {int(os.environ.get("SLURM_PROCID", 0))}: %(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.captureWarnings(True)

FRACTION_TRAINING_EXAMPLES = [0.01, 0.1, 0.25, 1.0]
MODEL_SEED_ARRAY = [1337]
DATA_SEED_ARRAY = [0, 1, 2]
DEFAULT_NUM_EPOCHS = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default='results')
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--data_seed_array", type=int, nargs='*', default=DATA_SEED_ARRAY)
    parser.add_argument("--model_seed_array", type=int, nargs='*', default=MODEL_SEED_ARRAY)
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--fraction_training_examples", type=float, default=1.)
    parser.add_argument("--prepare_data", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry_run", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args

@record
def main(args):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    ###
    # Create out_dir and log configs
    ###
    tmp_out_dir = os.path.join(args.out_dir, f'seed_{args.data_seed}', f'fraction_training_examples_{args.fraction_training_examples}')
    if not os.path.exists(tmp_out_dir) and not args.prepare_data:
        os.makedirs(tmp_out_dir, exist_ok=False)
    
    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_file(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_config_file(args.path_to_logger_config) if args.path_to_logger_config else None

    if args.test:
        console.info("Running in test mode")
        trainer_config.max_iters = 2
        trainer_config.batch_size = 2
        trainer_config.eval_iters = 1
        trainer_config.log_interval = 1
        trainer_config.eval_interval = 2

    dump_configs(args.out_dir, dataset_config, tokenizer_config, model_config, trainer_config, logger_config)
    ###

    ###
    # Initialize
    ###
    set_seed(seed=args.data_seed)
    train_dataset = AutoDataset.from_config(dataset_config, split='train', data_dir=args.data_dir)
    num_subsamples =  int(len(train_dataset) * args.fraction_training_examples)

    train_dataset._subset(num_samples=num_subsamples, seed=args.data_seed)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', data_dir=args.data_dir)
   
    if not args.test:
        trainer_config.correct_for_num_train_examples(num_train_examples=len(train_dataset))
    console.info(f"Selected Train: {len(train_dataset)} examples")

    if args.prepare_data:
        sys.exit()

    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    set_seed(seed=args.model_seed)
    model = AutoModel.from_config(model_config)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config)

    trainer = Trainer(
        out_dir=tmp_out_dir,
        seed=args.model_seed,
        config=trainer_config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        logger=logger
        )
    console.info(f"Max iters is set to: {trainer.max_iters}")

    if args.path_to_model_ckpt:
        trainer.resume_from_file(args.path_to_model_ckpt)
        console.info(f"Resuming pre-trained model from {args.path_to_model_ckpt}")
    else:
        console.info("Training from scratch")
    
    if args.dry_run:
        console.info("Dry run finished!")
        return 0.0

    trainer.train()
    return None


if __name__ == "__main__":
    args = parse_args()
    log_args(args)
    data_seed_array = args.data_seed_array
    for fraction_training_examples in FRACTION_TRAINING_EXAMPLES:
        args.fraction_training_examples = fraction_training_examples
        for data_seed in data_seed_array:
            args.data_seed = data_seed
            args.model_seed = args.model_seed_array[0]
            console.info(f"Script running for fraction {args.fraction_training_examples} and seed {args.data_seed}...")
            main(args)
    console.info("Script finished!")
