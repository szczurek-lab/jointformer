import os
import json
import argparse

from guacamol.assess_distribution_learning import assess_distribution_learning

from hybrid_transformer.configs.task import TaskConfig
from hybrid_transformer.configs.model import ModelConfig
from hybrid_transformer.configs.trainer import TrainerConfig
from hybrid_transformer.models.auto import AutoModel
from hybrid_transformer.models.utils import GuacamolModelWrapper
from hybrid_transformer.trainers.trainer import Trainer
from hybrid_transformer.utils.tokenizers.auto import AutoTokenizer
from hybrid_transformer.utils.objectives.moses.objective import get_objective

from scripts.joint_learning.train import DEFAULT_CONFIG_FILES
from scripts.pretrain.generate import generate

from scripts.pretrain.train import DEFAULT_CONFIG_FILES

DEFAULT_REFERENCE_FILE = './data/guacamol/train/smiles.txt'
FINE_TUNE_REFERENCE_FILE = './data/guacamol/train_10000/smiles.txt'
BATCH_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", nargs='?', type=int, default=0)
    parser.add_argument("-task", "--path_to_task_config", type=str, default=DEFAULT_CONFIG_FILES['task'])
    parser.add_argument("-model", "--path_to_model_config", type=str, default=DEFAULT_CONFIG_FILES['model'])
    parser.add_argument("-trainer", "--path_to_trainer_config", type=str, default=DEFAULT_CONFIG_FILES['trainer'])
    parser.add_argument("--reference_file", type=str, default=DEFAULT_REFERENCE_FILE)
    args = parser.parse_args()
    return args


def evaluate_distribution_learning_moses(trainer):
    num_samples = 30000
    samples = generate(trainer=trainer, num_samples=num_samples)
    metrics = get_objective(samples)
    with open(os.path.join(trainer.out_dir, "moses_distribution_learning_results.json"), 'w') as f:
        json.dump(metrics, f)
    return metrics


def evaluate_distribution_learning_guacamol(trainer, reference_file, temperature=1.0, top_k=None, filename=None):
    if filename is None:
        filename = "guacamol_distribution_learning_results"
        if temperature is not None:
            filename = filename + '_temperature_' + str(temperature)
        if top_k is not None:
            filename = filename + '_top_k_' + str(top_k)
        filename = filename + '.json'
        filename = os.path.join(trainer.out_dir, filename)
        print("Printing to:", filename)
    trainer._train_init()
    try:
        assess_distribution_learning(
            model=GuacamolModelWrapper(
                trainer.model, trainer.tokenizer, BATCH_SIZE, trainer.device, temperature=temperature, top_k=top_k),
            chembl_training_file=reference_file,
            json_output_file=filename,
            benchmark_version='v2',
        )

        with open(filename) as json_data:
            data = json.load(json_data)
        results = {
            "validity": data['results'][0]['score'],
            "uniqueness": data['results'][1]['score'],
            "novelty": data['results'][2]['score'],
            "kl_div": data['results'][3]['score'],
            "fcd": data['results'][4]['score'],
        }
    except:
        results = {
            "validity": 0.0,
            "uniqueness": 0.0,
            "novelty": 0.0,
            "kl_div": 0.0,
            "fcd": 0.0,
        }
    return results


def main():

    # Args
    args = parse_args()

    # Configs
    task_config = TaskConfig.from_pretrained(args.path_to_task_config)
    model_config = ModelConfig.from_pretrained(args.path_to_model_config)
    trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)

    # Override
    trainer_config.out_dir = args.out_dir
    print("Out dir: {}".format(trainer_config.out_dir))

    # Init
    tokenizer = AutoTokenizer.from_config(task_config)
    model = AutoModel.from_config(model_config)
    trainer = Trainer(config=trainer_config, model=model, tokenizer=tokenizer)

    # Load
    if os.path.isfile(os.path.join(trainer.out_dir, 'ckpt.pt')):
        trainer.load_checkpoint()
        if task_config.dataset_name == 'moses':
            evaluate_distribution_learning_moses(trainer=trainer)
        elif task_config.dataset_name == 'guacamol':
            print(f"Evaluating Guacamol distribution learning {args.temperature} and {args.top_k}")
            evaluate_distribution_learning_guacamol(
                trainer=trainer, reference_file=args.reference_file, temperature=args.temperature, top_k=args.top_k)
        else:
            raise ValueError("Invalid task name: {}".format(task_config.task_name))
    else:
        raise FileNotFoundError("No checkpoint file in {}".format(trainer.out_dir))


if __name__ == "__main__":
    main()
