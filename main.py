import argparse
import datetime
import json
import os.path
from typing import Dict

import torch.nn
from transformers import AutoTokenizer

import config
import trainer
from config import Config
from output_utils import print_title


def create_config(args: argparse.Namespace) -> Config:
    if args.roberta:
        base_model = 'roberta-base'
    elif args.electra:
        base_model = 'google/electra-small-discriminator'
    else:
        base_model = 'distilbert-base-uncased'

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if args.tanh:
        activation_func = torch.nn.Tanh()
    else:
        activation_func = torch.nn.ReLU()

    c = Config(
        base_model=base_model,
        tokenizer=tokenizer,
        activation_func=activation_func,
        train_file=args.train_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        use_ancestors=not args.no_ancestors,
        max_ancestors=args.max_ancestors,
        used_ancestors=min(args.max_ancestors, args.used_ancestors),
        ancestors_direction_start=not args.direction_end,
        epochs=args.epochs
    )

    return c


def build_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SARCBert model for sarcasm detection')

    # Add the models handler - default is distilbert
    models_group = parser.add_mutually_exclusive_group()
    models_group.add_argument('--roberta', action='store_true')
    models_group.add_argument('--electra', action='store_true')

    # Add the activation function handler - default is ReLU
    activation_group = parser.add_mutually_exclusive_group()
    activation_group.add_argument('--tanh', action='store_true')

    # train & test files
    parser.add_argument('--train_file', type=str, action='store', default=r'data/train-pol-balanced.json')
    parser.add_argument('--test_file', type=str, action='store', default=r'data/test-pol-balanced.json')

    # Batch size
    parser.add_argument('--batch_size', default=32)

    # Ancestors args
    parser.add_argument('--no_ancestors', type=bool, default=False)
    parser.add_argument('--max_ancestors', type=int, default=3)
    parser.add_argument('--used_ancestors', type=int, default=3)
    parser.add_argument('--direction_end', type=bool, default=False)

    # Trainer args
    parser.add_argument('--epochs', type=int, default=4)

    return parser


def print_config(c: Config):
    print_title('Running configuration')
    print(f'Batch size: {c.batch_size}')
    print(f'Num of epochs: {c.epochs}')
    print(f'Model: {c.base_model}')
    print(f'Training data: {c.train_file[:-5]}')
    print(f'Testing data: {c.test_file[:-5]}')
    if not c.use_ancestors:
        running_mode = 'No ancestors'
    else:
        running_mode = 'Use ancestors'
    print(f'Running mode: {running_mode}')

    if c.use_ancestors:
        direction = 'start' if c.ancestors_direction_start else 'end'
        print(f'Activation function: {str(c.activation_func)[:-2]}')
        print(f'Ancestors used: {c.used_ancestors} (padded to {c.max_ancestors})')
        print(f'Taking ancestors from the {direction} of the ancestors list')


def print_results(res: Dict, c: Config):
    # Print output
    print_title('Results')
    print(f'Accuracy: {res["eval_accuracy"]}')
    print(f'F1 score: {res["eval_f1"]}')
    print(f'Precision: {res["eval_precision"]}')
    print(f'Recall: {res["eval_recall"]}')

    # Save run data to json file
    saved_data = {
        'base_model': c.base_model,
        'activation_func': str(c.activation_func)[:-2],
        'train_file': c.train_file,
        'test_file': c.test_file,
        'batch_size': c.batch_size,
        'epochs': c.epochs,
        'accuracy': res['eval_accuracy'],
        'f1': res['eval_f1'],
        'precision': res['eval_precision'],
        'recall': res['eval_recall'],
    }

    ancestors_data = {
        'use_ancestors': c.use_ancestors
    }
    if c.use_ancestors:
        ancestors_data['max_ancestors'] = c.max_ancestors
        ancestors_data['used_ancestors'] = c.used_ancestors
        ancestors_data['direction_start'] = c.ancestors_direction_start

    saved_data['ancestors_data'] = ancestors_data

    # Create the results directory, and write the run data
    if not os.path.exists('./results'):
        os.mkdir('./results')

    file_name = datetime.datetime.now().strftime('./results/run_log_%Y_%m_%d_%H_%M_%s.json')
    with open(file_name, 'w') as f:
        json.dump(saved_data, f, indent=4)


def main(args: argparse.Namespace):
    # Set the config
    c = create_config(args)
    config.set_config(c)
    print_config(c)

    # Train the model
    print_title('Training')
    trained_model = trainer.train_model(config.get_config())

    # Print results
    result_metrics = trained_model.evaluate()
    print_results(result_metrics, c)


if __name__ == '__main__':
    # Get the args
    program_args = build_args_parser().parse_args()
    main(program_args)
