import argparse

import torch.nn
from transformers import AutoTokenizer

import config
import trainer
from config import Config


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
        ancestors_direction_start=not args.direction_end,
        epochs=args.epochs
    )

    return c


def main():
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
    parser.add_argument('--direction_end', type=bool, default=False)

    # Trainer args
    parser.add_argument('--epochs', type=int, default=4)

    # Get the args
    args = parser.parse_args()

    # Set the config
    config.set_config(create_config(args))

    # Train the model
    trained_model = trainer.train_model(config.get_config())

    # Print results
    # TODO


if __name__ == '__main__':
    main()
