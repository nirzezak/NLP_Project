from datasets import load_dataset
import random
import pandas as pd
from output_utils import print_title


def show_first_elements(dataset, num_elements=10):
    picks = random.sample(range(len(dataset)), num_elements)
    df = pd.DataFrame(dataset[picks])
    print(df)


def main():
    dataset = load_dataset('imdb')

    print_title('Examples')
    show_first_elements(dataset['train'])

    print_title('Full Example')
    example_idx = random.randint(0, len(dataset['train']) - 1)
    example = dataset['train'][example_idx]
    print(f'Text: {example["text"]}')
    print(f'Label: {example["label"]}')


if __name__ == '__main__':
    main()
