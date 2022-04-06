from typing import Dict

from config import get_config


def preprocess_no_ancestors(example: Dict) -> Dict:
    """
    Preprocess for when we don't use any ancestors.
    """
    tokenizer = get_config().tokenizer

    # Tokenize the comment
    tokenized_text = tokenizer(example['text'])

    return {
        'input_ids': tokenized_text['input_ids'],
    }


def preprocess_empty_ancestors(example: Dict) -> Dict:
    """
    Preprocess for when we treat all of the ancestors as empty comments.
    """
    tokenizer = get_config().tokenizer

    # Tokenize the comment
    tokenized_text = tokenizer(example['text'])

    # Tokenize an empty ancestor comment
    max_ancestors = get_config().max_ancestors
    tokenized_ancestors = {'input_ids': []}
    for _ in example['ancestors']:
        res = tokenizer([""] * max_ancestors)
        tokenized_ancestors['input_ids'].append(res['input_ids'])

    return {
        'input_ids': tokenized_text['input_ids'],
        'ancestor_input_ids': tokenized_ancestors['input_ids'],
    }


def preprocess_ancestors_from_start(example: Dict) -> Dict:
    """
    Preprocess for when we take `max_ancestors` ancestors from the start of the
    ancestors list.
    """
    tokenizer = get_config().tokenizer

    # Tokenize the comment
    tokenized_text = tokenizer(example['text'])

    # Tokenize an empty ancestor comment
    max_ancestors = get_config().max_ancestors
    used_ancestors = get_config().used_ancestors
    tokenized_ancestors = {'input_ids': []}
    for ancestor in example['ancestors']:
        taken_ancestors = ancestor[:used_ancestors]
        res = tokenizer(taken_ancestors + [""] * (max_ancestors - len(taken_ancestors)))
        tokenized_ancestors['input_ids'].append(res['input_ids'])

    return {
        'input_ids': tokenized_text['input_ids'],
        'ancestor_input_ids': tokenized_ancestors['input_ids'],
    }


def preprocess_ancestors_from_end(example: Dict) -> Dict:
    """
    Preprocess for when we take `max_ancestors` ancestors from the end of the
    ancestors list.
    """
    tokenizer = get_config().tokenizer

    # Tokenize the comment
    tokenized_text = tokenizer(example['text'])

    # Tokenize an empty ancestor comment
    max_ancestors = get_config().max_ancestors
    used_ancestors = get_config().used_ancestors
    tokenized_ancestors = {'input_ids': []}
    for ancestor in example['ancestors']:
        taken_ancestors = ancestor[-used_ancestors:]
        res = tokenizer(taken_ancestors + [""] * (max_ancestors - len(taken_ancestors)))
        tokenized_ancestors['input_ids'].append(res['input_ids'])

    return {
        'input_ids': tokenized_text['input_ids'],
        'ancestor_input_ids': tokenized_ancestors['input_ids'],
    }
