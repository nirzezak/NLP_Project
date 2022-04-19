from typing import Dict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
from transformers import AutoModel, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments

import preprocessors
from config import Config
from model import SARCBert, SARCBertDataCollator


def compute_metrics(eval_pred) -> Dict:
    """
    Calculate accuracy, F1, precision, and recall scores for the given prediction.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_model(c: Config) -> Trainer:
    data_files = {
        'train': c.train_file,
        'test': c.test_file,
    }

    dataset = load_dataset('json', data_files=data_files)

    # Choose correct preprocess function
    if not c.use_ancestors:
        preprocess_function = preprocessors.preprocess_no_ancestors
    elif c.used_ancestors == 0:
        preprocess_function = preprocessors.preprocess_empty_ancestors
    elif c.ancestors_direction_start:
        preprocess_function = preprocessors.preprocess_ancestors_from_start
    else:
        preprocess_function = preprocessors.preprocess_ancestors_from_end

    # Tokenize the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True, batch_size=c.batch_size, load_from_cache_file=False)

    # Create our model
    pretrained_model = AutoModel.from_pretrained(c.base_model)
    model = SARCBert(pretrained_model, c)
    data_collator = SARCBertDataCollator(tokenizer=c.tokenizer)

    # Train the model
    training_args = TrainingArguments(
        output_dir='./huggingface_results',
        learning_rate=3e-5,
        per_device_train_batch_size=c.batch_size,
        per_device_eval_batch_size=c.batch_size,
        num_train_epochs=c.epochs,
        logging_dir='./huggingface_logs',
        evaluation_strategy='steps',
        weight_decay=1e-5,
        eval_steps=200,
        save_strategy='steps',
        save_steps=200,
        seed=100,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=c.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    return trainer
