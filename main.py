import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_MODEL = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = load_metric('accuracy')


def preprocess_function(example):
    # return tokenizer(example['text1'], example['text2'], truncation=True, padding=True)
    ancestors = ['. '.join(ancestor) for ancestor in example['ancestors']]
    return tokenizer(example['text'], ancestors, truncation='only_second', padding=True)


def compute_metrics(eval_pred):
    # predictions, labels = eval_pred
    # predictions = np.argmax(predictions, axis=1)
    #
    # return metric.compute(predictions=predictions, references=labels)
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


def main():
    # Load the DB
    should_mini = True
    prefix = 'mini-' if should_mini else ''
    data_files = {
        'train': prefix + r'train-pol-balanced.json',
        'test': prefix + r'test-pol-balanced.json',
    }
    dataset = load_dataset('json', data_files=data_files)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    main()
