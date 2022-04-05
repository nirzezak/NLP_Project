from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

from model import SARCBert, SARCBertDataCollator

DEVICE = torch.device('cuda:0')

BASE_MODEL = 'distilbert-base-uncased'
BATCH_SIZE = 64
MAX_ANCESTORS = 3

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
data_collator = SARCBertDataCollator(tokenizer=tokenizer)


def preprocess_function(example):
    tokenized_text = tokenizer(example['text'])
    tokenized_ancestors = {'input_ids': []}
    for ancestor in example['ancestors']:
        # Make each sample have MAX_ANCESTORS, by padding with empty comments
        res = tokenizer(ancestor[:MAX_ANCESTORS] + [""] * (MAX_ANCESTORS - len(ancestor)))
        tokenized_ancestors['input_ids'].append(res['input_ids'])

    # tokenized_ancestors = tokenizer(example['ancestors'][0])

    return {
        'input_ids': tokenized_text['input_ids'],
        'ancestor_input_ids': tokenized_ancestors['input_ids'],
    }


def compute_metrics(eval_pred):
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
    data_files = {
        'train': r'train-pol-balanced.json',
        'test': r'test-pol-balanced.json',
    }
    dataset = load_dataset('json', data_files=data_files)
    tokenized_dataset = dataset.map(preprocess_function, batched=True, batch_size=BATCH_SIZE)

    bert_model = AutoModel.from_pretrained(BASE_MODEL)
    model = SARCBert(bert_model=bert_model)
    # model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        logging_dir='./logs'
        #dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    print("")


if __name__ == '__main__':
    main()
