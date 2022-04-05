from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer

# Get the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# This will pad the data
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Our model
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


def preprocess_function(example):
    return tokenizer(example['text'], truncation=True)


def main():
    # Load the DB
    imdb = load_dataset('imdb')

    # Tokenize
    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb['train'],
        eval_dataset=tokenized_imdb['test'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()


if __name__ == '__main__':
    main()
