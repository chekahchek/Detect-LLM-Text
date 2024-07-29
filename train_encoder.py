from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_domain', type=str, help='Dataset to train on. Use a comma separator to train on multiple dataset')
    parser.add_argument('--test_domain', type=str,  help='Dataset to validate and test on. Use a comma separator to train on multiple dataset')
    parser.add_argument('--model', type=str, help='Model to use')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum token length before truncation')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    args = parser.parse_args()
    train_domain = args.train_domain.split(',')
    test_domain = args.test_domain.split(',')

    ## Load dataset
    raw_dataset = load_dataset('chekahchek/ai6127-ghostbuster')

    # Preprocessing
    raw_dataset = raw_dataset.map(lambda x: {'labels' : 1 if x['labels'] == 'gpt' else 0})
    raw_dataset = raw_dataset.shuffle(seed=42)
    train_dataset = raw_dataset.filter(lambda x: x['domains'] in train_domain)
    test_dataset = raw_dataset.filter(lambda x: x['domains'] in test_domain)

    ## Tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    
    def tokenization(examples):
        return tokenizer(examples['texts'], max_length=args.max_len, truncation=True) 
    
    tokenized_train_dataset = train_dataset.map(tokenization, batched=True)
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['texts', 'domains'])
    tokenized_test_dataset = test_dataset.map(tokenization, batched=True)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(['texts', 'domains'])
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ## Model
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)

    ## Metrics
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, predictions)
        return {'f1' : f1}
    
    ## Training 
    training_args = TrainingArguments(
        output_dir = f"{args.model.replace('/', '-')}-finetuned",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        report_to='none',
        )
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train_dataset["train"],
        eval_dataset=tokenized_test_dataset["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        )
    
    trainer.train()

    ## Inference
    predictions = trainer.predict(tokenized_test_dataset['test'])
    print(f"Test Prediction = {predictions.metrics}")
    