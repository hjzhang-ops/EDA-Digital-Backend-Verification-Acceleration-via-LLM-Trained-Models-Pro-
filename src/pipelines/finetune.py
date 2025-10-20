
import argparse, json, random, os, yaml
from datasets import load_dataset, ClassLabel, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def build_dataset(train_file, val_file, label_path, model_name):
    labels = json.load(open(label_path))["classes"]
    label2id = {l:i for i,l in enumerate(labels)}
    tokenize = AutoTokenizer.from_pretrained(model_name)

    def _load(path):
        return load_dataset('json', data_files=path, split='train')

    def tok(batch):
        enc = tokenize(batch['text'], truncation=True, padding='max_length', max_length=512)
        enc['labels'] = [label2id[x] for x in batch['label']]
        return enc

    ds_train = _load(train_file).map(tok, batched=True)
    ds_val = _load(val_file).map(tok, batched=True)
    return DatasetDict({'train': ds_train, 'validation': ds_val}), labels

def compute_metrics(eval_pred, labels):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg['model_name']
    ds, labels = build_dataset(cfg['data']['train_file'], cfg['data']['val_file'], cfg['data']['label_map'], model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))

    training_args = TrainingArguments(
        output_dir=cfg['output']['dir'],
        num_train_epochs=cfg['train']['epochs'],
        per_device_train_batch_size=cfg['train']['batch_size'],
        per_device_eval_batch_size=cfg['train']['batch_size'],
        learning_rate=float(cfg['train']['lr']),
        weight_decay=float(cfg['train']['weight_decay']),
        warmup_steps=cfg['train']['warmup_steps'],
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        compute_metrics=lambda p: compute_metrics(p, labels)
    )

    trainer.train()
    trainer.save_model(cfg['output']['dir'])
    print("Model saved to", cfg['output']['dir'])

if __name__ == '__main__':
    main()
