import json
import torch
from datasets import load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset

class BC5Dataset(Dataset):
    """
    Reads formatted BC5‑CDR JSON:
      {
        "text": "...",
        "chemical": "...",
        "disease": "...",
        "label": 0/1,
        "chem_mesh": "...",   # optional, ignored
        "dis_mesh": "..."     # optional, ignored
      }
    Prepends entities to the input sequence:
      [CLS] chemical [SEP] disease [SEP] text [SEP]
    """
    def __init__(self, path, tokenizer, max_length=512):
        with open(path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        chem    = ex["chemical"]
        dis     = ex["disease"]
        text    = ex["text"]
        label   = ex["label"]

        # Concatenate entities and text to improve model's awareness
        # Format: "[CHEM] Naloxone [DIS] hypertensive [SEP] Naloxone reverses..."
        prompt = f"[CHEM] {chem} [DIS] {dis}"

        encoding = self.tokenizer(
            prompt,
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        item = {
            "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(label,                       dtype=torch.long),
        }
        return item

def compute_metrics(p):
    from sklearn.metrics import precision_recall_fscore_support
    preds  = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "eval_precision": precision,
        "eval_recall":    recall,
        "eval_f1":        f1,
    }

def main():
    model_name = "model/dmis-lab/biobert-v1.1"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Paths to the dataset files
    train_path = "process_data/train.json"
    dev_path   = "process_data/dev.json"
    test_path  = "process_data/test.json"

    train_ds = BC5Dataset(train_path, tokenizer)
    dev_ds   = BC5Dataset(dev_path,   tokenizer)
    test_ds  = BC5Dataset(test_path,  tokenizer)

    # Use DataCollatorWithPadding to dynamically pad inputs to the maximum length in batch
    data_collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir="./result/baseline",
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save model checkpoint at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        fp16=True,  # Use mixed precision
        logging_dir="./result/baseline/log",
        logging_steps=150,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 1) Train and validate
    trainer.train()
    print("Dev set results:", trainer.evaluate(eval_dataset=dev_ds))

    # 2) Evaluate on test set
    print("Test set results:", trainer.evaluate(eval_dataset=test_ds))

if __name__ == "__main__":
    main()
