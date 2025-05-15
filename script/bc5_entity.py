# baseline_bc5_entity.py

import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import precision_recall_fscore_support

class BC5Dataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        # Load preprocessed samples from JSON file
        with open(path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Return the total number of samples
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve a single example
        ex = self.samples[idx]
        chem = ex["chemical"]
        dis = ex["disease"]
        text = ex["text"]
        label = ex["label"]

        # 1) Locate entity positions in the text
        c_start = text.find(chem)
        c_end = c_start + len(chem) if c_start != -1 else 0
        d_start = text.find(dis)
        # If disease overlaps or occurs before chemical, search after chemical
        if d_start == -1 or (c_start <= d_start < c_end):
            d_start = text.find(dis, c_end)
        d_end = d_start + len(dis) if d_start != -1 else 0

        # 2) Insert entity markers, processing from end to start to avoid offset shifts
        marked = text
        spans = []
        if d_start != -1:
            spans.append((d_start, d_end, "[E2]", "[/E2]"))
        if c_start != -1:
            spans.append((c_start, c_end, "[E1]", "[/E1]"))
        for s, e, otag, ctag in sorted(spans, key=lambda x: x[0], reverse=True):
            marked = marked[:e] + ctag + marked[e:]
            marked = marked[:s] + otag + marked[s:]

        # 3) Tokenize and apply fixed padding
        encoding = self.tokenizer(
            marked,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        return {
            "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(label,                       dtype=torch.long),
        }


def compute_metrics(eval_pred):
    # Compute precision, recall, and F1 score for binary classification
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    preds = preds.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    # Configuration: adjust model path and data file paths as needed
    model_name = "model/dmis-lab/biobert-v1.1"
    train_path = "process_data/train.json"
    dev_path   = "process_data/dev.json"
    test_path  = "process_data/test.json"

    # 1. Initialize tokenizer and add special entity tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    })
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load model and resize token embeddings (disable mean resizing warning)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # 3. Prepare datasets and data collator
    train_ds = BC5Dataset(train_path, tokenizer)
    dev_ds   = BC5Dataset(dev_path,   tokenizer)
    test_ds  = BC5Dataset(test_path,  tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer, padding="max_length")

    # 4. Define training arguments (increase batch size as needed)
    args = TrainingArguments(
        output_dir="./result/entity",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1.6e-5,
        per_device_train_batch_size=8,   # Adjust based on GPU memory
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,   # Remove to increase effective batch size
        num_train_epochs=5,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./result/entity/log",
        logging_steps=150,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. Train and evaluate
    trainer.train()
    print("Dev set results:", trainer.evaluate(eval_dataset=dev_ds))
    print("Test set results:", trainer.evaluate(eval_dataset=test_ds))


if __name__ == "__main__":
    main()
