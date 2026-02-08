import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from utils import clean_text

BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/deberta-v3-large")
OUT_DIR = os.getenv("OUT_DIR", "artifacts")

# Map LIAR labels into binary:
# LIAR labels: [pants-fire, false, barely-true, half-true, mostly-true, true]
FALSE_SET = {0, 1}              # pants-fire, false
TRUE_SET  = {4, 5}              # mostly-true, true
# middle labels (2,3) can be treated as "uncertain"; we keep them as 0/1 via thresholding
# below we map: 0,1,2 -> 0  and 3,4,5 -> 1 (you can adjust)
def to_binary_label(liar_label: int) -> int:
    return 1 if liar_label >= 3 else 0

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

@dataclass
class TrainConfig:
    max_length: int = 256
    lr: float = 2e-5
    epochs: int = 2
    batch: int = 8
    grad_accum: int = 2
    warmup_ratio: float = 0.06

def main():
    cfg = TrainConfig()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading dataset: liar")
    ds = load_dataset("liar")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    def build_text(example: Dict[str, Any]) -> Dict[str, Any]:
        # LIAR fields include: statement, subject, speaker, state, party, context, etc.
        statement = clean_text(example.get("statement", ""))
        context = clean_text(example.get("context", ""))
        speaker = clean_text(example.get("speaker", ""))

        # We fuse fields to give model richer signal
        text = f"Statement: {statement} | Speaker: {speaker} | Context: {context}"
        label = to_binary_label(int(example["label"]))
        return {"text": text, "labels": label}

    ds = ds.map(build_text, remove_columns=ds["train"].column_names)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    ds = ds.map(tok, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Loading base model")
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    # LoRA config: good accuracy boost vs compute cost
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["query_proj", "value_proj"]  # DeBERTa-v3 uses these module names
    )

    # Some HF versions name modules slightly differently; fallback if needed:
    # If you get "target modules not found", set target_modules=["q_proj","v_proj"] or remove target_modules.
    model = get_peft_model(model, lora)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch,
        per_device_eval_batch_size=cfg.batch,
        gradient_accumulation_steps=cfg.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("Evaluating on test set")
    test_metrics = trainer.evaluate(ds["test"])
    print(test_metrics)

    print("Saving model + tokenizer")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    # Save metrics
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")

    print("Done. Artifacts saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
