import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from utils import clean_text, extract_url_features, clickbait_score

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/deberta-v3-large")

class FakeNewsMeter:
    def __init__(self, model_dir: str = ARTIFACTS_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # tokenizer from artifacts (saved)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        # load base model from HF
        base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

        # attach LoRA adapter (from artifacts)
        self.model = PeftModel.from_pretrained(base, model_dir)

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text: str, url: str = ""):
        text = clean_text(text)
        feats = extract_url_features(url)
        cb = clickbait_score(text)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)

        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        true_p = float(probs[1])
        false_p = float(probs[0])

        label = "LIKELY_TRUE" if true_p >= 0.5 else "LIKELY_FALSE"
        confidence = max(true_p, false_p)

        signals = {
            "base_model": BASE_MODEL,
            "clickbait_score": cb,
            "url_has_https": feats["is_https"],
            "url_tld_suspicious": feats["tld_suspicious"],
            "url_has_ip": feats["has_ip"],
            "url_num_dots": feats["num_dots"],
        }
        return label, confidence, {"true_prob": true_p, "false_prob": false_p}, signals
