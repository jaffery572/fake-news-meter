import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import clean_text, extract_url_features, clickbait_score

# Order matters:
# 1) If repo has artifacts/ folder -> use it
# 2) else use env var MODEL_ID (Streamlit secrets)
# 3) else fallback to a public HF model (always available)
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "distilbert-base-uncased-finetuned-sst-2-english")
LOCAL_ARTIFACTS = "artifacts"

def pick_model_source():
    if os.path.isdir(LOCAL_ARTIFACTS) and os.path.exists(os.path.join(LOCAL_ARTIFACTS, "config.json")):
        return LOCAL_ARTIFACTS
    return DEFAULT_MODEL_ID

class FakeNewsMeter:
    def __init__(self):
        self.source = pick_model_source()
        self.tokenizer = AutoTokenizer.from_pretrained(self.source, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.source)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # Generic classifier support:
        # If model has 2 labels: pick higher prob
        pred_idx = int(probs.argmax())
        conf = float(probs[pred_idx])

        label = "LIKELY_FALSE" if pred_idx == 0 else "LIKELY_TRUE"

        signals = {
            "model_source": self.source,
            "clickbait_score": cb,
            "url_has_https": feats["is_https"],
            "url_tld_suspicious": feats["tld_suspicious"],
            "url_has_ip": feats["has_ip"],
            "url_num_dots": feats["num_dots"],
        }
        return label, conf, {"probs": probs.tolist()}, signals
