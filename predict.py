import os
import re
import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from duckduckgo_search import DDGS

from utils import clean_text, extract_url_features, clickbait_score

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/deberta-v3-large")


# =========================
# 1) FAST CLASSIFIER (yours)
# =========================
class FakeNewsMeter:
    def __init__(self, model_dir: str = ARTIFACTS_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # tokenizer from artifacts
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        # base model from HF
        base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

        # attach LoRA adapter
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


# ==========================================
# 2) EVIDENCE CHECKER (REAL / SOURCE-BASED)
# ==========================================
class EvidenceChecker:
    """
    Evidence-based checker:
    - If URL is provided: fetch page text (best-effort)
    - Search web snippets for claim + "fact check"
    - Return SUPPORTED / REFUTED / NOT_ENOUGH_EVIDENCE
    """

    def __init__(self, max_sources: int = 6, timeout: int = 12):
        self.max_sources = max_sources
        self.timeout = timeout
        self.headers = {"User-Agent": "Mozilla/5.0 (FakeNewsMeter/1.0)"}

    def _fetch_article_text(self, url: str) -> str:
        if not url or not url.startswith(("http://", "https://")):
            return ""
        try:
            r = requests.get(url, timeout=self.timeout, headers=self.headers)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                tag.decompose()

            text = " ".join(soup.get_text(" ").split())
            return text[:9000]  # limit for Streamlit stability
        except Exception:
            return ""

    def _search_web(self, query: str):
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_sources):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        }
                    )
        except Exception:
            pass
        return results

    def _keywords(self, claim: str):
        words = re.findall(r"[A-Za-z]{4,}", claim.lower())
        # keep only first ~14 keywords (enough)
        return list(dict.fromkeys(words))[:14]

    def check(self, claim: str, url: str = ""):
        claim_clean = " ".join(claim.strip().split())
        if not claim_clean:
            return {
                "verdict": "NOT_ENOUGH_EVIDENCE",
                "confidence": 0.0,
                "summary": "Empty claim.",
                "sources": [],
            }

        # 1) fetch article text (if URL)
        article_text = self._fetch_article_text(url)
        keys = self._keywords(claim_clean)

        # 2) web search
        query = f"{claim_clean} fact check"
        sources = self._search_web(query)

        # 3) scoring (simple but practical)
        support_score = 0.0
        refute_score = 0.0

        # Support signal: many keywords appear in the provided URL text
        if article_text and keys:
            at = article_text.lower()
            hit = sum(1 for k in keys if k in at)
            support_score += min(1.0, hit / max(3, len(keys) // 2))

        # Refute/support signals from snippets
        for s in sources:
            snip = (s.get("snippet") or "").lower()

            # strong refute words
            if any(x in snip for x in ["false", "fake", "hoax", "misleading", "debunk", "not true"]):
                refute_score += 0.35

            # mild support words
            if any(x in snip for x in ["confirmed", "official", "according to", "statement", "report"]):
                support_score += 0.20

        # 4) decide verdict
        margin = support_score - refute_score

        if margin >= 0.45:
            verdict = "SUPPORTED"
            confidence = min(0.95, 0.55 + margin)
            summary = "Evidence snippets + URL text show signals that support this claim."
        elif margin <= -0.45:
            verdict = "REFUTED"
            confidence = min(0.95, 0.55 + (-margin))
            summary = "Evidence snippets include 'false/hoax/misleading' signals against this claim."
        else:
            verdict = "NOT_ENOUGH_EVIDENCE"
            confidence = 0.50
            summary = "Could not reliably confirm or refute. Needs manual verification."

        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "summary": summary,
            "sources": sources,
        }
