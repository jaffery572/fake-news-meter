import os
import re
import time
import hashlib
import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from duckduckgo_search import DDGS

from utils import clean_text, extract_url_features, clickbait_score

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/deberta-v3-large")

# Evidence settings
EVIDENCE_K = int(os.getenv("EVIDENCE_K", "7"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "20"))
STRICT_TRUSTED_ONLY = os.getenv("STRICT_TRUSTED_ONLY", "0") == "1"


# =========================
# 1) FAST CLASSIFIER
# =========================
class FakeNewsMeter:
    def __init__(self, model_dir: str = ARTIFACTS_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
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
    Evidence checker that:
    - searches the web (DuckDuckGo)
    - ranks sources (trusted domains boosted)
    - returns exactly top-K evidence with SUPPORT/REFUTE/NEUTRAL tags
    - produces verdict: TRUE / FALSE / UNKNOWN (no guessing)
    """

    def __init__(self, timeout: int = 12, retries: int = 2):
        self.timeout = timeout
        self.retries = retries
        self.headers = {"User-Agent": "Mozilla/5.0 (FakeNewsMeter/1.0)"}

        # Trusted domains get higher score (not automatically "true", just higher priority)
        self.trusted_domains = [
            "bbc.com", "bbc.co.uk",
            "reuters.com",
            "apnews.com",
            "who.int", "cdc.gov",
            ".gov",
            "wikipedia.org",
            "politifact.com", "snopes.com", "factcheck.org", "afp.com",
        ]

        # Keyword signals (simple + deploy-safe)
        self.refute_words = ["false", "fake", "hoax", "misleading", "debunk", "not true", "rumor", "rumour"]
        self.support_words = ["confirmed", "official", "according to", "statement", "report", "announced", "evidence"]

        # tiny in-memory cache (per process)
        self._cache = {}

    def _domain_score(self, url: str) -> float:
        u = (url or "").lower()
        score = 0.0
        for d in self.trusted_domains:
            if d.startswith(".") and d in u:
                score += 1.2
            elif d in u:
                score += 1.6
        return score

    def _allowed(self, url: str) -> bool:
        if not STRICT_TRUSTED_ONLY:
            return True
        u = (url or "").lower()
        return any(d in u for d in self.trusted_domains)

    def _dedupe(self, items):
        seen = set()
        out = []
        for it in items:
            u = (it.get("url") or "").strip()
            if not u:
                continue
            if u in seen:
                continue
            seen.add(u)
            out.append(it)
        return out

    def _search_web(self, query: str):
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=MAX_SEARCH_RESULTS):
                    url = r.get("href", "") or ""
                    if not self._allowed(url):
                        continue
                    results.append(
                        {
                            "title": (r.get("title", "") or "").strip(),
                            "url": url.strip(),
                            "snippet": (r.get("body", "") or "").strip(),
                        }
                    )
        except Exception:
            pass
        return results

    def _fetch_page_text(self, url: str) -> str:
        if not url or not url.startswith(("http://", "https://")):
            return ""

        # cache
        if url in self._cache:
            return self._cache[url]

        last_err = None
        for _ in range(self.retries + 1):
            try:
                r = requests.get(url, timeout=self.timeout, headers=self.headers)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                    tag.decompose()
                text = " ".join(soup.get_text(" ").split())
                text = text[:7000]  # keep it light
                self._cache[url] = text
                return text
            except Exception as e:
                last_err = e
                time.sleep(0.4)

        self._cache[url] = ""
        return ""

    def _keywords(self, claim: str):
        words = re.findall(r"[A-Za-z]{4,}", (claim or "").lower())
        # unique preserve order
        uniq = list(dict.fromkeys(words))
        return uniq[:14]

    def _tag_evidence(self, claim: str, text: str):
        t = (text or "").lower()

        # strong refute
        if any(w in t for w in self.refute_words):
            return "REFUTES"
        # strong support
        if any(w in t for w in self.support_words):
            return "SUPPORTS"

        # weak support if claim keywords match well
        keys = set(self._keywords(claim))
        if keys:
            hit = sum(1 for k in keys if k in t)
            if hit >= max(2, len(keys) // 3):
                return "SUPPORTS"

        return "NEUTRAL"

    def check(self, claim: str, url: str = ""):
        claim_clean = " ".join((claim or "").strip().split())
        if not claim_clean:
            return {
                "verdict": "UNKNOWN",
                "confidence": 0.0,
                "summary": "Empty claim.",
                "sources": [],
            }

        # search queries
        q1 = f"{claim_clean} fact check"
        q2 = f"{claim_clean} Reuters OR BBC OR AP"
        results = self._search_web(q1) + self._search_web(q2)
        results = self._dedupe(results)

        # score and sort (trusted first + snippet length)
        for r in results:
            u = r.get("url", "")
            r["_score"] = self._domain_score(u) + (0.2 if len(r.get("snippet", "")) > 120 else 0.0)

        results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

        # Build evidence list (top K), fetch page text if snippet too short
        evidence = []
        for r in results:
            if len(evidence) >= EVIDENCE_K:
                break

            title = r.get("title", "") or "(no title)"
            link = r.get("url", "")
            snippet = (r.get("snippet", "") or "").strip()

            # enhance snippet from page if too short
            if len(snippet) < 80:
                page_text = self._fetch_page_text(link)
                if page_text:
                    snippet = page_text[:420]

            tag = self._tag_evidence(claim_clean, snippet)

            evidence.append(
                {"title": title, "url": link, "snippet": snippet, "tag": tag}
            )

        supports = sum(1 for e in evidence if e["tag"] == "SUPPORTS")
        refutes = sum(1 for e in evidence if e["tag"] == "REFUTES")

        # Decision policy: no guessing
        # TRUE if >=2 supports and 0 refutes
        # FALSE if >=2 refutes and refutes > supports
        # else UNKNOWN
        if refutes >= 2 and refutes > supports:
            verdict = "FALSE"
            confidence = min(0.95, 0.60 + 0.10 * (refutes - supports))
            summary = "Multiple sources contain debunk/refute signals. Treat as FALSE unless strong counter-evidence appears."
        elif supports >= 2 and refutes == 0:
            verdict = "TRUE"
            confidence = min(0.95, 0.60 + 0.08 * supports)
            summary = "Multiple sources support/confirm signals and no refute signals found. Treat as TRUE."
        else:
            verdict = "UNKNOWN"
            confidence = 0.50
            summary = "Not enough clean evidence (or mixed signals). No guessing — needs human verification."

        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "summary": summary,
            "sources": evidence,
        }
