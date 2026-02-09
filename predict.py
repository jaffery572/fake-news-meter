import os
import re
import time
import hashlib
from urllib.parse import urlparse

import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from duckduckgo_search import DDGS

from utils import clean_text, extract_url_features, clickbait_score

try:
    import trafilatura
except Exception:
    trafilatura = None


ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/deberta-v3-large")

# Evidence mode tuning
EVIDENCE_K = int(os.getenv("EVIDENCE_K", "7"))                    # show top K evidence cards
SEARCH_RESULTS = int(os.getenv("SEARCH_RESULTS", "120"))          # how many DDG results to collect
FETCH_TOP = int(os.getenv("FETCH_TOP", "16"))                     # how many pages to actually fetch (avoid timeout)
STRICT_TRUSTED_ONLY = os.getenv("STRICT_TRUSTED_ONLY", "0") == "1"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))
REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "2"))

# lightweight similarity threshold (0-1), higher = stricter matching
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.28"))


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
# 2) EVIDENCE CHECKER (INTERNATIONAL + LOCAL)
# ==========================================
class EvidenceChecker:
    """
    - Collects 100+ search results (snippets)
    - Splits sources into INTERNATIONAL vs LOCAL
    - Clusters by similarity (do both sides talk about same thing?)
    - Fetches only top N pages to avoid timeouts
    - Verdict is strict: TRUE / FALSE / UNKNOWN
    """

    def __init__(self):
        self.timeout = REQUEST_TIMEOUT
        self.retries = REQUEST_RETRIES
        self.headers = {"User-Agent": "Mozilla/5.0 (FakeNewsMeter/3.0)"}
        self._page_cache = {}
        self._query_cache = {}

        # INTERNATIONAL sources (examples)
        self.international_domains = [
            "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
            "aljazeera.com", "theguardian.com", "nytimes.com",
            "washingtonpost.com", "cnn.com", "dw.com", "france24.com",
            "who.int", "cdc.gov", ".gov", "wikipedia.org",
            "politifact.com", "snopes.com", "factcheck.org", "afp.com",
        ]

        # LOCAL / REGIONAL sources (Pakistan + nearby; you can extend)
        self.local_domains = [
            "dawn.com", "geo.tv", "thenews.com.pk", "arynews.tv",
            "tribune.com.pk", "samaa.tv", "dunya.com.pk", "bolnews.com",
            "92newshd.tv", "radio.gov.pk", "nation.com.pk",
            "pakobserver.net", "pid.gov.pk",
        ]

        # refute/support words (snippet-level)
        self.refute_words = [
            "false", "fake", "hoax", "misleading", "debunk", "not true",
            "rumor", "rumour", "fabricated", "baseless", "no evidence"
        ]
        self.support_words = [
            "confirmed", "official", "according to", "statement", "report",
            "announced", "verified", "police said", "authorities said"
        ]

    # -----------------------
    # Utilities
    # -----------------------
    def _domain(self, url: str) -> str:
        try:
            return (urlparse(url).netloc or "").lower()
        except Exception:
            return ""

    def _is_international(self, url: str) -> bool:
        u = (url or "").lower()
        return any(d in u for d in self.international_domains)

    def _is_local(self, url: str) -> bool:
        u = (url or "").lower()
        return any(d in u for d in self.local_domains)

    def _allowed(self, url: str) -> bool:
        if not STRICT_TRUSTED_ONLY:
            return True
        # strict: allow intl + local lists only
        return self._is_international(url) or self._is_local(url)

    def _dedupe(self, items):
        seen = set()
        out = []
        for it in items:
            u = (it.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(it)
        return out

    def _keywords(self, text: str):
        words = re.findall(r"[A-Za-z]{4,}", (text or "").lower())
        uniq = list(dict.fromkeys(words))
        return uniq[:16]

    def _normalize(self, s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _jaccard_sim(self, a: str, b: str) -> float:
        A = set(self._normalize(a).split())
        B = set(self._normalize(b).split())
        if not A or not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))

    def _tag_evidence(self, claim: str, blob: str) -> str:
        t = (blob or "").lower()
        if any(w in t for w in self.refute_words):
            return "REFUTES"
        if any(w in t for w in self.support_words):
            return "SUPPORTS"

        keys = set(self._keywords(claim))
        if keys:
            hit = sum(1 for k in keys if k in t)
            if hit >= max(2, len(keys) // 3):
                return "SUPPORTS"
        return "NEUTRAL"

    # -----------------------
    # Web search
    # -----------------------
    def _search_web(self, query: str):
        q = (query or "").strip()
        if not q:
            return []
        q_key = hashlib.sha256(q.encode("utf-8")).hexdigest()
        if q_key in self._query_cache:
            return self._query_cache[q_key]

        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(q, max_results=SEARCH_RESULTS):
                    url = (r.get("href") or "").strip()
                    if not url:
                        continue
                    if not self._allowed(url):
                        continue
                    results.append({
                        "title": (r.get("title") or "").strip(),
                        "url": url,
                        "snippet": (r.get("body") or "").strip(),
                        "domain": self._domain(url),
                    })
        except Exception:
            pass

        results = self._dedupe(results)
        self._query_cache[q_key] = results
        return results

    # -----------------------
    # Fetch page text (limited)
    # -----------------------
    def _fetch_page_text(self, url: str) -> str:
        if not url or not url.startswith(("http://", "https://")):
            return ""
        if url in self._page_cache:
            return self._page_cache[url]

        last_err = None
        for _ in range(self.retries + 1):
            try:
                r = requests.get(url, timeout=self.timeout, headers=self.headers)
                r.raise_for_status()
                html = r.text

                if trafilatura is not None:
                    extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
                    if extracted and len(extracted.strip()) > 200:
                        text = " ".join(extracted.split())[:6500]
                        self._page_cache[url] = text
                        return text

                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                    tag.decompose()
                text = " ".join(soup.get_text(" ").split())[:6500]
                self._page_cache[url] = text
                return text

            except Exception as e:
                last_err = e
                time.sleep(0.35)

        self._page_cache[url] = ""
        return ""

    # -----------------------
    # Main check
    # -----------------------
    def check(self, claim: str, url: str = ""):
        claim_clean = " ".join((claim or "").strip().split())
        if not claim_clean:
            return {"verdict": "UNKNOWN", "confidence": 0.0, "summary": "Empty claim.", "sources": [], "compare": {}}

        # If claim is short headline fragment, expand query
        base_q = claim_clean
        if len(base_q.split()) <= 9:
            base_q = f"{base_q} what happened where when"

        q_exact = f"\"{claim_clean}\""
        q_fact = f"{base_q} fact check"
        q_news = f"{base_q} news report"
        q_intl = f"{base_q} Reuters OR BBC OR AP OR AlJazeera"
        q_local = f"{base_q} Dawn OR Geo OR ARY OR Tribune OR Samaa"

        # Search 100+ results (snippets)
        results = []
        for q in [q_exact, q_news, q_fact, q_intl, q_local]:
            results.extend(self._search_web(q))

        results = self._dedupe(results)

        # Bucket into intl/local/other
        intl = [r for r in results if self._is_international(r["url"])]
        local = [r for r in results if self._is_local(r["url"])]
        other = [r for r in results if (r not in intl and r not in local)]

        # Similarity linking: do local and intl talk about SAME event?
        # We'll compute best match pairs based on title+snippet similarity.
        pairs = []
        for li in local[:60]:
            best = None
            best_s = 0.0
            ltxt = (li.get("title", "") + " " + li.get("snippet", "")).strip()
            for ii in intl[:60]:
                itxt = (ii.get("title", "") + " " + ii.get("snippet", "")).strip()
                s = self._jaccard_sim(ltxt, itxt)
                if s > best_s:
                    best_s = s
                    best = ii
            if best and best_s >= SIM_THRESHOLD:
                pairs.append({
                    "local_title": li.get("title", ""),
                    "local_url": li.get("url", ""),
                    "intl_title": best.get("title", ""),
                    "intl_url": best.get("url", ""),
                    "similarity": round(best_s, 3),
                })

        agreement_score = 0.0
        if intl and local:
            # proportion of local entries that have an intl match
            agreement_score = min(1.0, len(pairs) / max(1, min(len(local), 30)))

        # Now pick evidence candidates: prefer intl+local, then other
        candidates = (intl + local + other)
        candidates = candidates[:max(SEARCH_RESULTS, 120)]

        # Fetch only top N pages for stronger snippets (avoid timeout)
        evidence = []
        fetched = 0
        for r in candidates:
            if len(evidence) >= EVIDENCE_K:
                break

            title = r.get("title") or "(no title)"
            link = r.get("url") or ""
            snippet = (r.get("snippet") or "").strip()

            if len(snippet) < 90 and fetched < FETCH_TOP:
                page_text = self._fetch_page_text(link)
                if page_text:
                    snippet = page_text[:420]
                fetched += 1

            tag = self._tag_evidence(claim_clean, snippet)

            evidence.append({
                "title": title,
                "url": link,
                "snippet": snippet,
                "tag": tag,
                "bucket": "INTERNATIONAL" if self._is_international(link) else ("LOCAL" if self._is_local(link) else "OTHER"),
                "domain": self._domain(link),
            })

        supports = sum(1 for e in evidence if e["tag"] == "SUPPORTS")
        refutes = sum(1 for e in evidence if e["tag"] == "REFUTES")

        # Strict decision policy (no guessing):
        # TRUE  => >=2 supports and 0 refutes AND agreement_score >= 0.15
        # FALSE => >=2 refutes and refutes > supports
        # else  => UNKNOWN
        if refutes >= 2 and refutes > supports:
            verdict = "FALSE"
            confidence = min(0.95, 0.60 + 0.10 * (refutes - supports))
            summary = "Multiple sources contain refute/debunk signals. Marked as FALSE."
        elif supports >= 2 and refutes == 0 and agreement_score >= 0.15:
            verdict = "TRUE"
            confidence = min(0.95, 0.60 + 0.08 * supports + 0.15 * agreement_score)
            summary = "Multiple sources support signals and cross-source agreement exists. Marked as TRUE."
        else:
            verdict = "UNKNOWN"
            confidence = 0.50
            if not evidence:
                summary = "No reliable sources found. Add URL or make claim more specific."
            else:
                summary = "Evidence is mixed/weak or cross-source agreement is low. Marked UNKNOWN (no guessing)."

        compare = {
            "search_results_total": len(results),
            "international_count": len(intl),
            "local_count": len(local),
            "other_count": len(other),
            "matched_pairs": pairs[:10],  # show top 10 pairs
            "agreement_score": round(agreement_score, 3),
            "notes": f"Collected {len(results)} results; fetched {min(fetched, FETCH_TOP)} pages for stronger evidence.",
        }

        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "summary": summary,
            "sources": evidence,
            "compare": compare,
        }
