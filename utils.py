import re
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {"xyz", "top", "icu", "click", "buzz", "info", "live"}
SUSPICIOUS_WORDS = {
    "shocking", "you won’t believe", "secret", "miracle", "cure", "exposed",
    "urgent", "breaking", "must see", "guaranteed", "they don’t want you to know"
}

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_url_features(url: str) -> dict:
    """
    Lightweight URL/domain heuristics (fast + helps a lot in practice).
    No external reputation API required.
    """
    if not url:
        return {
            "has_url": 0, "domain_len": 0, "path_len": 0, "num_dots": 0,
            "is_https": 0, "tld_suspicious": 0, "has_ip": 0
        }

    try:
        p = urlparse(url)
        domain = p.netloc.lower()
        path = (p.path or "").lower()
        tld = domain.split(".")[-1] if "." in domain else ""

        has_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain) else 0
        return {
            "has_url": 1,
            "domain_len": len(domain),
            "path_len": len(path),
            "num_dots": domain.count("."),
            "is_https": 1 if p.scheme == "https" else 0,
            "tld_suspicious": 1 if tld in SUSPICIOUS_TLDS else 0,
            "has_ip": has_ip,
        }
    except Exception:
        return {
            "has_url": 1, "domain_len": 0, "path_len": 0, "num_dots": 0,
            "is_https": 0, "tld_suspicious": 0, "has_ip": 0
        }

def clickbait_score(text: str) -> float:
    t = (text or "").lower()
    score = 0
    for w in SUSPICIOUS_WORDS:
        if w in t:
            score += 1
    # lots of CAPS and !!! are weak signals too
    caps = sum(1 for c in text if c.isupper())
    score += 1 if caps > 25 else 0
    score += 1 if "!!!" in text else 0
    return float(score)
