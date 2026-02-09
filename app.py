import streamlit as st
from predict import FakeNewsMeter, EvidenceChecker

st.set_page_config(page_title="Fake News Meter", page_icon="🧪", layout="centered")

st.title("🧪 Fake News Meter")
st.write("Choose mode: **Fast classifier** (quick signal) or **Evidence check** (real, source-based).")

mode = st.radio("Mode", ["Fast (Classifier)", "Real (Evidence-based)"], index=1)

text = st.text_area("Post / Claim text", height=180, placeholder="Write ONE clear claim. Example: 'Explosion happened at X place on Y date.'")
url = st.text_input("Optional URL (recommended)", placeholder="https://bbc.com/... or https://reuters.com/...")

col1, col2 = st.columns(2)
with col1:
    run = st.button("Analyze", type="primary")
with col2:
    st.caption("Tip: One clear claim + article URL = best results.")

@st.cache_resource
def load_fast_model():
    return FakeNewsMeter()

@st.cache_resource
def load_checker():
    return EvidenceChecker()

def badge(tag: str):
    if tag == "REFUTES":
        return "🟥 REFUTES"
    if tag == "SUPPORTS":
        return "🟩 SUPPORTS"
    return "🟨 NEUTRAL"

if run:
    if not text.strip():
        st.error("Please enter some text.")
        st.stop()

    if mode == "Fast (Classifier)":
        meter = load_fast_model()
        label, conf, probs, signals = meter.predict(text=text, url=url)

        if label == "LIKELY_FALSE":
            st.error(f"Result: **{label}** | confidence: **{conf:.2f}**")
        else:
            st.success(f"Result: **{label}** | confidence: **{conf:.2f}**")

        st.subheader("Probabilities")
        st.write(probs)

        st.subheader("Signals (lightweight heuristics)")
        st.write(signals)

        st.caption("⚠️ Fast mode is a classifier (not guaranteed fact-check).")

    else:
        checker = load_checker()
        verdict = checker.check(claim=text, url=url)

        st.subheader("Verdict")
        v = verdict["verdict"]
        conf = verdict["confidence"]

        if v == "FALSE":
            st.error(f"**{v}** (confidence: {conf:.2f})")
        elif v == "TRUE":
            st.success(f"**{v}** (confidence: {conf:.2f})")
        else:
            st.warning(f"**{v}** (confidence: {conf:.2f})")

        st.subheader("Why (summary)")
        st.write(verdict["summary"])

        # Comparison panel: international vs local
        comp = verdict.get("compare", {})
        st.subheader("International vs Local (comparison)")
        st.write({
            "total_results": comp.get("search_results_total"),
            "international_count": comp.get("international_count"),
            "local_count": comp.get("local_count"),
            "other_count": comp.get("other_count"),
            "agreement_score": comp.get("agreement_score"),
            "notes": comp.get("notes"),
        })

        pairs = comp.get("matched_pairs", [])
        if pairs:
            st.markdown("**Top matched pairs (local ↔ international)**")
            for p in pairs:
                st.markdown(f"- Similarity **{p['similarity']}**")
                st.markdown(f"  - LOCAL: {p['local_title']} — {p['local_url']}")
                st.markdown(f"  - INTL: {p['intl_title']} — {p['intl_url']}")

        st.subheader("Evidence (top sources)")
        for item in verdict["sources"]:
            st.markdown(f"**{badge(item['tag'])}**  |  **{item['bucket']}**  |  `{item['domain']}`")
            st.markdown(f"- {item['title']} — {item['url']}")
            st.caption(item["snippet"])

st.divider()
st.caption("Evidence mode checks many sources but cannot guarantee 100% truth. For critical decisions, verify manually.")
