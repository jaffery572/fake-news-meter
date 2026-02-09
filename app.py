import streamlit as st
from predict import FakeNewsMeter, EvidenceChecker

st.set_page_config(page_title="Fake News Meter", page_icon="🧪", layout="centered")

st.title("🧪 Fake News Meter")
st.write("Fast mode gives a quick signal. Evidence mode searches the web and shows **7 evidence items**.")

mode = st.radio("Mode", ["Fast (Classifier)", "Real (Evidence-based)"], index=1)

text = st.text_area("Post / Claim text", height=180, placeholder="Paste one clear claim/headline…")
url = st.text_input("Optional URL", placeholder="https://example.com/article")

col1, col2 = st.columns(2)
with col1:
    run = st.button("Analyze", type="primary")
with col2:
    st.caption("Tip: One claim per run = best results.")

@st.cache_resource
def load_fast_model():
    return FakeNewsMeter()

@st.cache_resource
def load_checker():
    return EvidenceChecker()

def badge(tag: str):
    if tag == "REFUTES":
        return "❌ REFUTES"
    if tag == "SUPPORTS":
        return "✅ SUPPORTS"
    return "➖ NEUTRAL"

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

        st.subheader("Signals")
        st.write(signals)

        st.caption("⚠️ Fast mode is NOT guaranteed. Use Evidence mode for real sources.")

    else:
        checker = load_checker()
        verdict = checker.check(claim=text, url=url)

        st.subheader("Verdict")
        v = verdict["verdict"]
        conf = verdict["confidence"]

        if v == "FALSE":
            st.error(f"**FALSE** | confidence: **{conf:.2f}**")
        elif v == "TRUE":
            st.success(f"**TRUE** | confidence: **{conf:.2f}**")
        else:
            st.warning(f"**UNKNOWN** | confidence: **{conf:.2f}**")

        st.subheader("Why")
        st.write(verdict["summary"])

        st.subheader("Evidence (Top 7)")
        for i, item in enumerate(verdict["sources"], start=1):
            with st.expander(f"{i}. {badge(item['tag'])} — {item['title']}", expanded=(i <= 2)):
                st.markdown(item["url"])
                st.write(item["snippet"])

st.divider()
st.caption("Evidence mode shows sources. Still: verify critical decisions manually.")
