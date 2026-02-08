import streamlit as st
from predict import FakeNewsMeter

st.set_page_config(page_title="Fake News Meter", page_icon="🧪", layout="centered")

@st.cache_resource
def load_model():
    return FakeNewsMeter()

meter = load_model()

st.title("🧪 Fake News Meter")
st.write("Paste a claim/post text (and optional URL). Model will estimate if it's **likely true vs likely false**.")

text = st.text_area("Post / Claim text", height=180, placeholder="e.g., 'Breaking: Scientists confirm...'")
url = st.text_input("Optional URL (if the post includes one)", placeholder="https://example.com/article")

col1, col2 = st.columns(2)
with col1:
    run = st.button("Analyze", type="primary")
with col2:
    st.caption("Tip: shorter, single-claim text works best.")

if run:
    if not text.strip():
        st.error("Please enter some text.")
    else:
        label, conf, probs, signals = meter.predict(text=text, url=url)

        if label == "LIKELY_FALSE":
            st.error(f"Result: **{label}**  | confidence: **{conf:.2f}**")
        else:
            st.success(f"Result: **{label}**  | confidence: **{conf:.2f}**")

        st.subheader("Probabilities")
        st.write(probs)

        st.subheader("Signals (lightweight heuristics)")
        st.write(signals)

st.divider()
st.caption("Disclaimer: This is a probabilistic classifier, not a fact-checker. Use as a triage tool, not absolute truth.")
