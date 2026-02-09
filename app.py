import streamlit as st
from predict import FakeNewsMeter, EvidenceChecker

st.set_page_config(page_title="Fake News Meter", page_icon="🧪", layout="centered")

st.title("🧪 Fake News Meter")
st.write("Choose mode: **Fast classifier** (quick signal) or **Evidence check** (real, source-based).")

mode = st.radio(
    "Mode",
    ["Fast (Classifier)", "Real (Evidence-based)"],
    index=1
)

text = st.text_area("Post / Claim text", height=180, placeholder="e.g., 'Breaking: Scientists confirm...'")
url = st.text_input("Optional URL (if the post includes one)", placeholder="https://example.com/article")

col1, col2 = st.columns(2)
with col1:
    run = st.button("Analyze", type="primary")
with col2:
    st.caption("Tip: one clear claim works best.")

@st.cache_resource
def load_fast_model():
    return FakeNewsMeter()

@st.cache_resource
def load_checker():
    return EvidenceChecker()

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

        st.caption("⚠️ This mode is a classifier (not guaranteed fact-check).")

    else:
        checker = load_checker()
        verdict = checker.check(claim=text, url=url)

        st.subheader("Verdict")
        v = verdict["verdict"]
        conf = verdict["confidence"]

        if v == "REFUTED":
            st.error(f"**{v}** (confidence: {conf:.2f})")
        elif v == "SUPPORTED":
            st.success(f"**{v}** (confidence: {conf:.2f})")
        else:
            st.warning(f"**{v}** (confidence: {conf:.2f})")

        st.subheader("Why (summary)")
        st.write(verdict["summary"])

        st.subheader("Evidence (sources)")
        for item in verdict["sources"]:
            st.markdown(f"- {item['title']} — {item['url']}")
            st.caption(item["snippet"])

st.divider()
st.caption("Evidence mode uses sources; still not 100% guaranteed. Always verify critical info manually.")
