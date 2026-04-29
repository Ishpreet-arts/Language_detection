
import streamlit as st
import pickle
import numpy as np
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=" Language Detector",
    page_icon="🌍",
    layout="centered"
)

# ── Language Emoji Map ─────────────────────────────────────────────────────────
LANG_EMOJI = {
    "Arabic":     "🇸🇦",
    "Chinese":    "🇨🇳",
    "Dutch":      "🇳🇱",
    "English":    "🇬🇧",
    "Estonian":   "🇪🇪",
    "French":     "🇫🇷",
    "Hindi":      "🇮🇳",
    "Indonesian": "🇮🇩",
    "Japanese":   "🇯🇵",
    "Korean":     "🇰🇷",
    "Latin":      "🏛️",
    "Persian":    "🇮🇷",
    "Portugese":  "🇵🇹",
    "Pushto":     "🇦🇫",
    "Romanian":   "🇷🇴",
    "Russian":    "🇷🇺",
    "Spanish":    "🇪🇸",
    "Swedish":    "🇸🇪",
    "Tamil":      "🇮🇳",
    "Thai":       "🇹🇭",
    "Turkish":    "🇹🇷",
    "Urdu":       "🇵🇰",
}

# ── Load Model & Vectorizer ────────────────────────────────────────────────────
@st.cache_resource   # cache so it only loads once
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    vec_path   = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        st.error("❌ Model files not found! Please run `python train_model.py` first.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌍 Language Detector")
st.markdown(
    "Type or paste any text below and the model will predict which language it is written in. "
    "Supports **22 languages**!"
)
st.divider()

# ── Input Area ────────────────────────────────────────────────────────────────
user_input = st.text_area(
    label="✏️ Enter your text here:",
    placeholder="e.g. Bonjour, comment ça va ?",
    height=150
)

# ── Try Example Buttons ───────────────────────────────────────────────────────
st.markdown("**Try an example:**")
examples = {
    "🇬🇧 English":  "The quick brown fox jumps over the lazy dog.",
    "🇫🇷 French":   "Bonjour, comment ça va ? Je m'appelle Marie.",
    "🇮🇳 Hindi":    "नमस्ते, आप कैसे हैं? मेरा नाम राज है।",
    "🇯🇵 Japanese": "こんにちは、お元気ですか？私はジョンです。",
    "🇸🇦 Arabic":   "مرحبا كيف حالك؟ أنا بخير شكرا لك.",
}

cols = st.columns(len(examples))
for col, (label, text) in zip(cols, examples.items()):
    if col.button(label, use_container_width=True):
        user_input = text
        st.session_state["example_text"] = text

# Handle example click via session state
if "example_text" in st.session_state and not user_input:
    user_input = st.session_state["example_text"]

st.divider()

# ── Predict Button ─────────────────────────────────────────────────────────────
if st.button("🔍 Detect Language", type="primary", use_container_width=True):
    text_to_use = user_input.strip()

    if not text_to_use:
        st.warning("⚠️ Please enter some text first!")
    elif len(text_to_use) < 3:
        st.warning("⚠️ Please enter at least 3 characters for a reliable prediction.")
    else:
        # Transform text and predict
        transformed = vectorizer.transform([text_to_use])
        prediction  = model.predict(transformed)[0]
        proba       = model.predict_proba(transformed)[0]

        # Get top 3 predictions
        classes     = model.classes_
        top3_idx    = np.argsort(proba)[::-1][:3]
        top3        = [(classes[i], proba[i] * 100) for i in top3_idx]

        # Show result
        emoji = LANG_EMOJI.get(prediction, "🌐")
        confidence = top3[0][1]

        st.success(f"### {emoji} Detected Language: **{prediction}**")

        # Confidence bar
        conf_color = "green" if confidence >= 80 else "orange" if confidence >= 50 else "red"
        st.markdown(f"**Confidence:** `{confidence:.1f}%`")
        st.progress(int(confidence))

        # Top 3
        st.markdown("**Top 3 Predictions:**")
        for lang, prob in top3:
            em = LANG_EMOJI.get(lang, "🌐")
            bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
            st.markdown(f"{em} **{lang}** &nbsp;&nbsp; `{bar}` &nbsp; {prob:.1f}%")

# ── Sidebar Info ──────────────────────────────────────────────────────────────

