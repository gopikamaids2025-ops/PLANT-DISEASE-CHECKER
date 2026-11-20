"""Streamlit app: upload a leaf photo and see the model's prediction.

Run (after training):
    streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st
from PIL import Image

from features import extract_features

st.set_page_config(page_title="Plant Disease Checker", page_icon="🌿")
st.title("🌿 Plant Disease Checker")
st.write("Upload a clear photo of a single leaf and click **Check leaf**.")


@st.cache_resource
def load_model():
    return joblib.load("models/plant_model.joblib")


try:
    saved = load_model()
except FileNotFoundError:
    st.error("No trained model found. Run `python train_model.py` first.")
    st.stop()

model = saved["model"]
class_names = saved["class_names"]

upload = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if upload is not None:
    image = Image.open(upload)
    st.image(image, caption="Your leaf", width=300)

    if st.button("Check leaf"):
        features = extract_features(image).reshape(1, -1)
        chances = model.predict_proba(features)[0]
        best = int(chances.argmax())

        st.success(f"Prediction: {class_names[best].replace('_', ' ')}")
        st.write(f"Confidence: {chances[best] * 100:.1f}%")

        if chances[best] < 0.60:
            st.warning("The model is not very sure. "
                       "Try a clearer, well-lit, close-up photo.")

        st.bar_chart(pd.DataFrame({"chance": chances}, index=class_names))

st.divider()
st.caption("Learning demo only. A prediction can be wrong, so it is not a "
           "substitute for advice from an agriculture expert.")
