import os
from pathlib import Path
from pickle import load

import numpy as np
import streamlit as st
from PIL import Image
from keras.applications.xception import Xception
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def clean_caption(text):
    words = [w for w in text.split() if w not in {"start", "end"}]
    return " ".join(words).strip()


def extract_features(image: Image.Image, encoder):
    image = image.convert("RGB").resize((299, 299))
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 127.5
    arr = arr - 1.0
    return encoder.predict(arr, verbose=0)


def generate_caption(model, tokenizer, photo, max_len):
    in_text = "start"
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)
        pred = model.predict([photo, sequence], verbose=0)
        pred_id = int(np.argmax(pred))
        word = word_for_id(pred_id, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "end":
            break
    return clean_caption(in_text)


def find_latest_checkpoint(models_dir: Path):
    checkpoints = sorted(models_dir.glob("model_*.h5"))
    if not checkpoints:
        return None
    return checkpoints[-1]


@st.cache_resource
def load_artifacts(model_path: str, tokenizer_path: str, max_len_path: str):
    caption_model = load_model(model_path, compile=False)
    tokenizer = load(open(tokenizer_path, "rb"))
    with open(max_len_path, "r", encoding="utf-8") as f:
        max_len = int(f.read().strip())
    encoder = Xception(include_top=False, pooling="avg", weights=None)
    return caption_model, tokenizer, max_len, encoder


st.set_page_config(page_title="Image Captioning", page_icon="🖼️", layout="centered")
st.title("Image Captioning Model")
st.caption("Upload an image and generate a caption using your trained CNN-LSTM model.")

models_dir = Path("models")
latest_model = find_latest_checkpoint(models_dir)

default_model = str(latest_model) if latest_model else "models/model_9.h5"
model_path = st.text_input("Model checkpoint path", value=default_model)
tokenizer_path = st.text_input("Tokenizer path", value="tokenizer.p")
max_len_path = st.text_input("Max length path", value="max_length.txt")

missing_paths = [
    path for path in [model_path, tokenizer_path, max_len_path] if not os.path.exists(path)
]

if missing_paths:
    st.error(
        "Missing required artifact files:\n- "
        + "\n- ".join(missing_paths)
        + "\n\nRun `python main.py` first to generate/checkpoint model files."
    )
    st.stop()

caption_model, tokenizer, max_len, encoder = load_artifacts(
    model_path, tokenizer_path, max_len_path
)

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Input image", use_container_width=True)
    if st.button("Generate Caption", type="primary"):
        with st.spinner("Generating caption..."):
            features = extract_features(image, encoder)
            caption = generate_caption(caption_model, tokenizer, features, max_len)
        st.success("Caption generated")
        st.write(f"**Caption:** {caption if caption else '(empty result)'}")
