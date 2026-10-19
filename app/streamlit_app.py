"""Inferencia local sobre decoder.keras; no entrena durante la visita."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Explorador VAE · Alexis De La Cruz", layout="wide")
st.title("Explorador del espacio latente")
st.caption("INF-8239 · Ejercicio 06 · Alexis De La Cruz (100805370)")
st.info(
    "Demostración con Fashion-MNIST. Las coordenadas no representan atributos con significado garantizado. Una muestra sintética no garantiza originalidad ni privacidad."
)


@st.cache_resource
def load_decoder():
    return tf.keras.models.load_model(ROOT / "models/decoder.keras")


if not (ROOT / "models/decoder.keras").exists():
    st.error(
        "Falta el decodificador. Ejecute el notebook o descargue los pesos del repositorio."
    )
    st.stop()
decoder = load_decoder()
if (ROOT / "reports/metrics.json").exists():
    meta = json.loads((ROOT / "reports/metrics.json").read_text())
    st.caption("Modelo seleccionado en validación: " + meta["chosen"])
mode = st.radio("Exploración", ["Punto latente", "Muestras nuevas"])
if mode == "Punto latente":
    x = st.slider("Latente X", -3.0, 3.0, 0.0, 0.1)
    y = st.slider("Latente Y", -3.0, 3.0, 0.0, 0.1)
    img = (
        decoder(np.array([[x, y]], dtype="float32"), training=False)
        .numpy()[0]
        .squeeze()
    )
    st.image(img, caption=f"Punto ({x:.1f}, {y:.1f})", clamp=True, width=280)
else:
    seed = int(
        st.number_input(
            "Semilla de muestreo",
            min_value=0,
            max_value=2147483647,
            value=31415,
            step=1,
        )
    )
    if st.button("Generar 16 muestras"):
        z = np.random.default_rng(seed).normal(size=(16, 2)).astype("float32")
        images = decoder(z, training=False).numpy().squeeze(-1)
        cols = st.columns(4)
        for i, img in enumerate(images):
            cols[i % 4].image(img, caption=f"Muestra {i + 1}", clamp=True, width=160)
        st.caption(
            "Ruido normal estándar N(0,I); sin selección manual de las imágenes."
        )
