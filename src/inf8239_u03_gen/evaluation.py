"""Descarga verificada de Fashion-MNIST y partición independiente de prueba."""

import gzip, hashlib, json, struct, urllib.request
from pathlib import Path
import numpy as np
from .data import normalize_images, validate_images

ROOT = Path(__file__).resolve().parents[2]


def load_data():
    manifest = json.loads((ROOT / "docs/data_source.json").read_text(encoding="utf-8"))
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, expected in manifest["files"].items():
        path = raw / name
        if not path.exists():
            urllib.request.urlretrieve(manifest["base_url"] + "/" + name, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("SHA-256 incorrecto: " + name)
        data = gzip.decompress(path.read_bytes())
        magic, count = struct.unpack(">II", data[:8])
        if magic == 2051:
            _, _, h, w = struct.unpack(">IIII", data[:16])
            a = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, h, w)
        elif magic == 2049:
            a = np.frombuffer(data, dtype=np.uint8, offset=8)
        else:
            raise ValueError("Formato IDX inesperado")
        arrays[name] = a
    x = normalize_images(arrays["train-images-idx3-ubyte.gz"])
    y = arrays["train-labels-idx1-ubyte.gz"]
    xt = normalize_images(arrays["t10k-images-idx3-ubyte.gz"])
    yt = arrays["t10k-labels-idx1-ubyte.gz"]
    validate_images(x)
    validate_images(xt)
    if len(x) != 60000 or len(xt) != 10000:
        raise ValueError("Fashion-MNIST incompleto")
    return x, y, xt, yt


def split_indices(n=60000, seed=42):
    p = np.random.default_rng(seed).permutation(n)
    nv = n // 10
    return p[nv:], p[:nv]


def mse_per_image(x, y):
    return np.mean(np.square(np.asarray(x) - np.asarray(y)), axis=(1, 2, 3))


def kl_numpy(mean, logvar):
    return 0.5 * np.sum(np.square(mean) + np.exp(logvar) - 1 - logvar, axis=-1)


def interpolation_points(a, b, n=12):
    if n < 2:
        raise ValueError("Se necesitan al menos dos puntos")
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    return (1 - t) * np.asarray(a) + t * np.asarray(b)


def nearest_mse(queries, reference, batch=32):
    """Vecino exacto en píxeles, contra todo el entrenamiento; consultas por bloques."""
    q = np.asarray(queries).reshape(len(queries), -1).astype("float32")
    r = np.asarray(reference).reshape(len(reference), -1).astype("float32")
    r2 = np.sum(r * r, axis=1)
    indices = []
    distances = []
    for start in range(0, len(q), batch):
        a = q[start : start + batch]
        d = (
            np.maximum((a * a).sum(1)[:, None] + r2[None, :] - 2 * a @ r.T, 0)
            / q.shape[1]
        )
        idx = d.argmin(axis=1)
        indices.extend(idx.tolist())
        distances.extend(d[np.arange(len(idx)), idx].tolist())
    return np.array(indices), np.array(distances)


def diversity_mse(images):
    x = np.asarray(images).reshape(len(images), -1).astype("float64")
    n = len(x)
    if n < 2:
        raise ValueError("Se requieren dos muestras")
    return float(2 * np.var(x, axis=0, ddof=1).mean())
