"""LAB10 completo: validación, test reservado y evidencia generativa no escogida."""

import os

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import json, time, shutil, platform, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import psutil
from threadpoolctl import threadpool_limits
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder
from inf8239_u03_gen.objectives import loss_terms, sample_latent
from inf8239_u03_gen.evaluation import (
    load_data,
    split_indices,
    mse_per_image,
    nearest_mse,
    diversity_mse,
    interpolation_points,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BATCH = 256


def batches(x, n=256):
    for i in range(0, len(x), n):
        yield i, x[i : i + n]


def predict(model, x):
    return np.concatenate(
        [model(b, training=False).numpy() for _, b in batches(x, 512)]
    )


def encode(model, x):
    outputs = [model(b, training=False) for _, b in batches(x, 512)]
    return tuple(np.concatenate([o[j].numpy() for o in outputs]) for j in (0, 1))


def train_ae(x, v, models):
    tf.keras.utils.set_random_seed(SEED)
    model = build_autoencoder(tf)
    opt = tf.keras.optimizers.Adam()
    history = []
    best = float("inf")
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)

    @tf.function
    def step(images):
        with tf.GradientTape() as tape:
            y = model(images, training=True)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(images, y))
        opt.apply_gradients(
            zip(tape.gradient(loss, model.trainable_weights), model.trainable_weights)
        )
        return loss

    for epoch in range(1, 7):
        order = rng.permutation(len(x))
        loss = 0
        for _, ids in batches(order):
            loss += float(step(x[ids])) * len(ids)
        vl = 0
        for _, b in batches(v, 512):
            vl += float(
                tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(b, model(b, training=False))
                )
            ) * len(b)
        vl /= len(v)
        history.append(
            {"epoch": epoch, "train_bce_mean": loss / len(x), "val_bce_mean": vl}
        )
        if vl < best:
            best = vl
            best_epoch = epoch
            model.save(models / "autoencoder.keras")
        print("AE", epoch, "val BCE", round(vl, 5), flush=True)
    return (
        tf.keras.models.load_model(models / "autoencoder.keras"),
        history,
        {
            "train_s": time.perf_counter() - start,
            "best_epoch": best_epoch,
            "val_objective": best,
        },
    )


def train_vae(x, v, beta, models):
    tf.keras.utils.set_random_seed(SEED)
    encoder, decoder = build_encoder_decoder(tf, 2)
    opt = tf.keras.optimizers.Adam()
    history = []
    best = float("inf")
    rng = np.random.default_rng(SEED)
    start = time.perf_counter()
    key = "vae_b1" if beta == 1 else "vae_b01"

    @tf.function
    def step(images, seed):
        with tf.GradientTape() as tape:
            mu, lv = encoder(images, training=True)
            z = sample_latent(mu, lv, seed)
            y = decoder(z, training=True)
            r, k = loss_terms(images, y, mu, lv)
            loss = tf.reduce_mean(r + beta * k)
        weights = encoder.trainable_weights + decoder.trainable_weights
        opt.apply_gradients(zip(tape.gradient(loss, weights), weights))
        return tf.reduce_mean(r), tf.reduce_mean(k)

    @tf.function
    def validate(images, seed):
        mu, lv = encoder(images, training=False)
        y = decoder(sample_latent(mu, lv, seed), training=False)
        r, k = loss_terms(images, y, mu, lv)
        return tf.reduce_mean(r), tf.reduce_mean(k)

    for epoch in range(1, 9):
        order = rng.permutation(len(x))
        tr = tk = 0
        for i, ids in batches(order):
            r, k = step(x[ids], tf.constant([SEED, epoch * 100000 + i], tf.int32))
            tr += float(r) * len(ids)
            tk += float(k) * len(ids)
        vr = vk = 0
        for i, b in batches(v, 512):
            r, k = validate(b, tf.constant([2026, i], tf.int32))
            vr += float(r) * len(b)
            vk += float(k) * len(b)
        tr /= len(x)
        tk /= len(x)
        vr /= len(v)
        vk /= len(v)
        # Misma regla de selección para beta distintos: R + KL sin ponderación.
        objective = vr + vk
        history.append(
            {
                "epoch": epoch,
                "train_reconstruction_sum": tr,
                "train_kl": tk,
                "train_total": tr + beta * tk,
                "val_reconstruction_sum": vr,
                "val_kl": vk,
                "val_total": vr + beta * vk,
                "val_elbo_negative": objective,
            }
        )
        if objective < best:
            best = objective
            best_epoch = epoch
            encoder.save(models / f"{key}_encoder.keras")
            decoder.save(models / f"{key}_decoder.keras")
        print(key, epoch, "val R", round(vr, 3), "KL", round(vk, 3), flush=True)
    return (
        (
            tf.keras.models.load_model(models / f"{key}_encoder.keras"),
            tf.keras.models.load_model(models / f"{key}_decoder.keras"),
        ),
        history,
        {
            "train_s": time.perf_counter() - start,
            "best_epoch": best_epoch,
            "val_objective": best,
            "beta": beta,
        },
    )


def test_model(x, model, encoder=None):
    # Reconstrucción VAE determinista desde mu, para pares comparables.
    losses = []
    ssims = []
    kls = []
    means = []
    start = time.perf_counter()
    for _, b in batches(x, 512):
        if encoder is None:
            y = model(b, training=False)
        else:
            mu, lv = encoder(b, training=False)
            y = model(mu, training=False)
            means.append(mu.numpy())
            _, kl = loss_terms(b, y, mu, lv)
            kls.extend(kl.numpy())
        losses.extend(mse_per_image(b, y.numpy()))
        ssims.extend(tf.image.ssim(b, y, max_val=1.0).numpy())
    duration = time.perf_counter() - start
    return {
        "test_mse": float(np.mean(losses)),
        "test_ssim": float(np.mean(ssims)),
        "test_kl": float(np.mean(kls)) if kls else None,
        "evaluation_s": duration,
        "active_latent_units": int(np.sum(np.var(np.concatenate(means), axis=0) > 0.01))
        if means
        else None,
    }, np.asarray(losses)


def plots(histories, arrays, metrics, train, neighbors):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    figdir = ROOT / "reports/figures"
    figdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), layout="constrained")
    h = pd.DataFrame(histories["ae"])
    ax[0].plot(h.epoch, h.train_bce_mean, label="ajuste")
    ax[0].plot(h.epoch, h.val_bce_mean, label="validación")
    ax[0].set_title("AE: BCE media por píxel")
    ax[0].legend()
    for key, color in [("vae_b1", "#2a78d6"), ("vae_b01", "#eb6834")]:
        h = pd.DataFrame(histories[key])
        label = "β=1" if key == "vae_b1" else "β=0.1"
        ax[1].plot(h.epoch, h.val_reconstruction_sum, label=label, color=color)
        ax[1].plot(h.epoch, h.train_reconstruction_sum, ls="--", color=color, alpha=0.6)
        ax[2].plot(h.epoch, h.val_kl, label=label, color=color)
        ax[2].plot(h.epoch, h.train_kl, ls="--", color=color, alpha=0.6)
    ax[1].set_title("VAE: BCE sumada en 784 píxeles")
    ax[2].set_title("VAE: KL sumada en 2 dimensiones")
    for a in ax:
        a.set_xlabel("Época")
    ax[1].legend()
    ax[2].legend()
    fig.savefig(figdir / "curves.png")
    plt.close(fig)
    fig, ax = plt.subplots(8, 8, figsize=(10, 9.5))
    collections = [
        arrays["originals"],
        arrays["ae"],
        arrays["vae"],
        arrays["generated"][:16],
    ]
    for block in range(2):
        for row, c in enumerate(collections):
            for col in range(8):
                i = block * 8 + col
                a = ax[block * 4 + row, col]
                a.imshow(c[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                a.set_xticks([])
                a.set_yticks([])
                if row == 0:
                    a.set_title("test " + str(arrays["panel_indices"][i]), fontsize=8)
                if col == 0:
                    a.set_ylabel(
                        ["Original", "AE", "VAE (μ)", "Prior nuevo"][row], fontsize=9
                    )
    fig.tight_layout()
    fig.savefig(figdir / "panel16.png")
    plt.close(fig)
    fig, ax = plt.subplots(1, 12, figsize=(12, 1.8))
    for i, (a, img) in enumerate(zip(ax, arrays["interpolation"])):
        a.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
        a.axis("off")
        a.set_title(f"t={i / 11:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "interpolation.png")
    plt.close(fig)
    fig, ax = plt.subplots(2, 8, figsize=(10, 3.3))
    for i in range(8):
        ax[0, i].imshow(arrays["generated"][i].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[1, i].imshow(train[neighbors[i]].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[0, i].set_title("Muestra " + str(i), fontsize=8)
        for a in ax[:, i]:
            a.set_xticks([])
            a.set_yticks([])
    ax[0, 0].set_ylabel("Prior nuevo")
    ax[1, 0].set_ylabel("Vecino ajuste")
    fig.tight_layout()
    fig.savefig(figdir / "neighbors.png")
    plt.close(fig)


def main():
    start = time.perf_counter()
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    reports = ROOT / "reports"
    models = ROOT / "models"
    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    x, y, xt, yt = load_data()
    ti, vi = split_indices()
    train = x[ti]
    valid = x[vi]
    np.savez(reports / "split_indices.npz", train=ti, validation=vi)
    histories = {}
    fit = {}
    ae, histories["ae"], fit["ae"] = train_ae(train, valid, models)
    vaes = {}
    for key, beta in [("vae_b1", 1.0), ("vae_b01", 0.1)]:
        vaes[key], histories[key], fit[key] = train_vae(train, valid, beta, models)
    chosen = min(vaes, key=lambda k: fit[k]["val_objective"])
    locked = {
        "chosen": chosen,
        "criterion": "menor BCE sumada + KL (peso común 1) en validación; checkpoint por misma regla",
        "seed": SEED,
        "time_unix": time.time(),
        "before_test": True,
    }
    (reports / "decision_locked.json").write_text(
        json.dumps(locked, indent=2), encoding="utf-8"
    )
    print("SELECCIÓN PREVIA A TEST:", chosen, flush=True)
    for key, h in histories.items():
        pd.DataFrame(h).to_csv(reports / f"history_{key}.csv", index=False)
    rows = []
    perclass = []
    for key in ["ae", "vae_b1", "vae_b01"]:
        enc, dec = (None, ae) if key == "ae" else vaes[key]
        result, errors = test_model(xt, dec, enc)
        # Latencia sin métricas SSIM: 3 repeticiones, lote 256, después de calentamiento.
        inp = xt[:256]
        times = []
        for _ in range(4):
            t = time.perf_counter()
            rec = (
                dec(inp, training=False)
                if enc is None
                else dec(enc(inp, training=False)[0], training=False)
            )
            rec.numpy()
            times.append((time.perf_counter() - t) * 1000 / 256)
        paths = (
            [models / "autoencoder.keras"]
            if enc is None
            else [models / f"{key}_encoder.keras", models / f"{key}_decoder.keras"]
        )
        result.update(
            model=key,
            **fit[key],
            parameters=dec.count_params() + (enc.count_params() if enc else 0),
            size_kib=sum(p.stat().st_size for p in paths) / 1024,
            latency_ms=float(np.median(times[1:])),
        )
        rows.append(result)
        for label in range(10):
            perclass.append(
                {
                    "model": key,
                    "class": label,
                    "n": int((yt == label).sum()),
                    "mse": float(errors[yt == label].mean()),
                }
            )
    enc, dec = vaes[chosen]
    shutil.copy2(models / f"{chosen}_decoder.keras", models / "decoder.keras")
    shutil.copy2(models / f"{chosen}_encoder.keras", models / "encoder.keras")
    rng = np.random.default_rng(2026)
    panel_ids = rng.choice(len(xt), 16, replace=False)
    originals = xt[panel_ids]
    mu, _ = encode(enc, originals)
    z = np.random.default_rng(31415).normal(size=(256, 2)).astype("float32")
    generated = predict(dec, z)
    arrays = {
        "originals": originals,
        "ae": predict(ae, originals),
        "vae": predict(dec, mu),
        "generated": generated,
        "panel_indices": panel_ids,
        "prior_z": z,
        "interpolation": predict(dec, interpolation_points(mu[0], mu[1])),
    }
    with threadpool_limits(limits=8):
        nn, dist = nearest_mse(generated, train)
        ref_ids = np.random.default_rng(2718).choice(len(valid), 256, replace=False)
        _, refdist = nearest_mse(valid[ref_ids], train)
    pd.DataFrame(
        {
            "sample": range(256),
            "train_partition_index": nn,
            "train_original_index": ti[nn],
            "nearest_mse": dist,
        }
    ).to_csv(reports / "nearest_neighbors.csv", index=False)
    diversity = {
        "n": 256,
        "generated_pairwise_mse": diversity_mse(generated),
        "real_validation_pairwise_mse": diversity_mse(valid[ref_ids]),
        "generated_nn_mse_median": float(np.median(dist)),
        "validation_nn_mse_median": float(np.median(refdist)),
        "threshold_validation_nn_p05": float(np.quantile(refdist, 0.05)),
        "fraction_generated_below_threshold": float(
            np.mean(dist < np.quantile(refdist, 0.05))
        ),
        "nearest_reference_count": len(train),
        "warning": "Distancias en píxeles: no FID, ni test formal de privacidad o prueba de memorización.",
    }
    np.savez_compressed(
        reports / "visual_evidence.npz",
        **arrays,
        nearest_train=nn,
        reference_validation_indices=vi[ref_ids],
    )
    pd.DataFrame(rows).to_csv(reports / "main_results.csv", index=False)
    pd.DataFrame(perclass).to_csv(reports / "errors_by_class.csv", index=False)
    plots(histories, arrays, rows, train, nn)
    mem = psutil.Process().memory_info()
    meta = {
        "seed": SEED,
        "train": len(train),
        "validation": len(valid),
        "test": len(xt),
        "split_sha256": hashlib.sha256(ti.tobytes() + vi.tobytes()).hexdigest(),
        "chosen": chosen,
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "system": platform.platform(),
        "gpu_visible": [d.name for d in tf.config.list_physical_devices("GPU")],
        "total_seconds": time.perf_counter() - start,
        "peak_working_set_mib": getattr(mem, "peak_wset", mem.rss) / 1024**2,
        "diversity": diversity,
        "latent_ae": 16,
        "latent_vae": 2,
        "epochs_ae": 6,
        "epochs_vae": 8,
        "batch_size": BATCH,
        "test_evaluated_after_selection": True,
    }
    (reports / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("DURACIÓN TOTAL:", meta["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
