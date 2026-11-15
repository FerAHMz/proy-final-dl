"""Etapa 5 — Evaluación técnica en el día de prueba (día 10) con intervalos de confianza.

Incluye una iteración 0 de referencia (it0_base: predecir siempre el CTR de train), las
métricas de train/val/test por iteración, IC 95 % por bootstrap, curva de calibración y
curvas de entrenamiento. Salida: reports/tables/metricas.json, reports/figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from config import EXPERIMENTS, FIG_DIR, MODELS_DIR, TABLES_DIR  # noqa: E402
from utils import auc_fn, bootstrap_ci, load_meta, load_split, logloss_fn, metrics, save_json  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
LABELS = {"it0_base": "Base (CTR constante)", "it1_lr": "Regresión logística",
          "it2_widedeep": "Wide & Deep", "it3_deepfm": "DeepFM", "it4_deepfm_grande": "DeepFM grande", "it5_deepfm_reg": "DeepFM regularizado"}


def main() -> None:
    meta = load_meta()
    _, yte = load_split("test")
    _, yva = load_split("val")
    out = {}

    p0 = np.full(len(yte), meta["train_ctr"], dtype=np.float32)
    out["it0_base"] = {"test": metrics(yte, p0), "val": metrics(yva, np.full(len(yva), meta["train_ctr"])),
                       "test_ci": {"logloss": bootstrap_ci(yte, p0, logloss_fn, n_boot=100)},
                       "params": 1, "train_seconds": 0.0}
    np.save(MODELS_DIR / "it0_test_pred.npy", p0)

    for exp in EXPERIMENTS:
        hpath = MODELS_DIR / exp / "history.json"
        if not hpath.exists():
            print(f"(sin entrenar: {exp})")
            continue
        h = json.loads(hpath.read_text())
        best = next(e for e in h["epochs"] if e["epoch"] == h["best_epoch"])
        p = np.load(MODELS_DIR / exp / "test_pred.npy")
        out[exp] = {
            "train": best["train"], "val": best["val"], "test": best["test"],
            "test_ci": {"auc": bootstrap_ci(yte, p, auc_fn, n_boot=100),
                        "logloss": bootstrap_ci(yte, p, logloss_fn, n_boot=100)},
            "params": h["params"], "train_seconds": h["train_seconds"], "best_epoch": h["best_epoch"],
            "epoch_seconds": float(np.mean([e["epoch_seconds"] for e in h["epochs"]])),
            "device": h["device"],
        }
        print(f"{exp:18s} test AUC={best['test']['auc']:.4f} {out[exp]['test_ci']['auc']} "
              f"logloss={best['test']['logloss']:.4f}")
    save_json(out, TABLES_DIR / "metricas.json")

    # curvas de entrenamiento --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for exp in EXPERIMENTS:
        hpath = MODELS_DIR / exp / "history.json"
        if not hpath.exists():
            continue
        h = json.loads(hpath.read_text())
        st = h["steps"]
        axes[0].plot([s["step"] for s in st], [s["train_logloss"] for s in st], label=LABELS[exp], lw=1)
        ep = h["epochs"]
        axes[1].plot([e["epoch"] for e in ep], [e["val"]["logloss"] for e in ep], "o-", label=f"{LABELS[exp]} (val)")
        axes[1].plot([e["epoch"] for e in ep], [e["train"]["logloss"] for e in ep], "x--", alpha=0.5)
    axes[0].set_title("Log loss de entrenamiento por paso")
    axes[0].set_xlabel("paso")
    axes[1].set_title("Log loss por época (o = validación, x = train)")
    axes[1].set_xlabel("época")
    for a in axes:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "curvas_entrenamiento.png")
    plt.close(fig)

    # calibración --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    bins = np.linspace(0, 1, 21)
    for exp in EXPERIMENTS:
        f = MODELS_DIR / exp / "test_pred.npy"
        if not f.exists():
            continue
        p = np.load(f)
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(20):
            m = idx == b
            if m.sum() > 2000:
                xs.append(p[m].mean())
                ys.append(yte[m].mean())
        ax.plot(xs, ys, "o-", ms=3, label=LABELS[exp])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("pCTR predicho")
    ax.set_ylabel("CTR observado")
    ax.set_title("Calibración en prueba (día 10)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibracion.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
