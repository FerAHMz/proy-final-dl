"""Etapa 9 — Explicabilidad, sesgos por grupo y monitoreo de deriva.

1. Importancia por permutación de cada campo (caída de AUC y aumento de log loss en prueba).
2. Fuerza de interacción de 2.º orden aprendida por la FM entre pares de campos (heatmap):
   evidencia de qué cruces "descubrió solo" el DeepFM.
3. Desempeño por grupo (tipo de dispositivo, conexión, posición, sitio vs app): AUC,
   calibración y utilidad por cada 1000 solicitudes → ¿el modelo es peor para algún grupo?
4. Monitoreo: PSI de la distribución de pCTR y tasa de OOV entre validación (día 9) y prueba
   (día 10), umbrales de alerta del plan de monitoreo.

Salida: reports/tables/explicabilidad.json, reports/figures/xai_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from config import (EXPERIMENTS, FIELDS, FIG_DIR, FINAL_EXPERIMENT, MODELS_DIR,  # noqa: E402
                    PROCESSED_DIR, SEED, TABLES_DIR, get_device)
from models import build_model  # noqa: E402
from utils import load_meta, load_split, predict_proba, save_json  # noqa: E402
from value_model import ValueParams, market_prices, publisher_ctr  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def psi(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    pb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def main(exp: str = FINAL_EXPERIMENT) -> None:
    device = get_device()
    meta = load_meta()
    cfg = EXPERIMENTS[exp]
    model = build_model(cfg["model"], meta["num_features"], len(FIELDS), **cfg).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / exp / "model.pt", map_location=device))
    model.eval()
    Xte, yte = load_split("test")
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(yte), 500_000, replace=False)
    Xs, ys = Xte[sub], yte[sub]
    p_base = predict_proba(model, Xs, device)
    auc0, ll0 = roc_auc_score(ys, p_base), log_loss(ys, p_base)
    res: dict = {"modelo": exp}

    # 1) importancia por permutación ---------------------------------------------------------
    imp = {}
    for j, f in enumerate(FIELDS):
        Xp = Xs.copy()
        Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        p = predict_proba(model, Xp, device)
        imp[f] = {"delta_auc": float(auc0 - roc_auc_score(ys, p)), "delta_logloss": float(log_loss(ys, p) - ll0)}
    res["importancia_permutacion"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]["delta_auc"]))
    order = list(res["importancia_permutacion"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(order[::-1], [imp[f]["delta_auc"] for f in order[::-1]], color="#2a6fdb")
    ax.set_xlabel("caída de AUC al permutar el campo")
    ax.set_title(f"Importancia por permutación ({exp}, prueba)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xai_importancia.png")
    plt.close(fig)
    print("Top campos:", ", ".join(f"{f} ({imp[f]['delta_auc']:.4f})" for f in order[:6]))

    # 2) interacciones FM --------------------------------------------------------------------
    if hasattr(model, "field_interaction_strength"):
        with torch.no_grad():
            M = model.field_interaction_strength(torch.from_numpy(Xs[:50_000].astype(np.int64)).to(device)).cpu().numpy()
        np.fill_diagonal(M, 0)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(len(FIELDS)), FIELDS, rotation=90, fontsize=7)
        ax.set_yticks(range(len(FIELDS)), FIELDS, fontsize=7)
        fig.colorbar(im, fraction=0.04)
        ax.set_title("|<v_i, v_j>| medio: interacciones aprendidas por la FM")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "xai_interacciones_fm.png")
        plt.close(fig)
        iu = np.triu_indices(len(FIELDS), 1)
        top = np.argsort(-M[iu])[:10]
        res["top_interacciones"] = [{"par": f"{FIELDS[iu[0][k]]} × {FIELDS[iu[1][k]]}", "fuerza": float(M[iu][k])}
                                    for k in top]
        print("Top interacciones:", [t["par"] for t in res["top_interacciones"][:5]])

    # 3) desempeño por grupo -----------------------------------------------------------------
    p_all = np.load(MODELS_DIR / exp / "test_pred.npy")
    p_lr = np.load(MODELS_DIR / "it1_lr" / "test_pred.npy") if (MODELS_DIR / "it1_lr" / "test_pred.npy").exists() else None
    raw = pq.read_table(PROCESSED_DIR / "raw_test.parquet",
                        columns=["device_type", "device_conn_type", "banner_pos", "app_id", "hour"]).to_pandas()
    raw["origen"] = np.where(raw["app_id"] == "ecad2386", "sitio_web", "app")
    raw["franja"] = raw["hour"].str[6:8].astype(int).map(lambda h: "madrugada" if h < 6 else "mañana" if h < 12
                                                       else "tarde" if h < 18 else "noche")
    vp = ValueParams()
    prices = market_prices(publisher_ctr("test"), vp, seed=SEED)
    alpha = 1.0
    vj = TABLES_DIR / "valor.json"
    if vj.exists():
        alpha = json.loads(vj.read_text())["comparacion"].get(exp, {}).get("alpha_opt_val", 1.0)
    win = alpha * p_all * vp.value_per_click > prices
    util = np.where(win, yte * vp.value_per_click - prices, 0.0)
    groups = {}
    for col in ["device_type", "device_conn_type", "banner_pos", "origen", "franja"]:
        groups[col] = {}
        for g, idx in raw.groupby(col).indices.items():
            if len(idx) < 20_000 or yte[idx].min() == yte[idx].max():
                continue
            row = {"n": int(len(idx)), "ctr": float(yte[idx].mean()),
                   "auc": float(roc_auc_score(yte[idx], p_all[idx])),
                   "calibracion": float(p_all[idx].mean() / yte[idx].mean()),
                   "utilidad_por_1000": float(util[idx].mean() * 1000)}
            if p_lr is not None:
                row["auc_lr"] = float(roc_auc_score(yte[idx], p_lr[idx]))
            groups[col][str(g)] = row
    res["grupos"] = groups
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, col in zip(axes, ["device_type", "device_conn_type", "origen"]):
        g = groups[col]
        ks = list(g)
        ax.bar([k for k in ks], [g[k]["auc"] for k in ks], color="#2a6fdb", label="DeepFM")
        if p_lr is not None:
            ax.plot(ks, [g[k]["auc_lr"] for k in ks], "o", color="#e8833a", label="LR")
        ax.set_ylim(0.5, 0.85)
        ax.set_title(f"AUC por {col}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xai_grupos.png")
    plt.close(fig)

    # 4) monitoreo: deriva día 9 → día 10 ---------------------------------------------------
    p_val = np.load(MODELS_DIR / exp / "val_pred.npy")
    Xva, yva = load_split("val", mmap=True)
    offs = np.array([meta["offsets"][f] for f in FIELDS])
    oov_val = float((np.asarray(Xva[:1_000_000]) == offs).mean())
    oov_te = float((Xte[:1_000_000] == offs).mean())
    res["monitoreo"] = {
        "psi_pctr_val_vs_test": psi(p_val, p_all),
        "oov_val": oov_val, "oov_test": oov_te,
        "calibracion_val": float(p_val.mean() / yva.mean()),
        "calibracion_test": float(p_all.mean() / yte.mean()),
        "umbrales_alerta": {"psi": 0.2, "calibracion_fuera_de": [0.9, 1.1], "oov_aumento_relativo": 0.25},
    }
    print("Monitoreo:", res["monitoreo"])
    save_json(res, TABLES_DIR / "explicabilidad.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else FINAL_EXPERIMENT)
