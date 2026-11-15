"""Etapa 4 — Entrenamiento de una iteración (LR, Wide & Deep o DeepFM).

Uso:
    python src/04_entrenar.py --exp it3_deepfm            # una iteración de config.EXPERIMENTS
    python src/04_entrenar.py --exp all                   # todas
    python src/04_entrenar.py --exp it3_deepfm --frac 0.1 # prueba rápida con 10 % de train

Pérdida: entropía cruzada binaria (log loss). Es la pérdida alineada con el negocio porque la
puja es proporcional a la probabilidad (puja = pCTR × valor del clic): necesitamos
probabilidades CALIBRADAS, no solo un buen ordenamiento. Una pérdida ponderada por clase
mejoraría el recall pero descalibraría pCTR y nos haría pujar de más (ver informe, sec. 4.4).

Salidas en models/<exp>/: model.pt, history.json, val_pred_e*.npy, test_pred_e*.npy
(predicciones por época para construir la curva valor ↔ métrica técnica).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import EXPERIMENTS, FIELDS, MODELS_DIR, get_device, set_seed  # noqa: E402
from models import build_model, count_params  # noqa: E402
from utils import (iterate_batches, load_meta, load_split, metrics,  # noqa: E402
                   predict_proba, save_json)


def train_one(exp: str, frac: float = 1.0, log_every: int = 500) -> dict:
    cfg = EXPERIMENTS[exp]
    set_seed()
    device = get_device()
    meta = load_meta()
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    Xte, yte = load_split("test")
    if frac < 1.0:
        n = int(len(Xtr) * frac)
        sel = np.random.default_rng(0).choice(len(Xtr), n, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]

    model = build_model(cfg["model"], meta["num_features"], len(FIELDS), **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    loss_fn = nn.BCEWithLogitsLoss()
    embed_l2 = cfg.get("embed_l2", 0.0)

    out_dir = MODELS_DIR / exp
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"[{exp}] device={device} params={count_params(model):,} train={len(Xtr):,}")
    hist = {"exp": exp, "config": cfg, "params": count_params(model), "train_rows": len(Xtr),
            "device": str(device), "steps": [], "epochs": []}
    best = {"val_logloss": np.inf, "epoch": -1}
    t0 = time.perf_counter()
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running, n_run = 0.0, 0
        t_ep = time.perf_counter()
        for xb, yb in iterate_batches(Xtr, ytr, cfg["batch_size"], shuffle=True, seed=epoch):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if embed_l2 > 0 and hasattr(model, "embedding"):
                # L2 solo sobre las filas de embedding usadas en el lote (regularización tipo FTRL)
                loss = loss + embed_l2 * model.embedding.emb(xb).pow(2).sum() / len(xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            n_run += len(xb)
            step += 1
            if step % log_every == 0:
                hist["steps"].append({"step": step, "epoch": epoch, "train_logloss": running / n_run})
                print(f"  ep{epoch} step {step} train_logloss={running / n_run:.4f}", flush=True)
                running, n_run = 0.0, 0
        ep_time = time.perf_counter() - t_ep

        # métricas por época (train sobre muestra de 1M para medir sobreajuste)
        sub = np.random.default_rng(1).choice(len(Xtr), min(1_000_000, len(Xtr)), replace=False)
        m_tr = metrics(ytr[sub], predict_proba(model, Xtr[sub], device))
        p_va = predict_proba(model, Xva, device)
        p_te = predict_proba(model, Xte, device)
        m_va, m_te = metrics(yva, p_va), metrics(yte, p_te)
        np.save(out_dir / f"val_pred_e{epoch}.npy", p_va.astype(np.float32))
        np.save(out_dir / f"test_pred_e{epoch}.npy", p_te.astype(np.float32))
        rec = {"epoch": epoch, "epoch_seconds": ep_time, "train": m_tr, "val": m_va, "test": m_te}
        hist["epochs"].append(rec)
        print(f"  ep{epoch} ({ep_time:.0f}s) train auc={m_tr['auc']:.4f} ll={m_tr['logloss']:.4f} | "
              f"val auc={m_va['auc']:.4f} ll={m_va['logloss']:.4f} | test auc={m_te['auc']:.4f}", flush=True)

        # early stopping sobre logloss de validación (el test NO se usa para decidir nada)
        if m_va["logloss"] < best["val_logloss"]:
            best = {"val_logloss": m_va["logloss"], "epoch": epoch}
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            print(f"  val logloss empeoró → early stopping (mejor época = {best['epoch']})")
            break

    hist["best_epoch"] = best["epoch"]
    hist["train_seconds"] = time.perf_counter() - t0
    shutil.copy(out_dir / f"val_pred_e{best['epoch']}.npy", out_dir / "val_pred.npy")
    shutil.copy(out_dir / f"test_pred_e{best['epoch']}.npy", out_dir / "test_pred.npy")
    save_json(hist, out_dir / "history.json")
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()
    exps = list(EXPERIMENTS) if args.exp == "all" else args.exp.split(",")
    for e in exps:
        train_one(e, args.frac)
