"""Funciones compartidas: carga de splits procesados, lotes, métricas e incertidumbre."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score

from config import PROCESSED_DIR, SEED


# ------------------------------------------------------------------------ datos
def load_split(split: str, mmap: bool = False) -> tuple[np.ndarray, np.ndarray]:
    mode = "r" if mmap else None
    X = np.load(PROCESSED_DIR / f"{split}_X.npy", mmap_mode=mode)
    y = np.load(PROCESSED_DIR / f"{split}_y.npy", mmap_mode=mode)
    return X, y


def load_meta() -> dict:
    with open(PROCESSED_DIR / "meta.json") as f:
        return json.load(f)


def iterate_batches(X: np.ndarray, y: np.ndarray | None, batch_size: int,
                    shuffle: bool = False, seed: int = SEED):
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n) if shuffle else None
    for start in range(0, n, batch_size):
        sl = idx[start:start + batch_size] if shuffle else slice(start, start + batch_size)
        xb = torch.from_numpy(np.ascontiguousarray(X[sl]).astype(np.int64))
        yb = None if y is None else torch.from_numpy(np.ascontiguousarray(y[sl]).astype(np.float32))
        yield xb, yb


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: np.ndarray, device, batch_size: int = 32768) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in iterate_batches(X, None, batch_size):
        out.append(torch.sigmoid(model(xb.to(device))).float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------- métricas
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "ctr_real": float(y.mean()),
        "ctr_pred": float(p.mean()),
        # >1 = el modelo sobreestima la probabilidad de clic (puja de más)
        "calibracion": float(p.mean() / y.mean()),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn, n_boot: int = 200, sample: int = 500_000,
                 seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    """IC por bootstrap (submuestras de `sample` filas con reemplazo)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), size=min(sample, len(y)))
        vals.append(fn(y[i], p[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def auc_fn(y, p):
    return roc_auc_score(y, p)


def logloss_fn(y, p):
    return log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])


@contextmanager
def timer():
    t = {"start": time.perf_counter()}
    yield t
    t["seconds"] = time.perf_counter() - t["start"]


def save_json(obj, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
