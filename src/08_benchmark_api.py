"""Etapa 8 — Métricas operativas: latencia, throughput y costo por predicción.

- Latencia del modelo en CPU con 1 hilo (cómo correría en un pod de inferencia barato),
  por solicitud individual (p50/p95/p99) y throughput en lotes.
- Latencia extremo a extremo de la API (HTTP + validación + preprocesamiento + modelo) con el
  cliente de pruebas de FastAPI, usando impresiones reales del día de prueba.
- Costo por millón de predicciones suponiendo instancias c7g.large (2 vCPU, USD 0.0725/h on-demand).

Salida: reports/tables/benchmark.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EXPERIMENTS, FIELDS, MODELS_DIR, TABLES_DIR  # noqa: E402
from models import build_model  # noqa: E402
from utils import load_meta, load_split, save_json  # noqa: E402

USD_PER_HOUR_INSTANCE = 0.0725   # c7g.large
VCPUS = 2
SUBASTA_TIMEOUT_MS = 100         # típico en OpenRTB: el DSP tiene ~100 ms para responder


def bench_model(exp: str, X: np.ndarray, n_single: int = 2000) -> dict:
    meta = load_meta()
    cfg = EXPERIMENTS[exp]
    m = build_model(cfg["model"], meta["num_features"], len(FIELDS), **cfg)
    m.load_state_dict(torch.load(MODELS_DIR / exp / "model.pt", map_location="cpu"))
    m.eval()
    lat = []
    with torch.no_grad():
        for i in range(200):  # calentamiento
            m(torch.from_numpy(X[i:i + 1].astype(np.int64)))
        for i in range(n_single):
            xb = torch.from_numpy(X[i:i + 1].astype(np.int64))
            t = time.perf_counter()
            m(xb)
            lat.append((time.perf_counter() - t) * 1000)
        xb = torch.from_numpy(X[:4096].astype(np.int64))
        t = time.perf_counter()
        reps = 20
        for _ in range(reps):
            m(xb)
        thr = reps * 4096 / (time.perf_counter() - t)
    lat = np.array(lat)
    # costo: cada vCPU atiende solicitudes individuales a 1/p50 por segundo (supuesto conservador)
    per_vcpu = 1000 / np.median(lat)
    usd_per_million = USD_PER_HOUR_INSTANCE / (per_vcpu * VCPUS * 3600) * 1e6
    return {"p50_ms": float(np.median(lat)), "p95_ms": float(np.quantile(lat, 0.95)),
            "p99_ms": float(np.quantile(lat, 0.99)), "throughput_lote_por_s": float(thr),
            "solicitudes_por_s_por_vcpu": float(per_vcpu), "usd_por_millon": float(usd_per_million),
            "tamano_mb": (MODELS_DIR / exp / "model.pt").stat().st_size / 1e6}


def bench_api(n: int = 500) -> dict:
    from fastapi.testclient import TestClient

    from app.main import app
    final = MODELS_DIR / "final" / "demo_samples.json"
    samples = json.loads(final.read_text())
    with TestClient(app) as client:
        for s in samples[:50]:
            client.post("/predict", json={k: str(v) for k, v in s.items() if k != "click"})
        lat = []
        for i in range(n):
            s = samples[i % len(samples)]
            t = time.perf_counter()
            r = client.post("/predict", json={k: str(v) for k, v in s.items() if k != "click"})
            lat.append((time.perf_counter() - t) * 1000)
            assert r.status_code == 200
    lat = np.array(lat)
    return {"p50_ms": float(np.median(lat)), "p95_ms": float(np.quantile(lat, 0.95)),
            "p99_ms": float(np.quantile(lat, 0.99)),
            "dentro_de_timeout_100ms": float((lat < SUBASTA_TIMEOUT_MS).mean())}


def main() -> None:
    torch.set_num_threads(1)
    Xte, _ = load_split("test", mmap=True)
    X = np.asarray(Xte[:5000])
    res = {"supuestos": {"instancia": "c7g.large", "usd_hora": USD_PER_HOUR_INSTANCE, "vcpus": VCPUS,
                         "hilos_torch": 1, "timeout_subasta_ms": SUBASTA_TIMEOUT_MS}, "modelos": {}}
    for exp in EXPERIMENTS:
        if (MODELS_DIR / exp / "model.pt").exists():
            res["modelos"][exp] = bench_model(exp, X)
            r = res["modelos"][exp]
            print(f"{exp:18s} p50={r['p50_ms']:.3f}ms p99={r['p99_ms']:.3f}ms "
                  f"lote={r['throughput_lote_por_s']:,.0f}/s  ${r['usd_por_millon']:.4f}/M  {r['tamano_mb']:.0f}MB")
    if (MODELS_DIR / "final" / "model.pt").exists():
        res["api"] = bench_api()
        print(f"API e2e p50={res['api']['p50_ms']:.2f}ms p99={res['api']['p99_ms']:.2f}ms")
    save_json(res, TABLES_DIR / "benchmark.json")


if __name__ == "__main__":
    main()
