"""API de puja en tiempo real: recibe la solicitud de anuncio y responde pCTR + puja recomendada.

Ejecutar:  uvicorn app.main:app --port 8000      (demo web en http://localhost:8000)

Endpoints
    GET  /health             estado y versión del modelo
    POST /predict            una solicitud de puja → probabilidad de clic, puja (CPM) y decisión
    POST /predict/batch      hasta 1000 solicitudes
    GET  /sample             una impresión real del día de prueba (no vista en entrenamiento)
    GET  /metrics            métricas operativas en vivo (latencia p50/p99, OOV, pCTR medio)
"""
from __future__ import annotations

import collections
import json
import random
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .predictor import FINAL_DIR, ROOT, Predictor

app = FastAPI(title="CTR Bidder — DeepFM", version="1.0")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

predictor: Predictor | None = None
LAT = collections.deque(maxlen=5000)
STATS = {"n": 0, "oov_fields": 0, "p_sum": 0.0, "pujas": 0}
SAMPLES: list[dict] = []


class BidRequest(BaseModel):
    """Campos de una solicitud de puja (mismo esquema que Avazu). Faltantes → OOV."""
    hour: str | None = Field(None, description="YYMMDDHH", examples=["14103014"])
    hour_of_day: str | None = None
    weekday: str | None = None
    C1: str | None = None
    banner_pos: str | None = None
    site_id: str | None = None
    site_domain: str | None = None
    site_category: str | None = None
    app_id: str | None = None
    app_domain: str | None = None
    app_category: str | None = None
    device_id: str | None = None
    device_ip: str | None = None
    device_model: str | None = None
    device_type: str | None = None
    device_conn_type: str | None = None
    C14: str | None = None
    C15: str | None = None
    C16: str | None = None
    C17: str | None = None
    C18: str | None = None
    C19: str | None = None
    C20: str | None = None
    C21: str | None = None
    floor_cpm: float | None = Field(None, description="precio mínimo de la subasta (USD CPM)")


class BatchRequest(BaseModel):
    requests: list[BidRequest] = Field(..., max_length=1000)


@app.on_event("startup")
def load() -> None:
    global predictor
    if not (FINAL_DIR / "model.pt").exists():
        raise RuntimeError("No existe models/final/model.pt — corre src/07_exportar.py")
    predictor = Predictor()
    sample_file = FINAL_DIR / "demo_samples.json"
    if sample_file.exists():
        SAMPLES.extend(json.loads(sample_file.read_text()))


def _run(reqs: list[BidRequest]) -> list[dict]:
    if predictor is None:
        raise HTTPException(503, "modelo no cargado")
    t0 = time.perf_counter()
    raw = [r.model_dump(exclude_none=True) for r in reqs]
    floors = [r.pop("floor_cpm", None) for r in raw]
    out = predictor.predict(raw, floors)
    ms = (time.perf_counter() - t0) * 1000
    LAT.append(ms)
    for o in out:
        STATS["n"] += 1
        STATS["oov_fields"] += len(o["campos_desconocidos"])
        STATS["p_sum"] += o["p_click"]
        STATS["pujas"] += o["decision"] == "pujar"
        o["latencia_total_ms"] = round(ms, 3)
        o["modelo"] = predictor.version
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok" if predictor else "sin_modelo",
            "modelo": predictor.version if predictor else None,
            "alpha": predictor.alpha if predictor else None,
            "valor_por_clic": predictor.value["value_per_click"] if predictor else None}


@app.post("/predict")
def predict(req: BidRequest):
    return _run([req])[0]


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return {"resultados": _run(req.requests)}


@app.get("/sample")
def sample():
    if not SAMPLES:
        raise HTTPException(404, "sin muestras de demo")
    s = dict(random.choice(SAMPLES))
    return s


@app.get("/metrics")
def metrics():
    lat = sorted(LAT)
    q = (lambda x: lat[min(len(lat) - 1, int(x * len(lat)))]) if lat else (lambda x: None)
    n = max(STATS["n"], 1)
    return {"predicciones": STATS["n"], "latencia_p50_ms": q(0.5), "latencia_p99_ms": q(0.99),
            "pctr_medio": STATS["p_sum"] / n, "campos_oov_por_solicitud": STATS["oov_fields"] / n,
            "tasa_puja": STATS["pujas"] / n,
            "ctr_esperado_referencia": predictor.bundle.get("test_ctr") if predictor else None}


__all__ = ["app", "ROOT"]
