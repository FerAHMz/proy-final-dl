"""Carga el modelo exportado y convierte una solicitud de puja cruda en pCTR + puja recomendada.

El artefacto vive en models/final/ (generado por src/07_exportar.py):
    model.pt        pesos del DeepFM
    bundle.json     configuración, vocabulario por campo, offsets, parámetros de valor, α óptimo
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models import build_model  # noqa: E402

FINAL_DIR = ROOT / "models" / "final"


class Predictor:
    def __init__(self, model_dir: Path = FINAL_DIR):
        bundle = json.loads((model_dir / "bundle.json").read_text())
        self.bundle = bundle
        self.fields: list[str] = bundle["fields"]
        self.offsets: dict[str, int] = bundle["offsets"]
        # vocab: campo -> {valor: índice local}; 0 = OOV
        self.lookup = {f: {v: i + 1 for i, v in enumerate(vals)} for f, vals in bundle["vocab"].items()}
        cfg = bundle["config"]
        torch.set_num_threads(1)  # una predicción = un hilo; se escala con réplicas/workers
        self.model = build_model(cfg["model"], bundle["num_features"], len(self.fields), **cfg)
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.model.eval()
        self.value = bundle["value_params"]
        self.alpha = bundle["alpha"]
        self.version = bundle["version"]

    # ------------------------------------------------------------------ preprocesamiento
    @staticmethod
    def derive(req: dict) -> dict:
        r = dict(req)
        hour = str(r.get("hour", ""))
        if len(hour) == 8 and hour.isdigit():  # YYMMDDHH (formato Avazu)
            import datetime as dt
            d = dt.date(2000 + int(hour[:2]), int(hour[2:4]), int(hour[4:6]))
            r.setdefault("hour_of_day", hour[6:8])
            r.setdefault("weekday", str(d.weekday()))
        return r

    def encode(self, reqs: list[dict]) -> tuple[np.ndarray, list[list[str]]]:
        X = np.zeros((len(reqs), len(self.fields)), dtype=np.int64)
        unknown = []
        for i, req in enumerate(reqs):
            r = self.derive(req)
            miss = []
            for j, f in enumerate(self.fields):
                local = self.lookup[f].get(str(r.get(f, "")), 0)
                if local == 0:
                    miss.append(f)
                X[i, j] = self.offsets[f] + local
            unknown.append(miss)
        return X, unknown

    # ------------------------------------------------------------------------- inferencia
    @torch.no_grad()
    def predict(self, reqs: list[dict], floor_cpm: list[float | None] | None = None) -> list[dict]:
        t0 = time.perf_counter()
        X, unknown = self.encode(reqs)
        p = torch.sigmoid(self.model(torch.from_numpy(X))).numpy()
        V = self.value["value_per_click"]
        out = []
        for i, pc in enumerate(p):
            bid_cpm = float(self.alpha * pc * V * 1000)
            floor = (floor_cpm or [None] * len(reqs))[i]
            decision = "pujar" if (floor is None or bid_cpm >= floor) else "no_pujar"
            n_unk = len(unknown[i])
            # comunicación de incertidumbre: muchos campos desconocidos = el modelo extrapola
            conf = "alta" if n_unk <= 2 else ("media" if n_unk <= 5 else "baja")
            out.append({
                "p_click": round(float(pc), 5),
                "valor_esperado_usd": round(float(pc * V), 6),
                "puja_cpm_usd": round(bid_cpm, 4),
                "decision": decision,
                "confianza": conf,
                "campos_desconocidos": unknown[i],
            })
        ms = (time.perf_counter() - t0) * 1000
        for o in out:
            o["latencia_modelo_ms"] = round(ms / len(out), 3)
        return out
