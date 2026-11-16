"""Etapa 7 — Exporta el modelo elegido para la API (models/final/).

Empaqueta pesos + vocabulario + offsets + parámetros de valor + α elegido en validación,
y 300 impresiones reales del día de prueba para la demo en vivo.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from config import (EXPERIMENTS, FINAL_EXPERIMENT, MODELS_DIR, PROCESSED_DIR,  # noqa: E402
                    RAW_CAT_FIELDS, SEED, TABLES_DIR, TIME_COL, VALUE_PARAMS)
from utils import load_meta  # noqa: E402


def main(exp: str = FINAL_EXPERIMENT) -> None:
    out = MODELS_DIR / "final"
    out.mkdir(exist_ok=True)
    shutil.copy(MODELS_DIR / exp / "model.pt", out / "model.pt")
    meta = load_meta()
    vocab = json.loads((PROCESSED_DIR / "vocab.json").read_text())
    valor = json.loads((TABLES_DIR / "valor.json").read_text()) if (TABLES_DIR / "valor.json").exists() else {}
    alpha = valor.get("comparacion", {}).get(exp, {}).get("alpha_opt_val", 1.0)
    bundle = {
        "version": f"{exp}-v1", "config": EXPERIMENTS[exp], "fields": meta["fields"],
        "offsets": meta["offsets"], "num_features": meta["num_features"], "vocab": vocab,
        "value_params": VALUE_PARAMS, "alpha": alpha, "test_ctr": meta["test_ctr"],
    }
    (out / "bundle.json").write_text(json.dumps(bundle))
    tbl = pq.read_table(PROCESSED_DIR / "raw_test.parquet").to_pandas()
    demo = tbl.sample(300, random_state=SEED)[["click", TIME_COL] + RAW_CAT_FIELDS]
    (out / "demo_samples.json").write_text(demo.to_json(orient="records"))
    size = (out / "model.pt").stat().st_size / 1e6
    print(f"Exportado {exp} → {out} (model.pt {size:.1f} MB, α={alpha})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else FINAL_EXPERIMENT)
