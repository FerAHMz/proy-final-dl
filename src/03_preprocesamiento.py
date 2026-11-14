"""Etapa 3 — Preprocesamiento: vocabularios por campo y codificación a índices enteros.

Decisiones (justificadas en el informe, sec. 4.2):
- Todos los campos son categóricos (incluso C14-C21 y banner_pos son códigos anonimizados):
  se usan embeddings, no escalado numérico.
- `hour` (YYMMDDHH) → `hour_of_day` (0-23) y `weekday`. La fecha absoluta NO entra al modelo
  porque el día de prueba nunca aparece en train.
- El vocabulario se construye SOLO con train (días 1-8) para no filtrar información de
  validación/prueba. Categorías con < MIN_COUNT apariciones → índice OOV (0) del campo; así
  el modelo también aprende qué hacer con apps/dispositivos nunca vistos (caso real en producción).
- `id` se descarta (identificador único por fila).

Salida en data/processed/: {train,val,test}_X.npy (int32, N×F, índices globales),
{split}_y.npy (int8), vocab.json, meta.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from config import FIELDS, LABEL, MIN_COUNT, PROCESSED_DIR, RAW_CAT_FIELDS, TIME_COL  # noqa: E402


def derive_time(hour: pa.ChunkedArray) -> dict[str, pa.ChunkedArray]:
    hod = pc.utf8_slice_codeunits(hour, 6, 8)
    day = pc.cast(pc.utf8_slice_codeunits(hour, 4, 6), pa.int32()).to_numpy()
    # 20-oct-2014 fue lunes → weekday 0 = lunes
    wd = pc.cast(pa.array((day - 20) % 7), pa.string())
    return {"hour_of_day": hod, "weekday": wd}


def column(tbl: pa.Table, name: str) -> pa.ChunkedArray:
    if name in RAW_CAT_FIELDS:
        return tbl[name]
    col = derive_time(tbl[TIME_COL])[name]
    return col if isinstance(col, pa.ChunkedArray) else pa.chunked_array([col])


def build_vocab(path: Path) -> dict[str, list[str]]:
    vocab = {}
    pf = pq.ParquetFile(path)
    for f in FIELDS:
        cols = [TIME_COL] if f not in RAW_CAT_FIELDS else [f]
        tbl = pf.read(columns=cols)
        vc = pc.value_counts(column(tbl, f).combine_chunks())
        vals, cnts = vc.field("values").to_numpy(zero_copy_only=False), vc.field("counts").to_numpy()
        order = np.argsort(-cnts, kind="stable")
        keep = [str(v) for v, c in zip(vals[order], cnts[order]) if c >= MIN_COUNT]
        vocab[f] = keep
        print(f"  {f:17s} únicos={len(vals):>9,}  en vocab (≥{MIN_COUNT})={len(keep):>8,}", flush=True)
    return vocab


def encode(path: Path, vocab: dict[str, list[str]], offsets: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    X = np.empty((n, len(FIELDS)), dtype=np.int32)
    for j, f in enumerate(FIELDS):
        cols = [TIME_COL] if f not in RAW_CAT_FIELDS else [f]
        col = column(pf.read(columns=cols), f).combine_chunks()
        idx = pc.index_in(col, value_set=pa.array(vocab[f], pa.string()))
        local = pc.fill_null(pc.add(idx, 1), 0).to_numpy()  # 0 = OOV
        X[:, j] = local + offsets[f]
    y = pf.read(columns=[LABEL])[LABEL].to_numpy().astype(np.int8)
    return X, y


def main() -> None:
    print("Construyendo vocabulario con train…")
    vocab = build_vocab(PROCESSED_DIR / "raw_train.parquet")
    offsets, total = {}, 0
    for f in FIELDS:
        offsets[f] = total
        total += len(vocab[f]) + 1  # +1 por OOV
    meta = {"fields": FIELDS, "offsets": offsets, "field_sizes": {f: len(vocab[f]) + 1 for f in FIELDS},
            "num_features": total, "min_count": MIN_COUNT}
    for split in ("train", "val", "test"):
        X, y = encode(PROCESSED_DIR / f"raw_{split}.parquet", vocab, offsets)
        np.save(PROCESSED_DIR / f"{split}_X.npy", X)
        np.save(PROCESSED_DIR / f"{split}_y.npy", y)
        oov = float((X == np.array([offsets[f] for f in FIELDS])).mean())
        meta[f"{split}_rows"] = int(len(y))
        meta[f"{split}_ctr"] = float(y.mean())
        meta[f"{split}_oov_rate"] = oov
        print(f"{split}: {len(y):,} filas, CTR={y.mean():.4f}, tasa OOV={oov:.3%}", flush=True)
    with open(PROCESSED_DIR / "vocab.json", "w") as fh:
        json.dump(vocab, fh)
    with open(PROCESSED_DIR / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"num_features={total:,}")


if __name__ == "__main__":
    main()
