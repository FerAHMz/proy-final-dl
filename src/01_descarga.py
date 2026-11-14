"""Etapa 1 — Descarga de Avazu y división TEMPORAL por día.

Fuente oficial: Kaggle, competencia "avazu-ctr-prediction" (train.gz). Como la descarga de
Kaggle requiere aceptar reglas y un token, por defecto se usa el mirror público
reczoo/Avazu_x4 de Hugging Face, que contiene exactamente las mismas 40,428,967 filas
repartidas al azar en train/valid/test. Nosotros IGNORAMOS esa partición aleatoria (filtra
información del futuro) y re-partimos por día usando la columna `hour` (YYMMDDHH):

    días 21-28 oct (1-8) → train      día 29 (9) → validación      día 30 (10) → prueba

Uso:
    python src/01_descarga.py                       # descarga del mirror HF
    python src/01_descarga.py --kaggle-file train.gz  # si ya bajaste train.gz desde Kaggle

Salida: data/processed/raw_{train,val,test}.parquet (columnas crudas como texto).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from config import (HF_URL, LABEL, PROCESSED_DIR, RAW_CAT_FIELDS, RAW_DIR,  # noqa: E402
                    TEST_DAYS, TIME_COL, TRAIN_DAYS, VAL_DAYS)

KEEP = [LABEL, TIME_COL] + RAW_CAT_FIELDS


def download_hf() -> list[Path]:
    zpath = RAW_DIR / "Avazu_x4.zip"
    if not zpath.exists():
        print(f"Descargando {HF_URL} (~1.3 GB)…")
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(zpath), HF_URL], check=True)
    out = RAW_DIR / "Avazu_x4"
    if not out.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    return sorted(out.rglob("*.csv"))


def split_of(day: int) -> str | None:
    if day in TRAIN_DAYS:
        return "train"
    if day in VAL_DAYS:
        return "val"
    if day in TEST_DAYS:
        return "test"
    return None


def main(files: list[Path]) -> None:
    schema = pa.schema([(c, pa.string()) for c in KEEP])
    writers = {s: pq.ParquetWriter(PROCESSED_DIR / f"raw_{s}.parquet", schema)
               for s in ("train", "val", "test")}
    counts = {s: 0 for s in writers}
    convert = pacsv.ConvertOptions(column_types={c: pa.string() for c in KEEP}, include_columns=KEEP)
    for f in files:
        print(f"Leyendo {f.name}…", flush=True)
        reader = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=64 << 20),
                                convert_options=convert)
        for batch in reader:
            tbl = pa.Table.from_batches([batch]).select(KEEP)
            day = pc.cast(pc.utf8_slice_codeunits(tbl[TIME_COL], 4, 6), pa.int32())
            for s, days in (("train", TRAIN_DAYS), ("val", VAL_DAYS), ("test", TEST_DAYS)):
                mask = pc.is_in(day, value_set=pa.array(days, pa.int32()))
                part = tbl.filter(mask)
                if part.num_rows:
                    writers[s].write_table(part.cast(schema))
                    counts[s] += part.num_rows
    for w in writers.values():
        w.close()
    total = sum(counts.values())
    for s, c in counts.items():
        print(f"{s:5s}: {c:>11,} filas ({c / total:.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-file", type=Path, default=None,
                    help="train.gz descargado de Kaggle (opcional)")
    args = ap.parse_args()
    main([args.kaggle_file] if args.kaggle_file else download_hf())
