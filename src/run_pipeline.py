"""Orquesta todo el pipeline de punta a punta (≈1-2 h en Apple M4 Pro / GPU modesta).

    python src/run_pipeline.py              # todo
    python src/run_pipeline.py --desde 4    # retomar desde la etapa 4 (entrenamiento)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent
STAGES = [
    (1, ["01_descarga.py"]),
    (2, ["02_eda.py"]),
    (3, ["03_preprocesamiento.py"]),
    (4, ["04_entrenar.py", "--exp", "all"]),
    (5, ["05_evaluar.py"]),
    (6, ["08_benchmark_api.py"]),   # latencia/costo por modelo (alimenta el modelo de valor)
    (7, ["06_modelo_valor.py"]),
    (8, ["07_exportar.py"]),
    (9, ["08_benchmark_api.py"]),   # ahora también mide la API extremo a extremo
    (10, ["09_explicabilidad.py"]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=1)
    a = ap.parse_args()
    for n, cmd in STAGES:
        if n < a.desde:
            continue
        print(f"\n===== Etapa {n}: {' '.join(cmd)} =====", flush=True)
        subprocess.run([sys.executable, str(SRC / cmd[0]), *cmd[1:]], check=True)
