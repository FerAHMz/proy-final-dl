"""Configuración central del proyecto: rutas, contratos de columnas, semillas e hiperparámetros.

Todas las etapas del pipeline (src/0X_*.py), la API (app/) y los tests leen de aquí,
así que cualquier cambio de ruta o de columna se hace en un solo lugar.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- rutas
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("AVAZU_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIG_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Fuente: Kaggle "Avazu Click-Through Rate Prediction" (train.gz, 40.4M filas).
# Mirror público sin autenticación (mismas filas, ya en CSV): reczoo/Avazu_x4 en Hugging Face.
HF_URL = "https://huggingface.co/datasets/reczoo/Avazu_x4/resolve/main/Avazu_x4.zip"
KAGGLE_COMPETITION = "avazu-ctr-prediction"

# ------------------------------------------------------------------- contrato de datos
LABEL = "click"
TIME_COL = "hour"  # formato YYMMDDHH en el dataset original

# Campos categóricos crudos de Avazu (sin `id`, que es un identificador único por fila).
RAW_CAT_FIELDS = [
    "C1", "banner_pos",
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
# Campos derivados de `hour` (la fecha completa no se usa: el modelo debe generalizar a días nuevos).
TIME_FIELDS = ["hour_of_day", "weekday"]
FIELDS = RAW_CAT_FIELDS + TIME_FIELDS

# División temporal (imita producción: se entrena con el pasado y se predice el día siguiente).
# Avazu cubre del 21 al 30 de octubre de 2014.
TRAIN_DAYS = list(range(21, 29))  # días 1-8
VAL_DAYS = [29]                   # día 9
TEST_DAYS = [30]                  # día 10

# Categorías con menos de MIN_COUNT apariciones en train se mapean a OOV (índice 0 de cada campo).
MIN_COUNT = 5

# ------------------------------------------------------------------ reproducibilidad
SEED = 42


def set_seed(seed: int = SEED) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------- iteraciones de modelo
# Cada iteración es una fila de la tabla de resultados (sección 4.5 del informe).
EXPERIMENTS = {
    "it1_lr": dict(model="lr", lr=1e-3, epochs=3, batch_size=4096, weight_decay=0.0),
    "it2_widedeep": dict(model="widedeep", embed_dim=16, mlp=[256, 128], dropout=0.2,
                         lr=1e-3, epochs=3, batch_size=4096, weight_decay=0.0),
    "it3_deepfm": dict(model="deepfm", embed_dim=16, mlp=[256, 128], dropout=0.2,
                       lr=1e-3, epochs=3, batch_size=4096, weight_decay=0.0),
    # it4: más capacidad (embeddings 24, MLP 400×3) → misma AUC, el doble de costo
    "it4_deepfm_grande": dict(model="deepfm", embed_dim=24, mlp=[400, 400, 400], dropout=0.3,
                              lr=1e-3, epochs=3, batch_size=4096, weight_decay=0.0,
                              embed_l2=1e-6),
    # it5: ataca el sobreajuste de una época: L2 fuerte en embeddings + lr menor
    "it5_deepfm_reg": dict(model="deepfm", embed_dim=16, mlp=[256, 128], dropout=0.3,
                           lr=5e-4, epochs=3, batch_size=4096, weight_decay=0.0,
                           embed_l2=3e-5),
}
FINAL_EXPERIMENT = "it5_deepfm_reg"

# ------------------------------------------------------------------ modelo de valor
# Todos los supuestos están documentados y justificados en docs/modelo_de_valor.md.
VALUE_PARAMS = dict(
    value_per_click=0.03,       # USD que vale un clic para el anunciante (CPC objetivo)
    market_k=1.2,               # competitividad: mediana del precio de mercado = k · V · CTR histórico del publisher
    market_sigma=0.5,           # dispersión log-normal del precio de mercado alrededor de esa mediana
    monthly_requests=300e6,     # solicitudes de puja evaluadas al mes por la campaña
    bid_multiplier=1.0,         # puja = multiplicador × pCTR × valor por clic (subasta de 2.º precio)
    infra_usd_per_million=0.0,  # se sobreescribe con la medición de latencia (ver 08_benchmark_api.py)
)
