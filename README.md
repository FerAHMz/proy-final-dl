# CTR Bidder: cuánto pagar por un anuncio móvil con DeepFM

Proyecto final de **CC3092 Deep Learning y Sistemas Inteligentes** (UVG, 2026).
Integrantes: **Fernando Hernández** y **Fernando Rueda**.

Construimos un sistema que predice si una persona va a darle clic a un anuncio en su celular
y, con esa predicción, decide **si conviene comprar ese espacio publicitario y cuánto pagar
por él**. El sistema funciona como una API que responde en menos de 1 ms. Comparamos un
**DeepFM** contra un **Wide & Deep** y una **regresión logística**, y traducimos cada mejora
de AUC y log loss a **dólares al mes** con un simulador de subastas de segundo precio.

## Resultados (día 10, que el modelo nunca vio)

| Iteración | AUC test [IC 95 %] | Log loss | Utilidad/mes* | Latencia p50 (CPU, 1 hilo) |
|---|---|---|---|---|
| it0 · Regla: CTR histórico del publisher | 0.726 | 0.406 | USD 124k | — |
| it1 · Regresión logística | 0.737 [0.735, 0.738] | 0.402 | USD 141k | 0.005 ms |
| it2 · Wide & Deep | 0.742 [0.740, 0.743] | 0.400 | USD 151k | 0.035 ms |
| it3 · DeepFM | 0.741 [0.739, 0.743] | 0.400 | USD 150k | 0.042 ms |
| it4 · DeepFM grande (31M parámetros) | 0.742 [0.740, 0.743] | 0.400 | USD 152k | 0.055 ms |
| **it5 · DeepFM regularizado (final)** | 0.741 [0.740, 0.743] | 0.401 | USD 148k | 0.042 ms |

\* Supuestos en [docs/modelo_de_valor.md](docs/modelo_de_valor.md). El modelo final se eligió
por utilidad en **validación** (USD 154k, la más alta). En prueba, los modelos profundos quedan
empatados entre sí y superan a la regresión logística por USD 7–11k al mes.
Cerca del punto de operación, **cada +0.01 de AUC ≈ USD 13.6k/mes**. La API responde en
0.72 ms (p50) de punta a punta.

## Estructura

```
├── src/
│   ├── config.py               rutas, campos, división temporal, iteraciones, supuestos de valor
│   ├── utils.py                carga de datos, lotes, métricas, bootstrap
│   ├── models.py               LR, Wide & Deep y DeepFM (PyTorch)
│   ├── value_model.py          función de valor (subasta de 2.º precio)
│   ├── 01_descarga.py          descarga de Avazu + división por día (1-8 / 9 / 10)
│   ├── 02_eda.py               análisis exploratorio → reports/figures/eda_*.png
│   ├── 03_preprocesamiento.py  vocabularios (solo train), OOV, codificación
│   ├── 04_entrenar.py          entrenamiento con early stopping en log loss de validación
│   ├── 05_evaluar.py           métricas + IC, curvas de entrenamiento, calibración
│   ├── 06_modelo_valor.py      curva de valor, punto de operación, sensibilidad, USD/mes
│   ├── 07_exportar.py          empaqueta el modelo final para la API
│   ├── 08_benchmark_api.py     latencia, throughput, costo por millón de predicciones
│   ├── 09_explicabilidad.py    importancia por permutación, interacciones FM, sesgo por grupo, deriva
│   └── run_pipeline.py         corre todo en orden
├── app/                        API FastAPI + demo web (app/static/index.html)
├── tests/                      pruebas de la función de valor y de las arquitecturas
├── docs/modelo_de_valor.md     función de valor, supuestos y fuentes
├── reports/                    figuras y tablas generadas (versionadas)
├── data/raw, data/processed    datos (no se versionan; los genera el pipeline)
├── models/                     pesos (no se versionan, pesan 85 MB)
├── codebook.md                 descripción de las variables del dataset
├── Dockerfile                  imagen de inferencia
└── requirements.txt
```

## Cómo reproducir

Requisitos: Python 3.11, ~10 GB libres en disco y 16 GB de RAM. El pipeline usa GPU si hay
(CUDA o Apple MPS). En un Apple M4 Pro, todo tarda unos 25 minutos.

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_pipeline.py          # descarga, EDA, entrena, evalúa, valor, exporta y explica
pytest -q
```

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\run_pipeline.py
```

Para retomar desde una etapa: `python src/run_pipeline.py --desde 4`.
Para usar el `train.gz` original de Kaggle: `python src/01_descarga.py --kaggle-file train.gz`.

Las semillas están fijas (`SEED = 42`). Las métricas de la tabla salen de
`reports/tables/metricas.json` y `reports/tables/valor.json`.

## API

```bash
uvicorn app.main:app --port 8000     # demo web en http://localhost:8000
```

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "hour":"14103014", "C1":"1005", "banner_pos":"0", "site_id":"1fbe01fe",
  "site_domain":"f3845767", "site_category":"28905ebd", "app_id":"ecad2386",
  "app_domain":"7801e8d9", "app_category":"07d7df22", "device_model":"8a4875bd",
  "device_type":"1", "device_conn_type":"0", "C14":"15706", "C15":"320", "C16":"50",
  "C17":"1722", "C18":"0", "C19":"35", "C20":"-1", "C21":"79", "floor_cpm": 2.5 }'
```

La respuesta trae `p_click`, `puja_cpm_usd`, `decision` (`pujar` / `no_pujar` según el
precio mínimo), `confianza` (baja si hay muchos campos nunca vistos) y
`campos_desconocidos`. También hay `/predict/batch`, `/sample` (una impresión real del día
de prueba), `/metrics` (latencia p50/p99, tasa de OOV y pCTR medio para monitoreo) y
`/health`.

Con Docker:
```bash
docker build -t ctr-bidder . && docker run -p 8000:8000 ctr-bidder
```

## Distribución del trabajo

| Integrante | Responsabilidades |
|---|---|
| **Fernando Hernández** | Pipeline de datos (descarga, división temporal, preprocesamiento, `config`/`utils`), EDA, codebook, modelo de valor (simulador de subastas, curva de valor, sensibilidad), documentación |
| **Fernando Rueda** | Arquitecturas (LR, Wide & Deep, DeepFM), entrenamiento e iteraciones, evaluación con IC, explicabilidad y monitoreo, API FastAPI + demo + Docker, benchmark de latencia, pruebas |

Los dos revisamos todas las partes para poder responder preguntas sobre cualquiera de ellas
en la presentación.
