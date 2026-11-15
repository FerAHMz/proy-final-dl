# Codebook — Avazu Click-Through Rate Prediction

**Fuente:** Kaggle, competencia *Avazu Click-Through Rate Prediction* (2014),
<https://www.kaggle.com/competitions/avazu-ctr-prediction>. Usamos el archivo `train.gz` de la
competencia, que es el único que trae la etiqueta. Lo descargamos del mirror público
[`reczoo/Avazu_x4`](https://huggingface.co/datasets/reczoo/Avazu_x4), que tiene las mismas
40,428,967 filas en formato CSV. Descartamos la partición aleatoria del mirror y volvemos a
partir el dataset por día (ver abajo).

**Licencia / uso:** los datos se pueden usar con fines académicos y no comerciales, según las
reglas de la competencia. Por eso no los versionamos en el repo: `src/01_descarga.py` los
descarga. Avazu anonimizó todos los identificadores con hashes; el dataset no trae información
personal en claro.

**Granularidad:** una fila es una impresión de anuncio móvil (sitio web móvil o app) que se
mostró entre el 21 y el 30 de octubre de 2014.

## Variables

| Variable | Tipo | Descripción | Valores / rango (train) |
|---|---|---|---|
| `id` | string | Identificador único de la impresión. **Se descarta.** | 40.4M únicos |
| `click` | binaria | **Etiqueta**: 1 si hubo clic | CTR = 0.1714 en train, 0.1566 en val, 0.1693 en test |
| `hour` | string `YYMMDDHH` | Fecha y hora de la impresión. Solo se usa para partir los datos y para derivar dos campos | 14102100 – 14103023 |
| `hour_of_day` | categórica (derivada) | Hora del día, de 0 a 23 | 24 |
| `weekday` | categórica (derivada) | Día de la semana (0 = lunes) | 7 |
| `C1` | categórica anonimizada | Variable de contexto sin documentar | 7 (91.7 % = 1005) |
| `banner_pos` | categórica | Posición del banner en pantalla | 7 (0: 72 %, 1: 28 %) |
| `site_id`, `site_domain`, `site_category` | categórica (hash) | Sitio web móvil donde se mostró el anuncio | 4,551 / 7,299 / 26 |
| `app_id`, `app_domain`, `app_category` | categórica (hash) | App donde se mostró el anuncio. `app_id = ecad2386` indica tráfico web | 8,095 / 539 / 36 |
| `device_id` | categórica (hash) | Identificador del dispositivo. El valor `a99f214a` (82.6 %) funciona como "desconocido" | 2.2M |
| `device_ip` | categórica (hash) | IP del dispositivo, ya con hash | 5.6M |
| `device_model` | categórica (hash) | Modelo del teléfono | 8,036 |
| `device_type` | categórica | Tipo de dispositivo | 5 (1: 92 %) |
| `device_conn_type` | categórica | Tipo de conexión | 4 (0: 87 %) |
| `C14`–`C21` | categóricas anonimizadas | Atributos del anuncio o de la campaña (C15 y C16 parecen ancho y alto del banner) | 2,252 / 8 / 9 / 372 / 4 / 63 / 172 / 52 |

Ninguna columna tiene nulos. Los valores "desconocidos" del dataset ya vienen como categorías
propias (por ejemplo `a99f214a` en `device_id` y `-1` en `C20`).

## Clases

| Índice | Clase | Frecuencia en train |
|---|---|---|
| 0 | sin clic | 82.86 % |
| 1 | clic | 17.14 % |

El desbalance es moderado (1 : 4.8). No balanceamos las clases porque la puja necesita
probabilidades calibradas (ver informe, sección 4.4).

## División (temporal, imita producción)

| Split | Días (oct-2014) | Filas | CTR | Tasa OOV* |
|---|---|---|---|---|
| train | 21 – 28 (días 1-8) | 32,377,421 | 0.1714 | 1.4 % |
| val | 29 (día 9) | 3,832,608 | 0.1566 | 4.9 % |
| test | 30 (día 10) | 4,218,938 | 0.1693 | 7.4 % |

\* Fracción de celdas (fila × campo) cuyo valor no está en el vocabulario de train.

## Transformaciones (`src/03_preprocesamiento.py`)

1. `hour` se convierte en `hour_of_day` y `weekday`. La fecha absoluta no entra al modelo.
2. Construimos un vocabulario por campo **solo con train**. Las categorías con menos de 5
   apariciones (`MIN_COUNT`) van a un índice OOV propio de cada campo. Con esto el vocabulario
   baja de 7.8M a 1.24M de categorías: por ejemplo, `device_ip` pasa de 5.6M a 995k y
   `device_id` de 2.2M a 223k.
3. Cada campo se codifica como un índice entero global (índice local + offset del campo), así
   que todos los campos comparten una sola tabla de embeddings de 1,239,851 filas.
4. No hay escalado numérico: todos los campos son categóricos y se representan con embeddings.
