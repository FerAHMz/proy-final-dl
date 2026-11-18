# Modelo de valor: de pCTR a dólares

Código: [`src/value_model.py`](../src/value_model.py) (la función de valor) y
[`src/06_modelo_valor.py`](../src/06_modelo_valor.py) (las curvas, la sensibilidad y la
comparación). Resultados: `reports/tables/valor.json` y `reports/figures/valor_*.png`.

## 1. Escenario

Un **DSP** (*demand-side platform*) compra impresiones móviles en tiempo real para un
anunciante que paga por desempeño. Cada vez que una app o un sitio va a mostrar un anuncio,
el exchange lanza una subasta (OpenRTB) y el DSP tiene unos **100 ms** para responder con una
puja. Nuestro sistema recibe la solicitud, estima la probabilidad de clic (pCTR) con DeepFM y
devuelve la puja recomendada.

## 2. Función de valor

En cada solicitud *i*:

```
puja_i     = α · pCTR_i · V
gana_i     = puja_i > m_i                  (subasta de segundo precio: si ganamos, pagamos m_i)
utilidad_i = gana_i · (click_i · V − m_i)

Valor mensual = (Σ_i utilidad_i) · (volumen mensual / impresiones del día de prueba)
              − costo de reentrenamiento − costo de inferencia
```

Leída como una matriz de confusión (la decisión es comprar o no comprar):

| | hubo clic | no hubo clic |
|---|---|---|
| **compramos** | VP: + (V − m_i) | FP: − m_i (gasto sin retorno) |
| **no compramos** | FN: se pierden (V − m_i) si m_i < V | VN: 0 |

**Por qué importan el AUC y el log loss.** En una subasta de segundo precio, la puja óptima es
el valor esperado de la impresión (Vickrey). Si el pCTR está calibrado, eso equivale a
`puja = pCTR · V`. El **AUC** mide si el modelo ordena bien las impresiones, es decir, si
compra las buenas y deja pasar las malas. El **log loss** mide además si la probabilidad tiene
la escala correcta, es decir, si puja el monto justo. Un modelo con buen AUC pero mal
calibrado puja de más o de menos, y eso cuesta dinero (ver la sección 6).

## 3. Supuestos y fuentes

| Parámetro | Valor base | Justificación | ¿Estimado? |
|---|---|---|---|
| `V`: valor de un clic | USD 0.03 | Con el CTR de Avazu (~17 %) da un valor esperado de ~USD 5 por mil impresiones, dentro del rango típico de CPM de display móvil (USD 1–10). El CTR de Avazu es mucho más alto que el típico de la industria (~0.5–1 %), probablemente porque los datos están muestreados, así que fijamos V para que la economía por impresión sea realista. | Sí |
| Precio de mercado `m_i` | log-normal: mediana `k·V·CTR_pub`, σ = 0.5 | La literatura de RTB (iPinYou, Zhang et al. 2014) modela los precios ganadores con distribuciones log-normales de cola pesada. Suponemos rivales **informados**: conocen el CTR histórico de cada publisher (sitio o app). | Sí (simulado) |
| `k`: competitividad | 1.2 | Con k = 1.2, un pujador sin información (CTR constante) queda en break-even (ROI ≈ 3 %), como se espera en un mercado competitivo. La mediana resultante es de ~USD 5.3 CPM. | Sí |
| `σ`: dispersión de precios | 0.5 | Variabilidad entre subastas por presupuesto, frequency caps, etc. | Sí |
| Volumen | 300M solicitudes/mes | ~10M solicitudes/día para una campaña regional mediana. El valor escala linealmente con este número. | Sí |
| `α`: multiplicador de puja | se elige en validación | Es el punto de operación: equivale a mover el umbral de decisión. | Se optimiza |
| Reentrenamiento | g5.xlarge, USD 1.006/h, 30 veces/mes | Precio on-demand de AWS. Reentrenamiento diario con una ventana móvil de 8 días. | Precio público |
| Inferencia | c7g.large, USD 0.0725/h, 2 vCPU | Precio on-demand de AWS. La latencia se mide en `08_benchmark_api.py`. | Precio público + medición |

Para que la comparación sea justa, todos los modelos se evalúan con **los mismos precios
simulados** (semilla fija), y el punto de operación α se elige en el día de validación y se
reporta en el día de prueba.

## 4. Referencias sin deep learning (cómo se resuelve hoy)

- **CTR constante**: pujar lo mismo en todas las impresiones. ≈ USD 8k/mes; está en
  break-even por construcción.
- **Regla: CTR histórico del publisher**: pujar según el CTR promedio del sitio o la app en los
  últimos 8 días. Es la heurística típica de un DSP sin modelo. ≈ USD 124k/mes.
- **Regresión logística** (modelo clásico): ≈ USD 141k/mes.

## 5. Resultados (día de prueba, IC 95 % por bootstrap)

Ver `reports/tables/valor.json`. En resumen:

| Modelo | AUC | Log loss | Utilidad/mes | Δ vs LR |
|---|---|---|---|---|
| Regla: CTR del publisher | 0.726 | 0.406 | USD 124k | −16k |
| Regresión logística | 0.737 | 0.402 | USD 141k | — |
| Wide & Deep | 0.742 | 0.400 | USD 151k | +10k |
| DeepFM (final, it5) | 0.741 | 0.401 | USD 148k | +7k |

Elegimos el modelo final por **utilidad en validación**, donde el DeepFM regularizado fue el
mejor (USD 154k). En prueba, los modelos profundos quedan estadísticamente empatados entre sí
y todos superan a LR con intervalos que no se traslapan.

**Pendiente de la curva de valor:** cerca del punto de operación, cada +0.01 de AUC vale
≈ **USD 13.6k/mes** (`valor_vs_metrica.png`).

## 6. Sensibilidad

Recalculamos la ventaja del DeepFM sobre LR moviendo cada supuesto ×0.5, ×0.8, ×1.2 y ×1.5
(`valor_sensibilidad.png`). La ventaja sigue siendo positiva en **todos** los escenarios,
incluido uno en el que los precios no dependen de la calidad de la impresión. Lo que más la
mueve es la competitividad `k`: en un mercado más agresivo (×1.5), el margen de todos se
reduce y la ventaja cae a ~USD 1.6k/mes. El valor por clic y el volumen la escalan
linealmente.

## 7. Limitaciones

- Los precios son simulados: Avazu no trae precios ganadores. Con logs reales del DSP, la
  función `market_prices` se reemplaza por los precios observados y el resto del cálculo no
  cambia.
- Suponemos subasta de segundo precio. Muchos exchanges ya usan primer precio, lo que exige
  *bid shading*: el modelo de valor se extiende cambiando el precio pagado por la puja.
- No modelamos el presupuesto diario: suponemos que la campaña puede comprar todo lo
  rentable.
