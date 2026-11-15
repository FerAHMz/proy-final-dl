"""Modelo de valor: convierte el pCTR (métrica técnica) en utilidad mensual (métrica de negocio).

Escenario: un DSP (demand-side platform) puja por impresiones móviles para un anunciante que
paga por desempeño. Cada solicitud de puja es una subasta de SEGUNDO precio:

    puja_i      = α · pCTR_i · V                    (V = valor de un clic, α = multiplicador)
    gana_i      = puja_i > m_i                      (m_i = precio de mercado = 2.º mejor postor)
    utilidad_i  = gana_i · (click_i · V − m_i)

    valor = Σ_i utilidad_i  −  costo de cómputo

En términos de la matriz de confusión (la decisión es comprar o no comprar):
    VP (compra y hay clic)    → + (V − m_i)
    FP (compra y no hay clic) → − m_i      (dinero gastado en una impresión que no convierte)
    FN (no compra, había clic y m_i < V) → oportunidad perdida (V − m_i)
    VN (no compra, no hay clic)          → 0

En una subasta de segundo precio (Vickrey), si el pCTR está calibrado y α = 1, pujar el valor
esperado es la estrategia óptima. Por eso importan DOS métricas técnicas: el AUC
(ordenamiento, para comprar las impresiones buenas) y el log loss / calibración (para pujar el
monto correcto).

Precio de mercado. Avazu no trae precios, así que los simulamos. Suponemos que los rivales
están INFORMADOS: conocen el CTR histórico de cada publisher (sitio o app), que es información
pública en el mercado de RTB. El precio sigue una log-normal centrada en lo que vale la
impresión para ellos:

    m_i = k · V · CTR_publisher_i · exp(σ · z_i),     z_i ~ N(0, 1)

k > 1 representa un mercado competitivo, en el que los rivales valoran el clic un poco más que
nosotros. Bajo este supuesto, un pujador sin información (CTR constante) queda en break-even
y solo gana dinero quien predice MEJOR que el promedio del publisher. Todos los modelos se
evalúan contra LOS MISMOS precios simulados (misma semilla). Ver docs/modelo_de_valor.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import PROCESSED_DIR, SEED, VALUE_PARAMS

SITE_COL, APP_COL = 2, 5   # posiciones de site_id y app_id en config.FIELDS
SMOOTHING = 100            # suavizado bayesiano del CTR histórico del publisher


@dataclass
class ValueParams:
    value_per_click: float = VALUE_PARAMS["value_per_click"]
    market_k: float = VALUE_PARAMS["market_k"]
    market_sigma: float = VALUE_PARAMS["market_sigma"]
    monthly_requests: float = VALUE_PARAMS["monthly_requests"]
    bid_multiplier: float = VALUE_PARAMS["bid_multiplier"]
    infra_usd_per_million: float = VALUE_PARAMS["infra_usd_per_million"]

    def replace(self, **kw) -> "ValueParams":
        return ValueParams(**{**self.__dict__, **kw})

    def scaled(self, **mult) -> "ValueParams":
        """Copia con parámetros multiplicados (para el análisis de sensibilidad)."""
        return self.replace(**{k: getattr(self, k) * m for k, m in mult.items()})


def publisher_ctr(split: str) -> np.ndarray:
    """CTR histórico (train) del publisher (site_id × app_id) de cada impresión de `split`.

    Es también la "regla de negocio" de referencia sin deep learning: pujar según el CTR
    promedio del sitio o app. Se cachea en data/processed/pubctr_<split>.npy.
    """
    cache = PROCESSED_DIR / f"pubctr_{split}.npy"
    if cache.exists():
        return np.load(cache)
    Xtr = np.load(PROCESSED_DIR / "train_X.npy", mmap_mode="r")
    ytr = np.load(PROCESSED_DIR / "train_y.npy")
    key_tr = np.asarray(Xtr[:, SITE_COL], dtype=np.int64) * 2_000_000 + np.asarray(Xtr[:, APP_COL])
    uniq, inv = np.unique(key_tr, return_inverse=True)
    clicks, counts = np.bincount(inv, weights=ytr), np.bincount(inv)
    prior = ytr.mean()
    ctr = (clicks + SMOOTHING * prior) / (counts + SMOOTHING)
    X = np.load(PROCESSED_DIR / f"{split}_X.npy", mmap_mode="r")
    key = np.asarray(X[:, SITE_COL], dtype=np.int64) * 2_000_000 + np.asarray(X[:, APP_COL])
    pos = np.clip(np.searchsorted(uniq, key), 0, len(uniq) - 1)
    out = np.where(uniq[pos] == key, ctr[pos], prior).astype(np.float32)
    np.save(cache, out)
    return out


def market_prices(ref_ctr: np.ndarray, p: ValueParams, seed: int = SEED,
                  independent: bool = False) -> np.ndarray:
    """Precio por impresión (USD) del segundo mejor postor.

    independent=True: escenario alternativo en el que el precio NO depende de la calidad de la
    impresión (mediana = k · V · CTR global). Lo usamos solo como prueba de robustez.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(len(ref_ctr))
    center = np.full(len(ref_ctr), ref_ctr.mean()) if independent else ref_ctr
    return p.market_k * p.value_per_click * center * np.exp(p.market_sigma * z)


def simulate(y: np.ndarray, pctr: np.ndarray, prices: np.ndarray, p: ValueParams) -> dict:
    """Resultado de la campaña sobre un conjunto de impresiones (p. ej. el día de prueba)."""
    bid = p.bid_multiplier * pctr * p.value_per_click
    win = bid > prices
    clicks = y[win]
    spend = prices[win].sum()
    revenue = clicks.sum() * p.value_per_click
    profit = revenue - spend
    n = len(y)
    scale = p.monthly_requests / n
    infra = p.infra_usd_per_million * p.monthly_requests / 1e6
    return {
        "impresiones_compradas": int(win.sum()),
        "win_rate": float(win.mean()),
        "clics": int(clicks.sum()),
        "ctr_comprado": float(clicks.mean()) if win.any() else 0.0,
        "gasto": float(spend),
        "ingreso": float(revenue),
        "utilidad": float(profit),
        "roi": float(profit / spend) if spend > 0 else 0.0,
        "cpc_efectivo": float(spend / max(clicks.sum(), 1)),
        "VP": int((win & (y == 1)).sum()),
        "FP": int((win & (y == 0)).sum()),
        # FN económico: no se compró pero era rentable (hubo clic y el precio era menor a V)
        "FN_rentable": int((~win & (y == 1) & (prices < p.value_per_click)).sum()),
        "utilidad_mensual": float(profit * scale - infra),
        "gasto_mensual": float(spend * scale),
        "clics_mensuales": float(clicks.sum() * scale),
        "infra_mensual": float(infra),
    }


def bid_multiplier_curve(y, pctr, prices, p: ValueParams, alphas=None) -> list[dict]:
    """Curva de valor según el punto de operación (multiplicador de puja ≈ umbral de decisión)."""
    alphas = np.round(np.linspace(0.2, 2.0, 37), 3) if alphas is None else alphas
    out = []
    for a in alphas:
        r = simulate(y, pctr, prices, p.replace(bid_multiplier=float(a)))
        r["alpha"] = float(a)
        out.append(r)
    return out


def best_alpha(y, pctr, prices, p: ValueParams, alphas=None) -> float:
    cv = bid_multiplier_curve(y, pctr, prices, p, alphas)
    return max(cv, key=lambda r: r["utilidad_mensual"])["alpha"]


def threshold_curve(y, pctr, prices, p: ValueParams, thresholds=None) -> list[dict]:
    """Variante de 'umbral fijo': comprar al precio de mercado si pCTR ≥ t (decisión binaria)."""
    thresholds = np.round(np.linspace(0.02, 0.6, 59), 3) if thresholds is None else thresholds
    out = []
    for t in thresholds:
        win = pctr >= t
        profit = (y[win].sum() * p.value_per_click) - prices[win].sum()
        out.append({"umbral": float(t), "utilidad_mensual": float(profit * p.monthly_requests / len(y)),
                    "win_rate": float(win.mean())})
    return out


def bootstrap_profit(y, pctr, prices, p: ValueParams, n_boot: int = 200, seed: int = SEED):
    """IC 95 % de la utilidad mensual, remuestreando impresiones del día de prueba."""
    rng = np.random.default_rng(seed)
    win = p.bid_multiplier * pctr * p.value_per_click > prices
    u = np.where(win, y * p.value_per_click - prices, 0.0)
    scale = p.monthly_requests / len(y)
    infra = p.infra_usd_per_million * p.monthly_requests / 1e6
    vals = [u[rng.integers(0, len(u), len(u))].sum() * scale - infra for _ in range(n_boot)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
