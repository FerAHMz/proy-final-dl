"""Etapa 6 — Modelo de valor: curva de valor, punto de operación, sensibilidad y comparación en USD.

1. Utilidad mensual de cada iteración en el día de prueba (mismos precios de mercado para
   todas), incluidas dos referencias sin deep learning: pujar un CTR constante y la regla
   "CTR histórico del publisher".
2. Punto de operación (multiplicador de puja α): se elige en VALIDACIÓN (día 9) y se reporta
   en PRUEBA (día 10).
3. Curva valor ↔ métrica técnica: utilidad vs AUC y vs log loss, con las predicciones por época
   y versiones degradadas del modelo final (ruido en el logit) → "USD/mes por cada +0.01 de AUC".
4. Sensibilidad: cada supuesto ×0.5, ×0.8, ×1.2 y ×1.5, más un escenario con precios
   independientes de la calidad de la impresión.
5. Costo de cómputo (reentrenamiento diario + inferencia) restado al valor.

Salida: reports/tables/valor.json y reports/figures/valor_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from config import EXPERIMENTS, FIG_DIR, FINAL_EXPERIMENT, MODELS_DIR, SEED, TABLES_DIR  # noqa: E402
from utils import load_meta, load_split, save_json  # noqa: E402
from value_model import (ValueParams, best_alpha, bid_multiplier_curve,  # noqa: E402
                         bootstrap_profit, market_prices, publisher_ctr, simulate,
                         threshold_curve)

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
LABELS = {"it0_base": "CTR constante", "it0_regla": "Regla: CTR del publisher",
          "it1_lr": "Regresión logística", "it2_widedeep": "Wide & Deep", "it3_deepfm": "DeepFM",
          "it4_deepfm_grande": "DeepFM grande", "it5_deepfm_reg": "DeepFM regularizado"}

# Costo de cómputo (precios on-demand de AWS us-east-1; ver docs/modelo_de_valor.md)
GPU_USD_PER_HOUR = 1.006   # g5.xlarge (NVIDIA A10G) para reentrenar
REENTRENOS_POR_MES = 30    # reentrenamiento diario con una ventana móvil de 8 días
ALPHAS = np.round(np.linspace(0.2, 2.0, 37), 3)


def load_preds(split: str) -> dict[str, np.ndarray]:
    meta = load_meta()
    n = meta[f"{split}_rows"]
    preds = {"it0_base": np.full(n, meta["train_ctr"], dtype=np.float32),
             "it0_regla": publisher_ctr(split)}
    for exp in EXPERIMENTS:
        f = MODELS_DIR / exp / f"{split}_pred.npy"
        if f.exists():
            preds[exp] = np.load(f)
    return preds


def infra_cost_per_million(exp: str) -> float:
    bench = TABLES_DIR / "benchmark.json"
    if bench.exists():
        b = json.loads(bench.read_text())
        if exp in b.get("modelos", {}):
            return b["modelos"][exp]["usd_por_millon"]
    return 0.0


def main() -> None:
    _, yva = load_split("val")
    _, yte = load_split("test")
    yva, yte = yva.astype(np.float64), yte.astype(np.float64)
    base = ValueParams()
    ref_va, ref_te = publisher_ctr("val"), publisher_ctr("test")
    pr_va = market_prices(ref_va, base, seed=SEED + 1)
    pr_te = market_prices(ref_te, base, seed=SEED)
    P_va, P_te = load_preds("val"), load_preds("test")
    hist = {e: json.loads((MODELS_DIR / e / "history.json").read_text())
            for e in EXPERIMENTS if (MODELS_DIR / e / "history.json").exists()}
    res: dict = {"supuestos": base.__dict__ | {"gpu_usd_hora": GPU_USD_PER_HOUR,
                                                "reentrenos_mes": REENTRENOS_POR_MES},
                 "precio_mercado_test": {"mediana_cpm": float(np.median(pr_te) * 1000),
                                         "media_cpm": float(pr_te.mean() * 1000),
                                         "p10_cpm": float(np.quantile(pr_te, 0.1) * 1000),
                                         "p90_cpm": float(np.quantile(pr_te, 0.9) * 1000)}}

    # 1-2) comparación por iteración, α elegido en validación --------------------------------
    comp, curves = {}, {}
    for exp, p_te in P_te.items():
        params = base.replace(infra_usd_per_million=infra_cost_per_million(exp))
        a_opt = best_alpha(yva, P_va[exp], pr_va, params, ALPHAS)
        curves[exp] = bid_multiplier_curve(yte, p_te, pr_te, params, ALPHAS)
        r1 = simulate(yte, p_te, pr_te, params)
        p_opt = params.replace(bid_multiplier=a_opt)
        ropt = simulate(yte, p_te, pr_te, p_opt)
        train_h = hist[exp]["train_seconds"] / 3600 if exp in hist else 0.0
        comp[exp] = {
            "auc": float(roc_auc_score(yte, p_te)) if exp != "it0_base" else 0.5,
            "logloss": float(log_loss(yte, np.clip(p_te, 1e-7, 1 - 1e-7))),
            "alpha_1": r1, "alpha_opt_val": a_opt, "opt": ropt,
            # criterio de selección del modelo final: utilidad en VALIDACIÓN (no en prueba)
            "utilidad_mensual_val": simulate(yva, P_va[exp], pr_va, params.replace(bid_multiplier=a_opt))["utilidad_mensual"],
            "ic95_utilidad_mensual": bootstrap_profit(yte, p_te, pr_te, p_opt, n_boot=100),
            "horas_entrenamiento": train_h,
            "costo_entrenamiento_mensual": train_h * GPU_USD_PER_HOUR * REENTRENOS_POR_MES,
            "costo_inferencia_mensual": ropt["infra_mensual"],
        }
        comp[exp]["valor_neto_mensual"] = ropt["utilidad_mensual"] - comp[exp]["costo_entrenamiento_mensual"]
        ic = comp[exp]["ic95_utilidad_mensual"]
        print(f"{exp:18s} AUC={comp[exp]['auc']:.4f} ll={comp[exp]['logloss']:.4f} α*={a_opt:.2f} "
              f"utilidad/mes=${ropt['utilidad_mensual']:,.0f} [{ic[0]:,.0f}; {ic[1]:,.0f}] "
              f"(α=1: ${r1['utilidad_mensual']:,.0f}) ROI={ropt['roi']:.1%} win={ropt['win_rate']:.1%}")
    ref = "it1_lr" if "it1_lr" in comp else "it0_regla"
    for exp in comp:
        comp[exp]["delta_vs_lr_mensual"] = comp[exp]["valor_neto_mensual"] - comp[ref]["valor_neto_mensual"]
        comp[exp]["delta_vs_regla_mensual"] = comp[exp]["valor_neto_mensual"] - comp["it0_regla"]["valor_neto_mensual"]
    res["comparacion"] = comp

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for exp, cv in curves.items():
        ax.plot([r["alpha"] for r in cv], [r["utilidad_mensual"] / 1e3 for r in cv], label=LABELS[exp])
        ax.plot(comp[exp]["alpha_opt_val"], comp[exp]["opt"]["utilidad_mensual"] / 1e3, "k|", ms=10)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(1, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("multiplicador de puja α  (puja = α · pCTR · V)")
    ax.set_ylabel("utilidad mensual (miles USD)")
    ax.set_title("Curva de valor por punto de operación (| = α elegido en validación)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "valor_vs_alpha.png")
    plt.close(fig)

    fin = FINAL_EXPERIMENT if FINAL_EXPERIMENT in P_te else list(P_te)[-1]
    res["curva_umbral_final"] = threshold_curve(yte, P_te[fin], pr_te, base)

    # 3) curva valor ↔ métrica técnica ------------------------------------------------------
    pts = []
    rng = np.random.default_rng(SEED)
    pf = np.clip(P_te[fin].astype(np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(pf / (1 - pf))
    a_fin = comp[fin]["alpha_opt_val"]
    for s in [0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]:
        noisy = 1 / (1 + np.exp(-(logit + rng.normal(0, s, len(logit)))))
        # reescalar la media para aislar el efecto del ORDENAMIENTO (AUC)
        noisy = np.clip(noisy * (pf.mean() / noisy.mean()), 1e-6, 1 - 1e-6)
        r = simulate(yte, noisy, pr_te, base.replace(bid_multiplier=a_fin))
        pts.append({"fuente": f"{fin}+ruido{s}", "ruido": s, "auc": float(roc_auc_score(yte, noisy)),
                    "logloss": float(log_loss(yte, noisy)), "utilidad_mensual": r["utilidad_mensual"]})
    for exp in EXPERIMENTS:
        for f in sorted((MODELS_DIR / exp).glob("test_pred_e*.npy")):
            p = np.load(f).astype(np.float64)
            r = simulate(yte, p, pr_te, base.replace(bid_multiplier=comp[exp]["alpha_opt_val"]))
            pts.append({"fuente": f"{exp}/{f.stem}", "auc": float(roc_auc_score(yte, p)),
                        "logloss": float(log_loss(yte, np.clip(p, 1e-7, 1 - 1e-7))),
                        "utilidad_mensual": r["utilidad_mensual"]})
    noise_pts = [q for q in pts if "ruido" in q["fuente"]]
    a = np.array([q["auc"] for q in noise_pts])
    u = np.array([q["utilidad_mensual"] for q in noise_pts])
    zone = a >= a.max() - 0.04  # pendiente local: zona de AUC alcanzable por modelos reales
    slope = float(np.polyfit(a[zone], u[zone], 1)[0])
    res["curva_valor_metrica"] = pts
    res["usd_mes_por_0.01_auc"] = slope * 0.01
    print(f"Pendiente local: ${slope * 0.01:,.0f} al mes por cada +0.01 de AUC")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    for ax, key, lab in ((axes[0], "auc", "AUC (prueba)"), (axes[1], "logloss", "log loss (prueba)")):
        ax.plot([q[key] for q in noise_pts], [q["utilidad_mensual"] / 1e3 for q in noise_pts], "o-",
                color="#999", ms=3, label="Modelo final degradado con ruido")
        for exp in EXPERIMENTS:
            ep = [q for q in pts if q["fuente"].startswith(exp + "/")]
            if ep:
                ax.plot([q[key] for q in ep], [q["utilidad_mensual"] / 1e3 for q in ep], "o", ms=6,
                        label=LABELS[exp] + " (cada época)")
        for e0 in ("it0_regla",):
            ax.plot(comp[e0][key], comp[e0]["opt"]["utilidad_mensual"] / 1e3, "s", color="k", ms=6,
                    label=LABELS[e0])
        ax.set_xlabel(lab)
        ax.set_ylabel("utilidad mensual (miles USD)")
        ax.axhline(0, color="k", lw=0.5)
    axes[0].set_title(f"Valor vs AUC: ≈ ${slope * 0.01:,.0f}/mes por +0.01 de AUC")
    axes[1].set_title("Valor vs log loss (más a la derecha = mejor)")
    axes[1].invert_xaxis()
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "valor_vs_metrica.png")
    plt.close(fig)

    # 4) sensibilidad -----------------------------------------------------------------------
    sens = []
    for par in ["value_per_click", "market_k", "market_sigma", "monthly_requests"]:
        for m in [0.5, 0.8, 1.2, 1.5]:
            q = base.scaled(**{par: m})
            pv, pt = market_prices(ref_va, q, seed=SEED + 1), market_prices(ref_te, q, seed=SEED)
            row = {"parametro": par, "factor": m}
            for exp in (fin, ref, "it0_regla"):
                a_opt = best_alpha(yva, P_va[exp], pv, q, ALPHAS)  # el punto de operación se recalcula
                row[exp] = simulate(yte, P_te[exp], pt, q.replace(bid_multiplier=a_opt))["utilidad_mensual"]
            row["delta"] = row[fin] - row[ref]
            sens.append(row)
            print(f"  sens {par:17s} ×{m:.1f}: final=${row[fin]:,.0f} LR=${row[ref]:,.0f} "
                  f"regla=${row['it0_regla']:,.0f} Δ(final-LR)=${row['delta']:,.0f}")
    pv = market_prices(ref_va, base, seed=SEED + 1, independent=True)
    pt = market_prices(ref_te, base, seed=SEED, independent=True)
    row = {"parametro": "precios_independientes", "factor": 1.0}
    for exp in (fin, ref, "it0_regla", "it0_base"):
        a_opt = best_alpha(yva, P_va[exp], pv, base, ALPHAS)
        row[exp] = simulate(yte, P_te[exp], pt, base.replace(bid_multiplier=a_opt))["utilidad_mensual"]
    row["delta"] = row[fin] - row[ref]
    sens.append(row)
    print(f"  precios independientes: final=${row[fin]:,.0f} LR=${row[ref]:,.0f} Δ=${row['delta']:,.0f}")
    res["sensibilidad"] = sens
    res["final"], res["referencia"] = fin, ref

    fig, ax = plt.subplots(figsize=(8, 3.6))
    base_delta = comp[fin]["opt"]["utilidad_mensual"] - comp[ref]["opt"]["utilidad_mensual"]
    names = ["value_per_click", "market_k", "market_sigma", "monthly_requests"]
    nice = {"value_per_click": "valor por clic V", "market_k": "competitividad k",
            "market_sigma": "dispersión σ de precios", "monthly_requests": "volumen mensual"}
    for i, par in enumerate(names):
        rows = {r["factor"]: r["delta"] for r in sens if r["parametro"] == par}
        for f_, col, al in ((0.5, "#c0392b", 0.35), (1.5, "#2a6fdb", 0.35), (0.8, "#c0392b", 1), (1.2, "#2a6fdb", 1)):
            ax.barh(i, (rows[f_] - base_delta) / 1e3, left=base_delta / 1e3, color=col, alpha=al)
    ax.set_yticks(range(len(names)), [nice[n] for n in names])
    ax.axvline(base_delta / 1e3, color="k", lw=0.8)
    ax.set_xlabel("ventaja mensual del DeepFM sobre LR (miles USD)")
    ax.set_title("Sensibilidad: rojo = supuesto ×0.8 (×0.5 claro) · azul = ×1.2 (×1.5 claro)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "valor_sensibilidad.png")
    plt.close(fig)

    save_json(res, TABLES_DIR / "valor.json")


if __name__ == "__main__":
    main()
