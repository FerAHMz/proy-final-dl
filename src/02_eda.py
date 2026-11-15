"""Etapa 2 — Análisis exploratorio (EDA) sobre los splits temporales crudos.

Genera figuras en reports/figures/eda_*.png y un resumen en reports/tables/eda.json:
volumen y CTR por día (desbalance), CTR por hora / posición / tipo de dispositivo / conexión,
cardinalidad y cola larga de cada campo, nulos y valores dominantes (posibles sesgos/fugas).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from config import FIG_DIR, LABEL, PROCESSED_DIR, RAW_CAT_FIELDS, TABLES_DIR, TIME_COL  # noqa: E402
from utils import save_json  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#2a6fdb", "#e8833a"


def ctr_by(df: pd.DataFrame, col: str, min_n: int = 1000) -> pd.DataFrame:
    g = df.groupby(col, observed=True)[LABEL].agg(["mean", "size"]).rename(columns={"mean": "ctr", "size": "n"})
    return g[g["n"] >= min_n].sort_values("n", ascending=False)


def main() -> None:
    summary: dict = {}
    frames = {}
    for s in ("train", "val", "test"):
        frames[s] = pq.read_table(PROCESSED_DIR / f"raw_{s}.parquet",
                                  columns=[LABEL, TIME_COL, "banner_pos", "device_type",
                                           "device_conn_type", "site_category", "app_id"]).to_pandas()
        frames[s][LABEL] = frames[s][LABEL].astype(np.int8)
    df = pd.concat(frames.values(), ignore_index=True)
    df["day"] = df[TIME_COL].str[4:6].astype(int)
    df["hour_of_day"] = df[TIME_COL].str[6:8].astype(int)
    df["is_app"] = df["app_id"] != "ecad2386"  # app_id por defecto = tráfico web (sitio)

    # 1) volumen y CTR por día ------------------------------------------------------------
    by_day = df.groupby("day")[LABEL].agg(["size", "mean"])
    summary["filas_total"] = int(len(df))
    summary["ctr_global"] = float(df[LABEL].mean())
    summary["por_dia"] = {int(d): {"filas": int(r["size"]), "ctr": float(r["mean"])} for d, r in by_day.iterrows()}
    summary["splits"] = {s: {"filas": int(len(f)), "ctr": float(f[LABEL].mean())} for s, f in frames.items()}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = [BLUE if d <= 28 else (ORANGE if d == 29 else "#c0392b") for d in by_day.index]
    ax.bar(by_day.index, by_day["size"] / 1e6, color=colors)
    ax.set_ylabel("millones de impresiones")
    ax.set_xlabel("día de octubre 2014 (azul=train, naranja=val, rojo=test)")
    ax2 = ax.twinx()
    ax2.plot(by_day.index, by_day["mean"], "k.-")
    ax2.set_ylabel("CTR")
    ax.set_title("Volumen y CTR por día — división temporal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_volumen_ctr_dia.png")
    plt.close(fig)

    # 2) CTR por hora del día ---------------------------------------------------------------
    by_h = df.groupby("hour_of_day")[LABEL].agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(by_h.index, by_h["size"] / 1e6, color=BLUE, alpha=0.6)
    ax.set_ylabel("millones")
    ax.set_xlabel("hora del día (UTC del dataset)")
    ax2 = ax.twinx()
    ax2.plot(by_h.index, by_h["mean"], color=ORANGE, marker="o")
    ax2.set_ylabel("CTR", color=ORANGE)
    ax.set_title("Patrón horario: volumen y CTR")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_ctr_hora.png")
    plt.close(fig)

    # 3) CTR por variables de contexto ------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    summary["ctr_por_grupo"] = {}
    for ax, col in zip(axes, ["banner_pos", "device_type", "device_conn_type", "is_app"]):
        g = ctr_by(df, col)
        summary["ctr_por_grupo"][col] = {str(k): {"ctr": float(r.ctr), "n": int(r.n)} for k, r in g.iterrows()}
        ax.bar(g.index.astype(str), g["ctr"], color=BLUE)
        ax.axhline(df[LABEL].mean(), color="k", ls="--", lw=0.8)
        ax.set_title(col)
        ax.set_ylabel("CTR")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("CTR por grupo (línea = CTR global)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_ctr_grupos.png")
    plt.close(fig)

    # 4) cardinalidad, cola larga, nulos y valores dominantes (sobre train) -----------------
    pf = pq.ParquetFile(PROCESSED_DIR / "raw_train.parquet")
    card = {}
    for f in RAW_CAT_FIELDS:
        col = pf.read(columns=[f])[f].combine_chunks()
        vc = pc.value_counts(col)
        cnt = np.sort(vc.field("counts").to_numpy())[::-1]
        top_val = vc.field("values")[int(np.argmax(vc.field("counts").to_numpy()))].as_py()
        n = cnt.sum()
        card[f] = {
            "unicos": int(len(cnt)),
            "nulos": int(col.null_count),
            "top_valor": top_val,
            "top_frac": float(cnt[0] / n),
            "cobertura_top100": float(cnt[:100].sum() / n),
            "frac_categorias_con_menos_de_5": float((cnt < 5).mean()),
        }
    summary["campos"] = card
    fig, ax = plt.subplots(figsize=(9, 3.6))
    names = list(card)
    ax.bar(names, [card[n]["unicos"] for n in names], color=BLUE)
    ax.set_yscale("log")
    ax.set_ylabel("categorías únicas (log)")
    ax.tick_params(axis="x", rotation=60)
    ax.set_title("Cardinalidad por campo (train) — por qué hacen falta embeddings")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_cardinalidad.png")
    plt.close(fig)

    save_json(summary, TABLES_DIR / "eda.json")
    print(f"filas={summary['filas_total']:,} CTR={summary['ctr_global']:.4f}")
    for s, v in summary["splits"].items():
        print(f"  {s}: {v['filas']:,} filas CTR={v['ctr']:.4f}")
    for f, v in card.items():
        print(f"  {f:17s} únicos={v['unicos']:>9,} top={v['top_frac']:.1%} nulos={v['nulos']}")


if __name__ == "__main__":
    main()
