"""Laboratorio 10 — Explainable AI sobre el modelo final del proyecto (it5_deepfm_reg).

Regenera TODAS las figuras y tablas del PDF del laboratorio con semilla fija:

    python xai/lab10_xai.py            # ≈ 3-5 min en Apple M4 Pro (MPS) / ~10 min en CPU

Método principal: Integrated Gradients (IG) a nivel de CAMPO.
    - DeepFM recibe índices categóricos; no se puede derivar respecto a un índice, así que IG
      se aplica sobre la representación continua de cada campo: su embedding v_f y su peso
      lineal w_f. La atribución de un campo es la suma de IG sobre sus componentes.
    - Línea base: la categoría OOV ("desconocido") de cada campo, es decir, la impresión de la
      que el modelo no sabe nada. Así, la atribución de un campo responde: "¿cuánto cambia el
      logit de clic por conocer el valor real de este campo en lugar de ignorarlo?".
    - Completitud: Σ_f IG_f = logit(x) − logit(base). Se verifica numéricamente.

Alternativa (usada como control): KernelSHAP con la MISMA línea base (campos "ausentes" → OOV),
para estimar valores de Shapley del juego v(S) = logit(x_S, OOV_resto). Medimos la concordancia
con IG y el costo de cada método.

Salida: reports/xai/*.png y reports/xai/*.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import EXPERIMENTS, FIELDS, FINAL_EXPERIMENT, MODELS_DIR, PROCESSED_DIR, SEED, TABLES_DIR, set_seed  # noqa: E402
from models import build_model, fm_interaction  # noqa: E402
from utils import load_meta, load_split, save_json  # noqa: E402
from value_model import ValueParams, market_prices, publisher_ctr, simulate  # noqa: E402

OUT = ROOT / "reports" / "xai"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 8})

# ----------------------------------------------------------------- parámetros de diseño
IG_STEPS = 64            # pasos de la integral (regla del punto medio); error de completitud < 1e-3
N_GLOBAL = 5_000         # muestra del día de prueba para la explicación global
KSHAP_NSAMPLES = 2_048   # coaliciones por instancia en KernelSHAP (2^23 ≈ 8.4M posibles)
N_KSHAP = 30             # instancias en las que se compara KernelSHAP contra IG
TOP_K = 8                # campos mostrados por explicación local
NICE = {"hour_of_day": "hora", "weekday": "día_sem", "device_conn_type": "conexión",
        "device_type": "tipo_disp", "banner_pos": "pos_banner"}


def nice(f: str) -> str:
    return NICE.get(f, f)


# ------------------------------------------------------------------------------- modelo
class FieldModel:
    """Envuelve el DeepFM para evaluarlo desde (pesos lineales, embeddings) por campo."""

    def __init__(self, exp: str, device):
        meta = load_meta()
        cfg = EXPERIMENTS[exp]
        self.m = build_model(cfg["model"], meta["num_features"], len(FIELDS), **cfg).to(device)
        self.m.load_state_dict(torch.load(MODELS_DIR / exp / "model.pt", map_location=device))
        self.m.eval()
        self.device = device
        self.offsets = np.array([meta["offsets"][f] for f in FIELDS])
        self.base_idx = torch.from_numpy(self.offsets).long().to(device)  # índice OOV de cada campo

    def parts(self, idx: torch.Tensor):
        return self.m.linear.w(idx).squeeze(-1), self.m.embedding(idx)  # (B,F), (B,F,D)

    def logit_from(self, lin: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        B = emb.shape[0]
        return (lin.sum(1) + self.m.linear.bias + fm_interaction(emb)
                + self.m.dnn(emb.reshape(B, -1)).squeeze(1))

    @torch.no_grad()
    def logit(self, X: np.ndarray, bs: int = 65536) -> np.ndarray:
        out = []
        for s in range(0, len(X), bs):
            out.append(self.m(torch.from_numpy(X[s:s + bs].astype(np.int64)).to(self.device)).cpu().numpy())
        return np.concatenate(out)

    def integrated_gradients(self, X: np.ndarray, steps: int = IG_STEPS, chunk: int = 64):
        """Devuelve (atribuciones (N,F), logit(x), logit(base))."""
        attrs, fx = [], []
        alphas = ((torch.arange(steps, device=self.device) + 0.5) / steps).view(-1, 1, 1)
        with torch.no_grad():
            bl, be = self.parts(self.base_idx.unsqueeze(0))
            f_base = float(self.logit_from(bl, be))
        for s in range(0, len(X), chunk):
            idx = torch.from_numpy(X[s:s + chunk].astype(np.int64)).to(self.device)
            with torch.no_grad():
                xl, xe = self.parts(idx)                       # (b,F), (b,F,D)
            b = len(idx)
            dl, de = xl - bl, xe - be
            # caminos: (steps, b, ...) → aplanados para un solo forward/backward
            pl = (bl.unsqueeze(0) + alphas * dl.unsqueeze(0)).reshape(steps * b, -1)
            pe = (be.unsqueeze(0) + alphas.unsqueeze(-1) * de.unsqueeze(0)).reshape(steps * b, *xe.shape[1:])
            pl.requires_grad_(True)
            pe.requires_grad_(True)
            out = self.logit_from(pl, pe)
            gl, ge = torch.autograd.grad(out.sum(), [pl, pe])
            gl = gl.view(steps, b, -1).mean(0)
            ge = ge.view(steps, b, *xe.shape[1:]).mean(0)
            a = dl * gl + (de * ge).sum(-1)                    # (b, F)
            attrs.append(a.detach().cpu().numpy())
            with torch.no_grad():
                fx.append(self.logit_from(xl, xe).cpu().numpy())
        return np.concatenate(attrs), np.concatenate(fx), f_base


def kernel_shap(fm: FieldModel, x: np.ndarray, nsamples: int, seed: int):
    """KernelSHAP a nivel de campo: un campo 'ausente' toma su categoría OOV (misma base que IG)."""
    import shap

    def f(Z):
        Xz = np.where(Z.astype(bool), x[None, :], fm.offsets[None, :])
        return fm.logit(Xz)

    np.random.seed(seed)
    expl = shap.KernelExplainer(f, np.zeros((1, len(FIELDS))), silent=True)
    return expl.shap_values(np.ones((1, len(FIELDS))), nsamples=nsamples, silent=True)[0]


# --------------------------------------------------------------------------------- main
def main(exp: str = FINAL_EXPERIMENT) -> None:
    set_seed(SEED)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    fm = FieldModel(exp, device)
    meta = load_meta()
    vocab = json.loads((PROCESSED_DIR / "vocab.json").read_text())
    Xte, yte = load_split("test")
    p_te = np.load(MODELS_DIR / exp / "test_pred.npy")
    valor = json.loads((TABLES_DIR / "valor.json").read_text())
    alpha = valor["comparacion"][exp]["alpha_opt_val"]
    vp = ValueParams()
    prices = market_prices(publisher_ctr("test"), vp, seed=SEED)
    bid = alpha * p_te * vp.value_per_click
    buy = bid > prices
    oov = Xte == fm.offsets
    n_oov = oov.sum(1)
    res: dict = {"modelo": exp, "parametros": {"ig_pasos": IG_STEPS, "linea_base": "OOV por campo",
                 "n_global": N_GLOBAL, "kshap_nsamples": KSHAP_NSAMPLES, "n_kshap": N_KSHAP,
                 "semilla": SEED, "alpha": alpha}}
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------- 1) selección de instancias
    def pick(mask, key=None, largest=True):
        cand = np.flatnonzero(mask)
        if key is None:
            return int(rng.choice(cand))
        order = np.argsort(key[cand])
        return int(cand[order[-1] if largest else order[0]])

    conn_col = FIELDS.index("device_conn_type")
    conn3 = fm.offsets[conn_col] + 1 + vocab["device_conn_type"].index("3")
    rel_gap = np.abs(bid - prices) / prices
    cases = {
        "acierto_confiado": pick((yte == 1) & buy & (p_te > np.quantile(p_te, 0.995))),
        "error_FP_confiado": pick((yte == 0) & buy & (n_oov <= 1), key=p_te, largest=True),
        "error_FN": pick((yte == 1) & ~buy & (prices < vp.value_per_click) & (n_oov <= 1), key=p_te, largest=False),
        "cerca_umbral": pick((rel_gap < 0.005) & (yte == 1)),
        "subgrupo_conexion3": pick((Xte[:, conn_col] == conn3) & (yte == 1) & (np.abs(p_te - 0.15) < 0.03)),
        "muchos_OOV": pick((n_oov >= 5)),
    }
    idx_local = np.array(list(cases.values()))
    A_loc, f_loc, f_base = fm.integrated_gradients(Xte[idx_local])
    res["logit_base"] = f_base
    res["p_base"] = float(1 / (1 + np.exp(-f_base)))

    def decode(i, j):
        local = Xte[i, j] - fm.offsets[j]
        return "OOV" if local == 0 else vocab[FIELDS[j]][local - 1]

    locales = {}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.2))
    for ax, (name, i), a, fx in zip(axes.ravel(), cases.items(), A_loc, f_loc):
        top = np.argsort(-np.abs(a))[:TOP_K][::-1]
        cols = ["#c0392b" if a[j] > 0 else "#2a6fdb" for j in top]
        ax.barh([f"{nice(FIELDS[j])}={decode(i, j)[:8]}" for j in top], a[top], color=cols)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"{name}\np={p_te[i]:.2f} · clic={int(yte[i])} · {'compra' if buy[i] else 'no compra'}",
                     fontsize=8)
        ax.tick_params(axis="y", labelsize=6.5)
        locales[name] = {
            "fila_test": int(i), "p": float(p_te[i]), "click": int(yte[i]), "compra": bool(buy[i]),
            "puja_cpm": float(bid[i] * 1000), "precio_cpm": float(prices[i] * 1000), "n_oov": int(n_oov[i]),
            "logit": float(fx), "suma_atribuciones": float(a.sum()),
            "error_completitud": float(abs(a.sum() - (fx - f_base))),
            "top": [{"campo": FIELDS[j], "valor": decode(i, j), "atribucion": float(a[j])}
                    for j in np.argsort(-np.abs(a))[:TOP_K]],
        }
    fig.suptitle(f"IG por campo (logit; rojo empuja hacia clic, azul en contra). Base OOV: p={res['p_base']:.2f}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "xai_locales.png")
    plt.close(fig)
    res["locales"] = locales
    for k, v in locales.items():
        print(f"{k:20s} p={v['p']:.3f} clic={v['click']} compra={v['compra']} err_compl={v['error_completitud']:.1e} "
              f"top={[(t['campo'], round(t['atribucion'], 2)) for t in v['top'][:3]]}")

    # --------------------------------------------------------------- 2) explicación global
    gidx = rng.choice(len(yte), N_GLOBAL, replace=False)
    t0 = time.perf_counter()
    A, fx, _ = fm.integrated_gradients(Xte[gidx])
    t_ig = (time.perf_counter() - t0) / N_GLOBAL
    compl = np.abs(A.sum(1) - (fx - f_base))
    res["completitud"] = {"error_medio": float(compl.mean()), "error_max": float(compl.max())}
    res["costo"] = {"ig_ms_por_instancia": t_ig * 1000, "dispositivo": str(device)}
    mean_abs = np.abs(A).mean(0)
    order = np.argsort(-mean_abs)
    res["global"] = {FIELDS[j]: {"media_abs": float(mean_abs[j]), "media": float(A[:, j].mean()),
                                 "frac_total_abs": float(mean_abs[j] / mean_abs.sum())} for j in order}
    perm = json.loads((TABLES_DIR / "explicabilidad.json").read_text())["importancia_permutacion"]
    rho = spearmanr([mean_abs[FIELDS.index(f)] for f in FIELDS], [perm[f]["delta_auc"] for f in FIELDS])[0]
    res["spearman_ig_vs_permutacion"] = float(rho)
    print(f"IG global: {t_ig * 1000:.2f} ms/instancia, completitud media {compl.mean():.1e}, "
          f"Spearman vs permutación={rho:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    axes[0].barh([nice(FIELDS[j]) for j in order[::-1]], mean_abs[order[::-1]], color="#2a6fdb")
    axes[0].set_xlabel("media |IG| (logit)")
    axes[0].set_title(f"Importancia global (n={N_GLOBAL:,} de prueba)")
    top12 = order[:12][::-1]
    for r, j in enumerate(top12):
        jit = rng.normal(0, 0.12, N_GLOBAL)
        c = np.where(oov[gidx, j], "#999999", np.where(yte[gidx] == 1, "#c0392b", "#2a6fdb"))
        axes[1].scatter(A[:, j], r + jit, s=1.2, c=c, alpha=0.35, linewidths=0)
    axes[1].set_yticks(range(len(top12)), [nice(FIELDS[j]) for j in top12])
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].set_xlabel("atribución IG (logit) por impresión")
    axes[1].set_title("Distribución (rojo = hubo clic, azul = no, gris = valor OOV)")
    axes[1].set_xlim(np.quantile(A[:, top12], 0.002), np.quantile(A[:, top12], 0.998))
    fig.tight_layout()
    fig.savefig(OUT / "xai_global.png")
    plt.close(fig)

    # atribución por GRUPO de campos correlacionados (la completitud se conserva al sumar)
    grupos_campos = {
        "publisher app (app_id, app_domain, app_category)": ["app_id", "app_domain", "app_category"],
        "publisher sitio (site_id, site_domain, site_category)": ["site_id", "site_domain", "site_category"],
        "usuario (device_id, device_ip)": ["device_id", "device_ip"],
        "dispositivo (modelo, tipo, conexión)": ["device_model", "device_type", "device_conn_type"],
        "anuncio (banner_pos, C1, C14-C21)": ["banner_pos", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"],
        "tiempo (hora, día)": ["hour_of_day", "weekday"],
    }
    res["global_grupos"] = {}
    for g, fs in grupos_campos.items():
        Ag = A[:, [FIELDS.index(f) for f in fs]].sum(1)
        res["global_grupos"][g] = {"media_abs": float(np.abs(Ag).mean()), "media": float(Ag.mean()),
                                   "suma_media_abs_individual": float(mean_abs[[FIELDS.index(f) for f in fs]].sum())}

    # contexto de los errores locales: CTR en train de la categoría que domina la explicación
    Xtr, ytr = load_split("train", mmap=True)
    for name in ("error_FP_confiado", "error_FN", "acierto_confiado"):
        t = locales[name]["top"][0]
        j = FIELDS.index(t["campo"])
        col = np.asarray(Xtr[:, j])
        m = col == Xte[cases[name], j]
        locales[name]["categoria_dominante_train"] = {"campo": t["campo"], "valor": t["valor"],
                                                      "n_train": int(m.sum()),
                                                      "ctr_train": float(np.asarray(ytr)[m].mean()) if m.any() else None}
        print(f"  {name}: {t['campo']}={t['valor']} n_train={int(m.sum()):,} "
              f"ctr_train={locales[name]['categoria_dominante_train']['ctr_train']}")

    # ------------------------------------------- 3) análisis de errores (FP vs VP con p alta)
    pg, yg, bg = p_te[gidx], yte[gidx], buy[gidx]
    hi = pg > np.quantile(pg, 0.8)
    fp, tp = hi & (yg == 0), hi & (yg == 1)
    diff = A[fp].mean(0) - A[tp].mean(0)
    od = np.argsort(-np.abs(diff))[:8]
    res["errores_FP_vs_VP_top20"] = {"n_fp": int(fp.sum()), "n_tp": int(tp.sum()),
                                     "dif_media": {FIELDS[j]: float(diff[j]) for j in od},
                                     "oov_fp": float(n_oov[gidx][fp].mean()), "oov_tp": float(n_oov[gidx][tp].mean())}
    # FN: clics que no se compraron
    fn = (yg == 1) & ~bg
    res["errores_FN"] = {"n": int(fn.sum()), "media_atrib": {FIELDS[j]: float(A[fn, j].mean()) for j in order[:6]},
                         "p_media": float(pg[fn].mean())}

    # -------------------------- 4) atajos / proxies: ablación en inferencia (sin reentrenar)
    lg = fm.logit(Xte)
    base_auc = roc_auc_score(yte, lg)
    base_val = simulate(yte.astype(float), p_te, prices, vp.replace(bid_multiplier=alpha))["utilidad_mensual"]
    abl = {}
    groups = {"identificadores_usuario (device_id, device_ip)": ["device_id", "device_ip"],
              "solo device_ip": ["device_ip"], "tiempo (hora, día)": ["hour_of_day", "weekday"],
              "publisher (site_*, app_*)": ["site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category"],
              "anonimizadas C14-C21": ["C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]}
    for name, fs in groups.items():
        Xa = Xte.copy()
        for f in fs:
            Xa[:, FIELDS.index(f)] = fm.offsets[FIELDS.index(f)]
        la = fm.logit(Xa)
        pa = 1 / (1 + np.exp(-la))
        abl[name] = {"auc": float(roc_auc_score(yte, la)), "delta_auc": float(roc_auc_score(yte, la) - base_auc),
                     "utilidad_mensual": simulate(yte.astype(float), pa, prices, vp.replace(bid_multiplier=alpha))["utilidad_mensual"]}
        abl[name]["delta_utilidad"] = abl[name]["utilidad_mensual"] - base_val
        print(f"  ablación {name:45s} ΔAUC={abl[name]['delta_auc']:+.4f} Δ$={abl[name]['delta_utilidad']:+,.0f}")
    res["ablacion"] = {"auc_base": float(base_auc), "utilidad_base": base_val, "grupos": abl}
    # ¿device_ip actúa como identificador memorizado? atribución según frecuencia en train
    ip = FIELDS.index("device_ip")
    seen = ~oov[gidx, ip]
    res["device_ip_memorizacion"] = {
        "frac_vista_en_train": float(seen.mean()),
        "media_abs_ig_vista": float(np.abs(A[seen, ip]).mean()),
        "media_abs_ig_oov": float(np.abs(A[~seen, ip]).mean()),
        "corr_signo_ig_con_clic": float(np.corrcoef(A[seen, ip], yg[seen])[0, 1]),
    }

    # ------------------------------------------------------- 5) KernelSHAP como alternativa
    kidx = np.concatenate([idx_local, rng.choice(len(yte), N_KSHAP - len(idx_local), replace=False)])
    A_k, _, _ = fm.integrated_gradients(Xte[kidx])
    rhos, t_ks, eff = [], [], []
    for n, i in enumerate(kidx):
        t0 = time.perf_counter()
        sv = kernel_shap(fm, Xte[i], KSHAP_NSAMPLES, SEED + n)
        t_ks.append(time.perf_counter() - t0)
        rhos.append(spearmanr(sv, A_k[n])[0])
        eff.append(abs(sv.sum() - A_k[n].sum()))
        if n == 0:
            sv0 = sv
    res["kernelshap"] = {"spearman_medio_vs_ig": float(np.mean(rhos)), "spearman_min": float(np.min(rhos)),
                         "top3_coinciden_frac": None, "s_por_instancia": float(np.mean(t_ks)),
                         "dif_suma_vs_ig": float(np.mean(eff))}
    print(f"KernelSHAP: {np.mean(t_ks):.2f} s/instancia, Spearman con IG={np.mean(rhos):.2f} (min {np.min(rhos):.2f})")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.scatter(A_k[0], sv0, s=14, color="#2a6fdb")
    lim = [min(A_k[0].min(), sv0.min()), max(A_k[0].max(), sv0.max())]
    ax.plot(lim, lim, "k--", lw=0.6)
    for j in np.argsort(-np.abs(A_k[0]))[:4]:
        ax.annotate(nice(FIELDS[j]), (A_k[0][j], sv0[j]), fontsize=6)
    ax.set_xlabel("IG (logit)")
    ax.set_ylabel("KernelSHAP (logit)")
    ax.set_title(f"IG vs KernelSHAP, caso acierto_confiado\nρ medio en {N_KSHAP} casos = {np.mean(rhos):.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "xai_ig_vs_kernelshap.png")
    plt.close(fig)

    save_json(res, OUT / "lab10_resultados.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else FINAL_EXPERIMENT)
