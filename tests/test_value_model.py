"""Pruebas del modelo de valor y de las arquitecturas (no requieren el dataset)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import build_model, fm_interaction  # noqa: E402
from value_model import ValueParams, market_prices, simulate  # noqa: E402


def test_fm_interaction_matches_brute_force():
    e = torch.randn(4, 5, 3)
    brute = sum((e[:, i] * e[:, j]).sum(1) for i in range(5) for j in range(i + 1, 5))
    assert torch.allclose(fm_interaction(e), brute, atol=1e-5)


def test_models_forward_shapes():
    x = torch.randint(0, 100, (8, 6))
    for name in ("lr", "widedeep", "deepfm"):
        m = build_model(name, 100, 6, embed_dim=4, mlp=[8])
        m.eval()
        assert m(x).shape == (8,)


def test_perfect_model_beats_constant():
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.17).astype(float)
    p = ValueParams()
    prices = market_prices(np.full(len(y), 0.17), p, seed=1)
    perfect = simulate(y, np.where(y == 1, 0.99, 0.01), prices, p)
    const = simulate(y, np.full(len(y), 0.17), prices, p)
    assert perfect["utilidad"] > const["utilidad"]


def test_no_bid_means_no_spend():
    y = np.array([1.0, 0.0, 1.0])
    r = simulate(y, np.zeros(3), np.full(3, 0.001), ValueParams())
    assert r["gasto"] == 0 and r["utilidad"] == 0


def test_profit_accounting():
    p = ValueParams(value_per_click=1.0, bid_multiplier=1.0)
    y = np.array([1.0, 0.0])
    prices = np.array([0.2, 0.3])
    r = simulate(y, np.array([0.5, 0.5]), prices, p)  # puja 0.5 gana ambas
    assert np.isclose(r["utilidad"], 1.0 - 0.5)
    assert r["VP"] == 1 and r["FP"] == 1
