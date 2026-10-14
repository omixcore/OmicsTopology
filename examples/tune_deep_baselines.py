"""Tune the deep baselines on the positive controls alone, then freeze.

Protocol
--------
``learned_baselines.DEEP_SETTINGS`` must not contain a single value that was
chosen with a negative cloud in view.  This script is the audit trail: it runs
a small grid over four independent synthetic circadian rings -- and nothing
else -- scores every configuration with the same ``circular_succession_score``
the benchmark uses, and writes the whole grid to ``benchmark/deep_tuning.json``.
The winners are copied into ``DEEP_SETTINGS`` by hand, with this file as the
provenance.  Every sensitivity number the deep baselines earn in the benchmark
is therefore in-sample and every specificity number is out-of-sample.

The grid is deliberately small: with 4 rings the point is to give the deep
methods a fair, cheap shot at the one thing they must do -- read a ring -- not
to search a landscape.  Anything that cannot read a ring on all four controls
does not deserve a row in the benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from learned_baselines import deep_autoencoder_latent, _angle_order  # noqa: E402

from omictopo.cycles import circular_succession_score  # noqa: E402
from omictopo.geometry import standardize  # noqa: E402

N_CONTROL_RINGS = 4
GRID = {
    "ae_deep": {"weight_decay": [1e-2, 3e-2], "lr": [6e-3, 2e-2]},
    "dae": {"noise_sigma": [0.1, 0.25, 0.5], "weight_decay": [1e-2, 3e-2]},
    "vae": {"kl_beta": [0.01, 0.03, 0.1], "weight_decay": [1e-2, 3e-2]},
}
EPOCHS = 1200


def control_ring(seed: int) -> tuple:
    """A synthetic circadian ring, the same generator the linear check uses."""
    angles = np.sort(np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, 24))
    pattern = np.random.default_rng(seed + 1).normal(size=(300, 2))
    X = np.c_[np.cos(angles), np.sin(angles)] @ pattern.T
    X = X + np.random.default_rng(seed + 2).normal(0.0, 0.35, X.shape)
    return standardize(X), angles


def score_config(Z: np.ndarray, angles: np.ndarray, **kw) -> float:
    code, _ = deep_autoencoder_latent(Z, latent=2, epochs=EPOCHS, **kw)
    order = _angle_order(code)
    return round(float(circular_succession_score(angles, order)), 4) \
        if order.size else 0.0


def sweep(variant: str, rings: list, **fixed) -> dict:
    rows = []
    axes = GRID[variant]
    names = list(axes)
    for values in _product([axes[k] for k in names]):
        cfg = {k: float(v) for k, v in zip(names, values)}
        scores = [score_config(Z, ang, **cfg, **fixed) for Z, ang in rings]
        rows.append({
            "config": cfg,
            "scores": scores,
            "mean": round(float(np.mean(scores)), 4),
            "worst": round(float(np.min(scores)), 4),
        })
    best = max(rows, key=lambda r: (r["mean"], r["worst"]))
    return {"rows": rows, "best": best["config"],
            "best_mean": best["mean"], "best_worst": best["worst"]}


def _product(axes: list) -> list:
    out = [[]]
    for axis in axes:
        out = [row + [v] for row in out for v in axis]
    return out


def sweep_umap(rings: list) -> dict:
    """UMAP's only real hyperparameter, swept on the same positive controls."""
    try:
        import umap
    except Exception:
        return {"rows": [], "best": None, "best_mean": None,
                "note": "umap-learn not installed"}
    import warnings
    rows = []
    for nn in (5, 8, 12, 15, 23):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = []
            for Z, ang in rings:
                emb = umap.UMAP(n_components=2, n_neighbors=nn,
                                metric="precomputed",
                                random_state=0).fit_transform(
                    _pairwise(Z))
                scores.append(round(float(
                    circular_succession_score(ang, _angle_order(emb))), 4))
        rows.append({"config": {"n_neighbors": nn}, "scores": scores,
                     "mean": round(float(np.mean(scores)), 4),
                     "worst": round(float(np.min(scores)), 4)})
    best = max(rows, key=lambda r: (r["mean"], r["worst"]))
    return {"rows": rows, "best": best["config"],
            "best_mean": best["mean"], "best_worst": best["worst"]}


def _pairwise(Z: np.ndarray) -> np.ndarray:
    from omictopo.geometry import pairwise_distances
    return pairwise_distances(Z)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="benchmark")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rings = [control_ring(s) for s in range(N_CONTROL_RINGS)]
    print(f"{len(rings)} positive control rings; grid: "
          f"{sum(int(np.prod([len(v) for v in a.values()])) for a in GRID.values())} "
          f"configs x {EPOCHS} epochs each")

    results = {
        "protocol": "positive controls only; winners frozen into "
                    "learned_baselines.DEEP_SETTINGS",
        "n_control_rings": N_CONTROL_RINGS,
        "epochs": EPOCHS,
        "layers": [64, 32, 16],
        "grid": {},
    }
    for variant, fixed in (("ae_deep", {"kl_beta": 0.0, "noise_sigma": 0.0}),
                           ("dae", {"kl_beta": 0.0}),
                           ("vae", {"noise_sigma": 0.0})):
        res = sweep(variant, rings, **fixed)
        results["grid"][variant] = res
        print(f"{variant:8s} best {res['best']}  "
              f"mean={res['best_mean']} worst={res['best_worst']}")

    umap_res = sweep_umap(rings)
    results["grid"]["umap"] = umap_res
    if umap_res["best"]:
        print(f"umap     best {umap_res['best']}  "
              f"mean={umap_res['best_mean']} worst={umap_res['best_worst']}")

    out = outdir / "deep_tuning.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
