"""Diagnosis: how much do biological replicates contribute to loop detection?

GSE137732 samples each time point **once** (no biological replicates) and was
the only circadian dataset the benchmark called line-like.  GSE158444 samples
each time point three times and was called loop-like.  This script drops the
replicates of GSE158444 at random (one sample per time point, 60 draws) and
re-runs the identical pipeline, so the only thing that changes is n and the
replicate averaging.  This is a post-hoc diagnosis, not part of the
pre-registered benchmark.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_prepared
from multi_dataset_benchmark import h1_summary, mantel_block
from omictopo.geometry import pairwise_distances, standardize

def one_sample_per_time(X, t, rng):
    idx = [int(rng.choice(np.where(t == tt)[0])) for tt in sorted(set(t.tolist()))]
    return np.asarray(idx, dtype=int)

def main(outdir="data/benchmark", n_trials=60, seed=0):
    X, cond, t, rep, _ = load_prepared(os.path.join("data", "datasets", "ath_diurnal"))
    m = cond == "control"
    Xc, tc = X[m], t[m]

    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_trials):
        idx = one_sample_per_time(Xc, tc, rng)
        Z = standardize(Xc[idx])
        D = pairwise_distances(Z, metric="correlation")
        h1 = h1_summary(D)["h1_max_rel"]
        mb = mantel_block(Xc[idx], tc[idx], 24.0, n_perm=1000, seed=k)
        rows.append({
            "trial": k, "n": int(idx.size),
            "h1_max_rel": round(h1, 4),
            "r_linear": mb["r_linear"],
            "best_r_circular": mb["best_r_circular"],
            "circular_advantage": mb["circular_advantage"],
            "best_period": mb["best_period"],
        })
        if (k + 1) % 20 == 0:
            print(f"  trial {k+1}/{n_trials}")

    h1 = np.array([r["h1_max_rel"] for r in rows])
    adv = np.array([r["circular_advantage"] for r in rows])
    per = np.array([r["best_period"] for r in rows])
    rc = np.array([r["best_r_circular"] for r in rows])
    summary = {
        "description": "GSE158444 control, one random sample per time point "
                       f"(n={rows[0]['n']}), {n_trials} draws",
        "h1_max_rel": {"median": round(float(np.median(h1)), 4),
                       "q25": round(float(np.quantile(h1, .25)), 4),
                       "q75": round(float(np.quantile(h1, .75)), 4)},
        "circular_advantage": {"median": round(float(np.median(adv)), 4),
                               "fraction_positive": round(float((adv > 0).mean()), 4)},
        "best_r_circular": {"median": round(float(np.median(rc)), 4)},
        "best_period": {"median": round(float(np.median(per)), 2)},
        "full_data_reference": {"n": 26, "h1_max_rel": 0.608,
                                 "r_linear": 0.647, "best_r_circular": 0.948},
        "gse137732_reference": {"h1_max_rel": 0.088, "r_linear": 0.527,
                                 "best_r_circular": 0.428},
        "trials": rows,
    }
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "diagnosis_replicates.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("\n=== replicate-dropout diagnosis (GSE158444 control) ===")
    print(f"  n per draw            : {rows[0]['n']} (one sample per time point)")
    print(f"  h1_max_rel  median    : {summary['h1_max_rel']['median']} "
          f"[IQR {summary['h1_max_rel']['q25']}..{summary['h1_max_rel']['q75']}]"
          f"   (full data: 0.608, GSE137732: 0.088)")
    print(f"  circular advantage    : median {summary['circular_advantage']['median']} "
          f"positive in {summary['circular_advantage']['fraction_positive']*100:.0f}% of draws"
          f"   (full data: +0.30, GSE137732: -0.10)")
    print(f"  best period  median   : {summary['best_period']['median']} h")
    print(f"  wrote {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
