"""Class-imbalance weights for the 3 severity classes -> data/processed/class_weights.json

Reads : data/processed/edges_labeled.csv
Writes: data/processed/class_weights.json
  counts, frequencies
  inverse_frequency     w_c = N / (K * n_c)                   (weighted CE / weighted sampling)
  effective_number      Cui et al. 2019, beta=0.9999, mean-normalised
  focal_alpha           inverse frequency normalised to sum 1  (alpha for focal loss)
  focal_gamma           2.0 default
  sampler_class_weight  1 / n_c  (per-sample weight = sampler_class_weight[label])
Recompute on the TRAIN split only once eval/split.py exists:
  from data.class_weights import compute_class_weights; compute_class_weights(train_labels)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROC_DIR = Path(__file__).resolve().parent / "processed"
SEVERITY_NAMES = ["minor", "moderate", "major"]


def compute_class_weights(labels, num_classes=3, beta=0.9999, gamma=2.0):
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    if (counts == 0).any():
        raise ValueError(f"empty class in labels: counts={counts.tolist()}")
    n = counts.sum()
    inv = n / (num_classes * counts)
    eff_n = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    eff = (1.0 / eff_n)
    eff = eff / eff.sum() * num_classes
    alpha = (1.0 / counts) / (1.0 / counts).sum()
    return {
        "classes": SEVERITY_NAMES[:num_classes],
        "counts": counts.astype(int).tolist(),
        "frequencies": (counts / n).round(6).tolist(),
        "inverse_frequency": inv.round(6).tolist(),
        "effective_number": eff.round(6).tolist(),
        "effective_number_beta": beta,
        "focal_alpha": alpha.round(6).tolist(),
        "focal_gamma": gamma,
        "sampler_class_weight": (1.0 / counts).tolist(),
    }


def main(in_dir=PROC_DIR):
    in_dir = Path(in_dir)
    lab = pd.read_csv(in_dir / "edges_labeled.csv")
    w = compute_class_weights(lab["severity"].values)
    (in_dir / "class_weights.json").write_text(json.dumps(w, indent=2))
    print(json.dumps(w, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PROC_DIR))
    main(ap.parse_args().dir)