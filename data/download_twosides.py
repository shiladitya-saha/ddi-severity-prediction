"""Download TWOSIDES into data/raw/twosides_raw.csv.gz.

Sources (first match wins): --file (already downloaded), --hf-repo, --url.
--pairs-only also writes twosides_pairs_raw.csv (one row per drug pair).
"""
import argparse
import shutil
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
DEFAULT_URL = "https://tatonettilab-resources.s3.amazonaws.com/nsides/TWOSIDES.csv.gz"


def download_url(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def download_hf(repo, dest):
    from datasets import load_dataset
    ds = load_dataset(repo, split="train")
    ds.to_pandas().to_csv(dest, index=False, compression="gzip")


def _find(cols, *must, any_of=()):
    """First column whose lowercase name contains all `must` substrings (and one of any_of)."""
    for c in cols:
        lc = c.lower().replace(" ", "_")
        if all(m in lc for m in must) and (not any_of or any(a in lc for a in any_of)):
            return c
    return None


def build_pairs(src, dst, chunksize=500_000):
    import sys
    import pandas as pd
    head = pd.read_csv(src, nrows=0)
    cols = [c.strip() for c in head.columns]
    d1 = _find(cols, "drug", "1", any_of=("rxnorm", "rxnorn", "id"))
    d2 = _find(cols, "drug", "2", any_of=("rxnorm", "rxnorn", "id"))
    cond = (_find(cols, "condition", "meddra") or _find(cols, "condition", "id")
            or _find(cols, "condition", "name") or _find(cols, "condition"))
    prr = next((c for c in cols if c.lower() == "prr"), None)
    print(f"columns: {cols}\ndetected -> drug1={d1} drug2={d2} condition={cond} prr={prr}")
    if not (d1 and d2 and cond):
        sys.exit("Could not detect drug/condition columns; edit build_pairs with the names above.")

    keys = [d1, d2]
    parts = []
    for ch in pd.read_csv(src, chunksize=chunksize, low_memory=False):
        ch.columns = [c.strip() for c in ch.columns]
        agg = {"n_side_effects": (cond, "nunique")}
        if prr:
            agg["max_prr"] = (prr, "max")
        parts.append(ch.groupby(keys).agg(**agg).reset_index())
    cat = pd.concat(parts)
    final = {"n_side_effects": ("n_side_effects", "sum")}
    if prr:
        final["max_prr"] = ("max_prr", "max")
    pairs = cat.groupby(keys).agg(**final).reset_index()
    pairs = pairs.rename(columns={d1: "drug_1_id", d2: "drug_2_id"})
    pairs.to_csv(dst, index=False)
    print(f"{len(pairs)} drug pairs -> {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--file", help="already-downloaded TWOSIDES.csv.gz (skips download)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--hf-repo", help="HuggingFace dataset id instead of --url")
    ap.add_argument("--out", default=str(RAW_DIR))
    ap.add_argument("--pairs-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "twosides_raw.csv.gz"
    if a.file:
        shutil.copyfile(a.file, dest)
    elif a.hf_repo:
        download_hf(a.hf_repo, dest)
    else:
        download_url(a.url, dest)
    print(f"Saved {dest}")
    if a.pairs_only:
        build_pairs(dest, out / "twosides_pairs_raw.csv")


if __name__ == "__main__":
    main()