"""
download_twosides.py
Owner: M1 (Data Lead)

Pulls the TWOSIDES dataset (~63K drug-drug pairs with polypharmacy side effects)
from Hugging Face / TDC for cross-dataset validation (consumed by M5 on Day 3/4).

Outputs:
  - data/raw/twosides_raw.parquet (or .csv)
  - data/raw/twosides_raw_meta.json
"""

import os
import sys
import json
import logging
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_twosides")

RAW_DIR = Path(__file__).resolve().parent / "raw"
DEFAULT_OUTPUT_FILE = RAW_DIR / "twosides_raw.parquet"
DEFAULT_META_FILE = RAW_DIR / "twosides_raw_meta.json"

# Known public mirrors for TWOSIDES (HuggingFace Hub / TDC)
HF_TWOSIDES_URL = "https://huggingface.co/datasets/mims-harvard/TDC/resolve/main/twosides.csv.gz"
DATAVERSE_TWOSIDES_URL = "https://dataverse.harvard.edu/api/access/datafile/4264669"


def download_from_huggingface(output_path: Path) -> pd.DataFrame:
    """
    Attempt to load TWOSIDES using huggingface `datasets` or direct download
    from Hugging Face Hub / TDC mirror.
    """
    logger.info("Attempting to load TWOSIDES from Hugging Face / TDC mirror...")
    
    # 1. Try Hugging Face `datasets` library if installed
    try:
        from datasets import load_dataset
        logger.info("Using Hugging Face `datasets` library...")
        # Check standard huggingface repos
        dataset = load_dataset("reach-ai/twosides", split="train")
        df = dataset.to_pandas()
        logger.info(f"Loaded {len(df):,} records via HF datasets library.")
        return standardize_twosides_df(df)
    except Exception as e:
        logger.info(f"HF `datasets` direct load skipped ({e}). Falling back to URL download.")

    # 2. Try TDC library if installed
    try:
        from tdc.multi_pred import DDI
        logger.info("Using TDC DDI(name='TWOSIDES')...")
        data = DDI(name="TWOSIDES")
        df = data.get_data()
        logger.info(f"Loaded {len(df):,} records via TDC.")
        return standardize_twosides_df(df)
    except Exception as e:
        logger.info(f"TDC library load skipped ({e}). Trying direct HTTP download from mirror.")

    # 3. Direct HTTP download of gz/csv
    cache_file = RAW_DIR / "twosides_download.csv.gz"
    for url in [HF_TWOSIDES_URL, DATAVERSE_TWOSIDES_URL]:
        try:
            logger.info(f"Fetching from: {url}")
            urllib.request.urlretrieve(url, cache_file)
            logger.info("Download complete. Reading into pandas DataFrame...")
            df = pd.read_csv(cache_file, compression="infer", low_memory=False)
            return standardize_twosides_df(df)
        except Exception as err:
            logger.warning(f"Download from {url} failed: {err}")

    raise RuntimeError(
        "Could not automatically download TWOSIDES. Please manually place the TWOSIDES CSV "
        "file in 'data/raw/twosides.csv' or run with '--sample' to generate a mock subset for testing."
    )


def standardize_twosides_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize TWOSIDES columns into a consistent schema:
    [drug_id_1, drug_id_2, name_1, name_2, side_effect_name, side_effect_id]
    """
    rename_candidates = {
        # Drug 1
        "Drug1_ID": "drug_id_1",
        "drug_1": "drug_id_1",
        "drug_1_id": "drug_id_1",
        "Drug1": "drug_id_1",
        "stitch_id1": "drug_id_1",
        # Drug 2
        "Drug2_ID": "drug_id_2",
        "drug_2": "drug_id_2",
        "drug_2_id": "drug_id_2",
        "Drug2": "drug_id_2",
        "stitch_id2": "drug_id_2",
        # Side effect
        "Y": "side_effect_name",
        "side_effect": "side_effect_name",
        "condition_concept_name": "side_effect_name",
        "event_name": "side_effect_name",
        # Side effect concept code
        "condition_meddra_id": "side_effect_id",
        "side_effect_id": "side_effect_id",
        "meddra_id": "side_effect_id",
    }
    
    df = df.rename(columns={k: v for k, v in rename_candidates.items() if k in df.columns})

    # Ensure required columns exist
    if "drug_id_1" not in df.columns or "drug_id_2" not in df.columns:
        # Fall back to positional if two ID-like columns exist
        cols = list(df.columns)
        if len(cols) >= 2:
            df = df.rename(columns={cols[0]: "drug_id_1", cols[1]: "drug_id_2"})
            if len(cols) >= 3 and "side_effect_name" not in df.columns:
                df = df.rename(columns={cols[2]: "side_effect_name"})

    if "side_effect_name" not in df.columns:
        df["side_effect_name"] = "unspecified_adverse_event"

    df["drug_id_1"] = df["drug_id_1"].astype(str).str.strip()
    df["drug_id_2"] = df["drug_id_2"].astype(str).str.strip()
    df["side_effect_name"] = df["side_effect_name"].astype(str).str.strip()

    return df


def generate_sample_twosides(num_pairs: int = 500) -> pd.DataFrame:
    """
    Generates a synthetic sample for offline testing and pipeline dry-runs.
    """
    logger.info(f"Generating synthetic sample of {num_pairs} TWOSIDES records for testing...")
    sample_drugs = [f"DB{1000 + i:05d}" for i in range(50)]
    sample_effects = [
        "Hypotension", "Hyperkalemia", "QT prolongation", "Gastrointestinal hemorrhage",
        "Serotonin syndrome", "Headache", "Nausea", "Dizziness", "Bradycardia", "Renal failure"
    ]
    
    rows = []
    import random
    random.seed(42)
    for _ in range(num_pairs):
        d1, d2 = random.sample(sample_drugs, 2)
        effect = random.choice(sample_effects)
        rows.append({
            "drug_id_1": d1,
            "drug_id_2": d2,
            "side_effect_name": effect,
            "side_effect_id": f"MEDDRA_{abs(hash(effect)) % 100000}",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Download TWOSIDES polypharmacy dataset.")
    parser.add_argument("--local-path", type=str, default=None, help="Path to local TWOSIDES file if pre-downloaded")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE), help="Output path (.parquet or .csv)")
    parser.add_argument("--sample", action="store_true", help="Generate a mock sample for offline dry-run")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)

    if args.sample:
        df = generate_sample_twosides(num_pairs=1000)
        source = "Synthetic dry-run sample (offline)"
    elif args.local_path and Path(args.local_path).exists():
        logger.info(f"Loading from specified local path: {args.local_path}")
        df = pd.read_csv(args.local_path, compression="infer", low_memory=False)
        df = standardize_twosides_df(df)
        source = f"Local file ({Path(args.local_path).name})"
    elif (RAW_DIR / "twosides.csv").exists():
        logger.info(f"Loading from existing raw file: {RAW_DIR / 'twosides.csv'}")
        df = pd.read_csv(RAW_DIR / "twosides.csv", low_memory=False)
        df = standardize_twosides_df(df)
        source = "Local data/raw/twosides.csv"
    else:
        df = download_from_huggingface(output_path)
        source = "HuggingFace / TDC Mirror"

    # Compute unique pairs (undirected count)
    pairs = set()
    for _, row in df.iterrows():
        pair = tuple(sorted([str(row["drug_id_1"]), str(row["drug_id_2"])]))
        pairs.add(pair)

    unique_drugs = set(df["drug_id_1"]).union(set(df["drug_id_2"]))

    # Save output
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    # Save metadata summary
    metadata = {
        "source": source,
        "num_records": len(df),
        "num_unique_drug_pairs": len(pairs),
        "num_unique_drugs": len(unique_drugs),
        "num_unique_side_effects": df["side_effect_name"].nunique() if "side_effect_name" in df.columns else 0,
        "output_file": str(output_path.name),
    }
    with open(DEFAULT_META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done! Saved {len(df):,} TWOSIDES records ({len(pairs):,} unique pairs) to {output_path}")
    logger.info(f"Metadata saved to {DEFAULT_META_FILE}")


if __name__ == "__main__":
    main()
