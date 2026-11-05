"""
data/download_drugbank.py
Owner: M1 (Data Lead)

Acquires and extracts DrugBank 5.0 raw data:
- Parses local DrugBank XML release (`full database.xml`) into tabular DDI pairs
- Extracts drug metadata (DrugBank ID, SMILES, common name, type)
- Falls back to public open-access DDI mirrors (TDC/BioSNAP) if XML is absent
- Outputs:
    - data/raw/drugbank_raw.parquet (or .csv)
    - data/raw/drugbank_raw_meta.json
"""

import os
import sys
import json
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_drugbank")

RAW_DIR = Path(__file__).resolve().parent / "raw"
DEFAULT_OUTPUT_FILE = RAW_DIR / "drugbank_raw.parquet"
DEFAULT_META_FILE = RAW_DIR / "drugbank_raw_meta.json"
XML_NAMESPACE = "{http://www.drugbank.ca}"


def parse_drugbank_xml(xml_path: Path) -> pd.DataFrame:
    """
    Incrementally stream and parse the large DrugBank 5.0 XML database.
    Extracts:
        - drugbank_id, name, smiles
        - drug-drug interactions (drug1_id, drug2_id, description)
    """
    logger.info(f"Streaming and parsing DrugBank XML from: {xml_path}")
    interactions: List[Dict[str, str]] = []
    drugs: Dict[str, Dict[str, Optional[str]]] = {}

    context = ET.iterparse(str(xml_path), events=("end",))
    
    for event, elem in context:
        if elem.tag == f"{XML_NAMESPACE}drug":
            # Primary DrugBank ID
            db_id_elem = elem.find(f"{XML_NAMESPACE}drugbank-id[@primary='true']")
            if db_id_elem is None:
                db_id_elem = elem.find(f"{XML_NAMESPACE}drugbank-id")
            
            if db_id_elem is not None and db_id_elem.text:
                drug_id = db_id_elem.text.strip()
                name_elem = elem.find(f"{XML_NAMESPACE}name")
                name = name_elem.text.strip() if name_elem is not None and name_elem.text else ""
                drug_type = elem.attrib.get("type", "small molecule")

                # Extract Canonical SMILES
                smiles = None
                calc_props = elem.find(f"{XML_NAMESPACE}calculated-properties")
                if calc_props is not None:
                    for prop in calc_props.findall(f"{XML_NAMESPACE}property"):
                        kind = prop.find(f"{XML_NAMESPACE}kind")
                        if kind is not None and kind.text in ["SMILES", "Canonical SMILES"]:
                            val = prop.find(f"{XML_NAMESPACE}value")
                            if val is not None and val.text:
                                smiles = val.text.strip()
                                break

                drugs[drug_id] = {
                    "drug_id": drug_id,
                    "name": name,
                    "type": drug_type,
                    "smiles": smiles,
                }

                # Extract Drug-Drug Interactions
                ddi_list = elem.find(f"{XML_NAMESPACE}drug-interactions")
                if ddi_list is not None:
                    for ddi in ddi_list.findall(f"{XML_NAMESPACE}drug-interaction"):
                        partner_id_elem = ddi.find(f"{XML_NAMESPACE}drugbank-id")
                        desc_elem = ddi.find(f"{XML_NAMESPACE}description")
                        if partner_id_elem is not None and partner_id_elem.text:
                            partner_id = partner_id_elem.text.strip()
                            desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""
                            interactions.append({
                                "drug_id_1": drug_id,
                                "drug_id_2": partner_id,
                                "interaction_text": desc,
                            })

            # Clear element from memory to prevent memory bloat
            elem.clear()

    logger.info(f"Parsed {len(drugs):,} drugs and {len(interactions):,} raw directional interaction pairs.")

    # Convert to DataFrame and enrich with drug names / SMILES
    df = pd.DataFrame(interactions)
    df["smiles_1"] = df["drug_id_1"].map(lambda x: drugs.get(x, {}).get("smiles"))
    df["smiles_2"] = df["drug_id_2"].map(lambda x: drugs.get(x, {}).get("smiles"))
    df["name_1"] = df["drug_id_1"].map(lambda x: drugs.get(x, {}).get("name"))
    df["name_2"] = df["drug_id_2"].map(lambda x: drugs.get(x, {}).get("name"))

    return df


def download_open_ddi_fallback() -> pd.DataFrame:
    """
    Fallback loader for open-source DDI benchmarks (e.g., TDC DrugBank benchmark)
    when the licensed DrugBank XML cannot be found locally.
    """
    logger.info("Attempting to load DrugBank DDI dataset via Therapeutics Data Commons (TDC)...")
    try:
        from tdc.single_pred import DDI
        data = DDI(name="DrugBank")
        split = data.get_split()
        df = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)
        # Standardize columns
        df = df.rename(columns={
            "Drug1_ID": "drug_id_1",
            "Drug2_ID": "drug_id_2",
            "Drug1": "smiles_1",
            "Drug2": "smiles_2",
            "Y": "interaction_text",
        })
        logger.info(f"Successfully retrieved {len(df):,} pairs from open mirror.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch open mirror: {e}")
        raise RuntimeError(
            "DrugBank XML not found and TDC fallback unavailable. "
            "Please download 'full database.xml' from https://go.drugbank.com/releases/latest "
            "and place it inside 'data/raw/'."
        )


def main():
    parser = argparse.ArgumentParser(description="Download and parse DrugBank 5.0 raw data.")
    parser.add_argument("--xml-path", type=str, default=None, help="Path to full database.xml")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE), help="Output path")
    parser.add_argument("--force-fallback", action="store_true", help="Force using open-access mirror")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)

    # Locate XML
    xml_candidate = Path(args.xml_path) if args.xml_path else RAW_DIR / "full database.xml"
    if not xml_candidate.exists():
        xml_candidate = RAW_DIR / "drugbank.xml"

    if xml_candidate.exists() and not args.force_fallback:
        df = parse_drugbank_xml(xml_candidate)
        source = f"DrugBank 5.0 XML ({xml_candidate.name})"
    else:
        logger.warning(f"DrugBank XML not found at {xml_candidate}. Falling back to open mirror.")
        df = download_open_ddi_fallback()
        source = "Open-Access DDI Mirror (TDC)"

    # Save output
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    # Save metadata summary
    metadata = {
        "source": source,
        "num_records": len(df),
        "unique_drugs": len(set(df["drug_id_1"]).union(set(df["drug_id_2"]))),
        "has_smiles": bool("smiles_1" in df.columns and df["smiles_1"].notna().any()),
        "output_file": str(output_path.name),
    }
    with open(DEFAULT_META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done! Saved {len(df):,} interactions to {output_path}")
    logger.info(f"Metadata saved to {DEFAULT_META_FILE}")


if __name__ == "__main__":
    main()
