"""Ingest DrugBank-derived DDI data into data/raw/ (NEVER commit the output).

  1. Official XML (licensed):  python data/download_drugbank.py --xml /path/full_database.xml
  2. Open pair table (CSV):    python data/download_drugbank.py --csv /path/DDI_data.csv

Outputs (data/raw/):
  drugbank_drugs_raw.csv   drug_id, name, smiles, groups
  drugbank_raw.csv         drug_id_1, drug_id_2, description
"""
import argparse
import csv
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

RAW_DIR = Path(__file__).resolve().parent / "raw"
NS = "{http://www.drugbank.ca}"


def _text(elem, path):
    node = elem.find(path)
    return (node.text or "").strip() if node is not None and node.text else ""


def parse_xml(xml_path, out_dir=RAW_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_drugs = n_edges = 0
    with open(out_dir / "drugbank_drugs_raw.csv", "w", newline="", encoding="utf-8") as fd, \
         open(out_dir / "drugbank_raw.csv", "w", newline="", encoding="utf-8") as fe:
        wd, we = csv.writer(fd), csv.writer(fe)
        wd.writerow(["drug_id", "name", "smiles", "groups"])
        we.writerow(["drug_id_1", "drug_id_2", "description"])
        depth = 0
        for event, elem in ET.iterparse(str(xml_path), events=("start", "end")):
            if event == "start":
                depth += 1
                continue
            if elem.tag == NS + "drug" and depth == 2:
                did = ""
                for node in elem.findall(NS + "drugbank-id"):
                    if node.get("primary") == "true":
                        did = (node.text or "").strip()
                        break
                if did:
                    smiles = ""
                    for prop in elem.findall(f"{NS}calculated-properties/{NS}property"):
                        if _text(prop, NS + "kind") == "SMILES":
                            smiles = _text(prop, NS + "value")
                            break
                    groups = "|".join((g.text or "").strip()
                                      for g in elem.findall(f"{NS}groups/{NS}group"))
                    wd.writerow([did, _text(elem, NS + "name"), smiles, groups])
                    n_drugs += 1
                    for ix in elem.findall(f"{NS}drug-interactions/{NS}drug-interaction"):
                        other = _text(ix, NS + "drugbank-id")
                        if other:
                            we.writerow([did, other, _text(ix, NS + "description")])
                            n_edges += 1
                elem.clear()
            depth -= 1
    print(f"Parsed {n_drugs} drugs, {n_edges} directed interactions -> {out_dir}")


def _pick(df_cols, given, candidates):
    if given:
        if given not in df_cols:
            sys.exit(f"Column '{given}' not in CSV. Available: {list(df_cols)}")
        return given
    low = {c.lower().replace(" ", "").replace("_", ""): c for c in df_cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def from_csv(src, out_dir=RAW_DIR, id1=None, id2=None, label=None,
             name1=None, name2=None, smiles1=None, smiles2=None):
    """Ingest any open DDI pair table (local path or http(s) URL) -> data/raw/."""
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src, sep=None, engine="python", dtype=str).fillna("")
    cols = df.columns
    c_id1 = _pick(cols, id1, ["drug1id", "drugid1", "id1", "drugaid"])
    c_id2 = _pick(cols, id2, ["drug2id", "drugid2", "id2", "drugbid"])
    c_lab = _pick(cols, label, ["y", "label", "interactiontype", "type", "event", "description", "interaction"])
    if not (c_id1 and c_id2 and c_lab):
        sys.exit(f"Could not detect id/label columns in {list(cols)}; pass --id1 --id2 --label.")
    c_n1 = _pick(cols, name1, ["drug1name", "drugname1", "name1"])
    c_n2 = _pick(cols, name2, ["drug2name", "drugname2", "name2"])
    c_s1 = _pick(cols, smiles1, ["drug1", "drug1smiles", "smiles1"])
    c_s2 = _pick(cols, smiles2, ["drug2", "drug2smiles", "smiles2"])
    if c_s1 in (c_id1, c_n1, c_lab): c_s1 = None
    if c_s2 in (c_id2, c_n2, c_lab): c_s2 = None

    edges = pd.DataFrame({"drug_id_1": df[c_id1], "drug_id_2": df[c_id2],
                          "description": df[c_lab]})
    edges.to_csv(out_dir / "drugbank_raw.csv", index=False)

    def side(i, n, s):
        d = pd.DataFrame({"drug_id": df[i]})
        d["name"] = df[n] if n else ""
        d["smiles"] = df[s] if s else ""
        d["groups"] = ""
        return d
    drugs = pd.concat([side(c_id1, c_n1, c_s1), side(c_id2, c_n2, c_s2)])
    drugs["_k"] = (drugs["name"] != "").astype(int) + (drugs["smiles"] != "").astype(int)
    drugs = (drugs.sort_values("_k", ascending=False).drop_duplicates("drug_id")
             .drop(columns="_k"))
    drugs[["drug_id", "name", "smiles", "groups"]].to_csv(
        out_dir / "drugbank_drugs_raw.csv", index=False)
    print(f"CSV source: {len(drugs)} drugs, {len(edges)} interactions -> {out_dir}"
          f" (label col='{c_lab}', smiles={'yes' if c_s1 else 'NO'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--xml", help="path to DrugBank full_database.xml (licensed)")
    g.add_argument("--csv", help="open DDI pair table: local path or http(s) URL")
    ap.add_argument("--out", default=str(RAW_DIR))
    for a in ("id1", "id2", "label", "name1", "name2", "smiles1", "smiles2"):
        ap.add_argument(f"--{a}", help=f"column name for {a} (auto-detected if omitted)")
    a = ap.parse_args()
    if a.xml:
        parse_xml(a.xml, a.out)
    else:
        from_csv(a.csv, a.out, a.id1, a.id2, a.label, a.name1, a.name2, a.smiles1, a.smiles2)


if __name__ == "__main__":
    main()