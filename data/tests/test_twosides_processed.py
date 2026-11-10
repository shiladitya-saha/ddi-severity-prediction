import gzip, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data import twosides_processed as T


def test_names_and_severity():
    assert T.norm_name("Warfarin Sodium (Coumadin)")[1] == "warfarin"
    assert T.heuristic_severity(0, 0) == -1
    assert T.heuristic_severity(1, 1) == 2
    assert T.heuristic_severity(5, 0) == 1
    assert T.heuristic_severity(2, 0) == 0


def test_pipeline(tmp_path):
    proc = tmp_path / "proc"; proc.mkdir()
    pd.DataFrame({"node_idx": [0, 1, 2], "drug_id": ["DB1", "DB2", "DB3"],
                  "name": ["Warfarin", "Aspirin", "Zzzdrug"]}).to_csv(proc / "nodes.csv", index=False)
    pd.DataFrame({"src": [0], "dst": [1], "description": ["bleeding"], "severity": [2],
                  "severity_name": ["major"], "rule": ["x"]}).to_csv(proc / "edges_labeled.csv", index=False)
    rows = []
    for cond, prr in [("Haemorrhage", 5), ("Nausea", 3), ("Nausea", 1.1)]:
        rows.append(["1", "Warfarin Sodium", "2", "Aspirin", "c", cond, 10, 1, 1, 1, prr, 0.1, 0.1])
    rows += [["3", "Unknownium", "2", "Aspirin", "c", "Nausea", 10, 1, 1, 1, 3, 0.1, 0.1]]
    df = pd.DataFrame(rows, columns=["drug_1_rxnorn_id", "drug_1_concept_name", "drug_2_rxnorm_id",
                                     "drug_2_concept_name", "condition_meddra_id",
                                     "condition_concept_name", "A", "B", "C", "D", "PRR",
                                     "PRR_error", "mean_reporting_frequency"])
    raw = tmp_path / "ts.csv.gz"; df.to_csv(raw, index=False, compression="gzip")
    m = T.run(raw, proc, chunksize=2)
    assert len(m) == 1
    r = m.iloc[0]
    assert (r.src, r.dst) == (0, 1) and r.twosides_severity == 2 and r.n_serious == 1
    assert bool(r.in_drugbank_graph) and r.drugbank_severity == 2
    assert (proc / "twosides_summary.json").exists()