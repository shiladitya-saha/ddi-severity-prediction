import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data import label_severity as L, class_weights as C


def test_rules():
    assert L.classify("QTc-prolonging activities")[0] == 2
    assert L.classify("serum concentration")[0] == 0
    assert L.classify("risk or severity of adverse effects")[0] == 1
    assert L.classify("metabolism || risk or severity of bleeding")[0] == 2
    assert L.classify("")[0] == 1
    assert L.classify("metabolism", {"metabolism": 2})[0] == 2


def test_weights():
    w = C.compute_class_weights([0] * 70 + [1] * 20 + [2] * 10)
    assert w["inverse_frequency"][2] > w["inverse_frequency"][0]
    assert abs(sum(w["focal_alpha"]) - 1) < 1e-4


def test_pipeline(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir()
    pd.DataFrame({"node_idx": [0, 1, 2], "drug_id": ["a", "b", "c"], "name": "",
                  "smiles": "", "groups": ""}).to_csv(raw / "drugbank_graph_nodes.csv", index=False)
    pd.DataFrame({"src": [0, 0, 1], "dst": [1, 2, 2],
                  "description": ["metabolism", "QTc-prolonging activities",
                                  "hypotensive activities"]}).to_csv(
        raw / "drugbank_graph_edges.csv", index=False)
    L.main(raw, out, tmp_path / "none.csv")
    C.main(out)
    assert (out / "class_weights.json").exists()
    assert sorted(pd.read_csv(out / "edges_labeled.csv")["severity"]) == [0, 1, 2]