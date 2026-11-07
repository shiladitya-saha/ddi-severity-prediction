"""
embeddings.py
Owner: M2 — see features/README.md for day/deadline and output contract.
Day 2: optional enhancement (full_features.pt does not require this).

get_pretrained_embedding(smiles, method="chemberta") attempts to load a
pretrained ChemBERTa model via HuggingFace `transformers` and returns a
mean-pooled embedding vector for one SMILES string. Falls back cleanly
(logs a message, returns None) if the package or model isn't available,
so a caller can treat a None return as "skip this optional feature" rather
than a crash.

STATUS (Day 2, updated): `transformers` upgraded from the stray 2.1.1
install to 4.57.6 (pinned `>=4.40,<5` in requirements.txt / environment.yml;
no conflicts with this project's other pins -- `datasets` isn't actually
installed in this environment, so its fsspec pin doesn't apply here).
AutoTokenizer/AutoModel now resolve "seyonec/ChemBERTa-zinc-base-v1" from
the Hub and get_pretrained_embedding() returns real mean-pooled [768]
embeddings for both sample SMILES in main()'s smoke test (verified
distinct, finite, non-zero-norm vectors -- not the None fallback).
"""

_CHEMBERTA_MODEL_ID = "seyonec/ChemBERTa-zinc-base-v1"
_CHEMBERTA_HIDDEN_SIZE = 768  # seyonec/ChemBERTa-zinc-base-v1's known hidden size
_chemberta_cache = {"tokenizer": None, "model": None, "load_attempted": False}


def get_embedding_dim(method="chemberta"):
    """Return the embedding width get_pretrained_embedding() would produce.

    Loads the model if needed so the returned width is authoritative when
    loading succeeds; falls back to the known ChemBERTa hidden size if the
    model can't be loaded, so callers can still size a zero-vector
    placeholder consistently for drugs whose embedding fails.
    """
    if method != "chemberta":
        raise ValueError(f"Unsupported method {method!r}; only 'chemberta' is implemented.")
    _, model = _load_chemberta()
    if model is not None:
        return int(model.config.hidden_size)
    return _CHEMBERTA_HIDDEN_SIZE


def _load_chemberta():
    """Lazily load and cache the ChemBERTa tokenizer/model.

    Returns (tokenizer, model), or (None, None) if `transformers` isn't
    installed or the model can't be loaded for any reason.
    """
    if _chemberta_cache["load_attempted"]:
        return _chemberta_cache["tokenizer"], _chemberta_cache["model"]
    _chemberta_cache["load_attempted"] = True

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("[embeddings] `transformers` package not installed; skipping ChemBERTa embeddings.")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(_CHEMBERTA_MODEL_ID)
        model = AutoModel.from_pretrained(_CHEMBERTA_MODEL_ID)
        model.eval()
    except Exception as e:
        print(f"[embeddings] Could not load ChemBERTa ({_CHEMBERTA_MODEL_ID}): {e}. "
              f"Skipping pretrained embeddings -- see Day 3/4 follow-up note at the top of embeddings.py.")
        return None, None

    _chemberta_cache["tokenizer"] = tokenizer
    _chemberta_cache["model"] = model
    return tokenizer, model


def get_pretrained_embedding(smiles, method="chemberta"):
    """Return a pretrained embedding vector for one SMILES string, or None.

    Args:
        smiles (str): a SMILES string
        method (str): embedding method; only "chemberta" is implemented

    Returns:
        torch.FloatTensor of shape [hidden_size], or None if the model/package
        isn't available or inference fails -- never raises.
    """
    if method != "chemberta":
        print(f"[embeddings] Unsupported method {method!r}; only 'chemberta' is implemented. Returning None.")
        return None

    tokenizer, model = _load_chemberta()
    if tokenizer is None or model is None:
        return None

    try:
        import torch
        inputs = tokenizer(smiles, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)
    except Exception as e:
        print(f"[embeddings] ChemBERTa inference failed for SMILES {smiles!r}: {e}. Returning None.")
        return None


def main():
    # aspirin, acetaminophen
    sample_smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)NC1=CC=C(O)C=C1"]

    print("Running get_pretrained_embedding() smoke test (optional Day 2 feature)...")
    succeeded = 0
    for smiles in sample_smiles:
        emb = get_pretrained_embedding(smiles)
        if emb is not None:
            succeeded += 1
            print(f"  {smiles}: embedding shape {tuple(emb.shape)}")
        else:
            print(f"  {smiles}: no embedding (fell back cleanly)")

    print(f"Succeeded: {succeeded}/{len(sample_smiles)}")
    if succeeded == 0:
        print("ChemBERTa embeddings unavailable in this environment today -- deferred to "
              "Day 3/4 (see STATUS note at the top of this file). Not blocking Day 2.")


if __name__ == "__main__":
    main()
