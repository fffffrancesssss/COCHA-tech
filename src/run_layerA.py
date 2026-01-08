import re
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import orthogonal_procrustes


VocabType = Union[Dict[str, int], List[str], Tuple[str, ...]]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    if isinstance(vocab, dict):
        # assume word -> idx
        return vocab
    if isinstance(vocab, (list, tuple)):
        return {w: i for i, w in enumerate(vocab)}
    raise TypeError(f"Unsupported vocab type: {type(vocab)}")


def load_W(w_path: Path) -> np.ndarray:
    W = np.load(w_path)
    if not isinstance(W, np.ndarray) or W.ndim != 2:
        raise ValueError(f"Bad embedding matrix: {w_path} shape={getattr(W, 'shape', None)}")
    return W


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)


def cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    return float(u @ v / ((np.linalg.norm(u) + eps) * (np.linalg.norm(v) + eps)))


def build_token_filter(exclude_regex: List[str], min_len: int = 2):
    patterns = [re.compile(p) for p in exclude_regex]

    def ok(tok: str) -> bool:
        if tok is None:
            return False
        if len(tok) < min_len:
            return False
        for pat in patterns:
            if pat.search(tok):
                return False
        return True

    return ok


def vocab_ordered_tokens(word_to_ix: Dict[str, int]) -> List[str]:
    # sort by index to approximate "frequency rank" if vocab was built that way
    return [w for w, _ in sorted(word_to_ix.items(), key=lambda kv: kv[1])]


def choose_anchors(
    ref_w2i: Dict[str, int],
    cur_w2i: Dict[str, int],
    topk: int,
    tok_ok,
) -> List[str]:
    ref_tokens = vocab_ordered_tokens(ref_w2i)
    anchors = []
    cur_set = set(cur_w2i.keys())
    for w in ref_tokens:
        if w in cur_set and tok_ok(w):
            anchors.append(w)
            if len(anchors) >= topk:
                break
    return anchors


def align_to_reference(
    W_cur: np.ndarray,
    w2i_cur: Dict[str, int],
    W_ref: np.ndarray,
    w2i_ref: Dict[str, int],
    anchors: List[str],
) -> np.ndarray:
    # Build matrices for Procrustes: X (cur) and Y (ref), both shape (n_anchors, dim)
    X = np.stack([W_cur[w2i_cur[w]] for w in anchors], axis=0)
    Y = np.stack([W_ref[w2i_ref[w]] for w in anchors], axis=0)

    # Normalize rows (optional but helps stability)
    Xn = normalize_rows(X)
    Yn = normalize_rows(Y)

    R, _ = orthogonal_procrustes(Xn, Yn)  # minimizes ||X R - Y||, R orthogonal
    W_aligned = W_cur @ R
    return W_aligned


def mean_vec(tokens: List[str], W: np.ndarray, w2i: Dict[str, int]) -> Tuple[Optional[np.ndarray], int]:
    vecs = []
    for t in tokens:
        idx = w2i.get(t)
        if idx is None:
            continue
        vecs.append(W[idx])
    if not vecs:
        return None, 0
    M = np.mean(np.stack(vecs, axis=0), axis=0)
    return M, len(vecs)


def concept_centroid(
    concept_name: str,
    seeds_cfg: dict,
    W: np.ndarray,
    w2i: Dict[str, int],
    version: str = "base",
) -> Tuple[Optional[np.ndarray], dict]:
    tech_cfg = seeds_cfg["tech_concepts"][concept_name]

    # combined concept
    if "combine" in tech_cfg:
        subs = tech_cfg["combine"]
        sub_vecs = []
        meta = {"concept": concept_name, "type": "combine", "sub_hits": {}}
        for sub in subs:
            v_sub, info_sub = concept_centroid(sub, seeds_cfg, W, w2i, version=version)
            if v_sub is not None:
                sub_vecs.append(v_sub)
            meta["sub_hits"][sub] = info_sub.get("hits", 0)
        if not sub_vecs:
            meta["hits"] = 0
            return None, meta
        v = np.mean(np.stack(sub_vecs, axis=0), axis=0)
        meta["hits"] = sum(meta["sub_hits"].values())
        return v, meta

    # normal concept
    tokens = tech_cfg.get(version, tech_cfg.get("base", []))
    v, hits = mean_vec(tokens, W, w2i)
    meta = {"concept": concept_name, "type": "seed", "hits": hits, "n_seeds": len(tokens)}
    return v, meta


def build_sentiment_axis(concerns_cfg: dict, W: np.ndarray, w2i: Dict[str, int], version: str) -> Tuple[Optional[np.ndarray], dict]:
    pos = concerns_cfg["sentiment"]["positive"].get(version, concerns_cfg["sentiment"]["positive"]["base"])
    neg = concerns_cfg["sentiment"]["negative"].get(version, concerns_cfg["sentiment"]["negative"]["base"])
    v_pos, h_pos = mean_vec(pos, W, w2i)
    v_neg, h_neg = mean_vec(neg, W, w2i)
    meta = {"pos_hits": h_pos, "neg_hits": h_neg, "pos_n": len(pos), "neg_n": len(neg)}
    if v_pos is None or v_neg is None:
        return None, meta
    axis = normalize_vec(v_pos - v_neg)
    return axis, meta


def concern_vector(concerns_cfg: dict, name: str, W: np.ndarray, w2i: Dict[str, int], version: str) -> Tuple[Optional[np.ndarray], dict]:
    tokens = concerns_cfg["concerns"][name].get(version, concerns_cfg["concerns"][name]["base"])
    v, hits = mean_vec(tokens, W, w2i)
    meta = {"hits": hits, "n": len(tokens)}
    return (normalize_vec(v) if v is not None else None), meta


def baseline_assoc(
    W: np.ndarray,
    w2i: Dict[str, int],
    axis_vec: np.ndarray,
    sample_size: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    tokens = list(w2i.keys())
    if not tokens:
        return float("nan")
    n = min(sample_size, len(tokens))
    sample = rng.sample(tokens, n)
    vecs = np.stack([W[w2i[t]] for t in sample], axis=0)
    vecs = normalize_rows(vecs)
    a = normalize_vec(axis_vec)
    cos_vals = vecs @ a
    return float(np.mean(cos_vals))


def tech_assoc_to_axis(tech_vec: np.ndarray, axis_vec: np.ndarray) -> float:
    return cosine(tech_vec, axis_vec)


def parse_decade_from_filename(name: str) -> Optional[str]:
    # expects like "1900-w.npy"
    m = re.match(r"^(\d{4})-w\.npy$", name)
    return m.group(1) if m else None


def main():
    root = Path(__file__).resolve().parents[1]
    cfg_run = load_yaml(root / "configs" / "run.yaml")
    cfg_seeds = load_yaml(root / "configs" / "seeds.yaml")
    cfg_conc = load_yaml(root / "configs" / "concerns.yaml")

    raw_dir = Path(cfg_run["paths"]["raw_dir"]).expanduser()
    out_dir = (root / cfg_run["paths"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover decades
    w_files = sorted(raw_dir.glob(cfg_run["data"]["decade_glob"]))
    decades = []
    for p in w_files:
        d = parse_decade_from_filename(p.name)
        if d:
            decades.append(d)
    decades = sorted(set(decades))
    if not decades:
        raise RuntimeError(f"No decade embeddings found in {raw_dir} with glob={cfg_run['data']['decade_glob']}")

    ref_decade = str(cfg_run["alignment"]["reference_decade"])
    if ref_decade not in decades:
        raise RuntimeError(f"Reference decade {ref_decade} not found. Available: {decades[:5]} ... {decades[-5:]}")

    tok_ok = build_token_filter(
        exclude_regex=cfg_run["alignment"]["anchors"].get("exclude_regex", []),
        min_len=int(cfg_run["alignment"]["anchors"].get("min_token_length", 2)),
    )

    # load reference
    ref_w_path = raw_dir / f"{ref_decade}{cfg_run['data']['vector_suffix']}"
    ref_vocab_path = raw_dir / f"{ref_decade}{cfg_run['data']['vocab_suffix']}"
    W_ref = load_W(ref_w_path)
    w2i_ref = load_vocab(ref_vocab_path)

    # options
    tech_list = cfg_run["concepts"]["tech_concepts_to_run"]
    sub_list = cfg_run["concepts"].get("also_run_subconcepts", [])
    concepts_to_compute = tech_list + sub_list

    sentiment_version = cfg_run["axes"].get("sentiment_version", "base")
    concerns_version = cfg_run["axes"].get("concerns_version", "base")
    min_hits_axis = int(cfg_run["axes"].get("min_hits_per_axis", 5))
    min_hits_concept = int(cfg_run["axes"].get("min_hits_per_concept", 4))

    baseline_enabled = bool(cfg_run["baseline"]["enabled"])
    baseline_size = int(cfg_run["baseline"].get("sample_size", 5000))
    baseline_seed = int(cfg_run["baseline"].get("random_seed", 13))

    align_enabled = bool(cfg_run["alignment"]["enabled"])
    anchors_topk = int(cfg_run["alignment"]["anchors"].get("topk", 50000))

    rows = []

    for decade in decades:
        w_path = raw_dir / f"{decade}{cfg_run['data']['vector_suffix']}"
        vocab_path = raw_dir / f"{decade}{cfg_run['data']['vocab_suffix']}"

        W = load_W(w_path)
        w2i = load_vocab(vocab_path)

        # align
        if align_enabled and decade != ref_decade:
            anchors = choose_anchors(w2i_ref, w2i, topk=anchors_topk, tok_ok=tok_ok)
            if len(anchors) < 1000:
                # too few anchors => alignment unstable; still proceed but mark
                anchor_note = f"low_anchors:{len(anchors)}"
            else:
                anchor_note = f"anchors:{len(anchors)}"
            W_use = align_to_reference(W, w2i, W_ref, w2i_ref, anchors)
        else:
            W_use = W
            anchor_note = "reference" if decade == ref_decade else "unaligned"

        # build axes
        sent_axis, sent_meta = build_sentiment_axis(cfg_conc, W_use, w2i, sentiment_version)
        concern_names = list(cfg_conc["concerns"].keys())
        concern_vecs = {}
        concern_meta = {}
        for cn in concern_names:
            v_c, m_c = concern_vector(cfg_conc, cn, W_use, w2i, concerns_version)
            concern_vecs[cn] = v_c
            concern_meta[cn] = m_c

        # baseline for each axis
        baseline_cache = {}
        if baseline_enabled:
            if sent_axis is not None:
                baseline_cache["sentiment"] = baseline_assoc(W_use, w2i, sent_axis, baseline_size, baseline_seed)
            for cn in concern_names:
                if concern_vecs[cn] is not None:
                    baseline_cache[cn] = baseline_assoc(W_use, w2i, concern_vecs[cn], baseline_size, baseline_seed)
        else:
            baseline_cache = {k: 0.0 for k in (["sentiment"] + concern_names)}

        for concept in concepts_to_compute:
            tech_vec, tech_meta = concept_centroid(concept, cfg_seeds, W_use, w2i, version="base")

            row = {
                "decade": decade,
                "concept": concept,
                "align": anchor_note,
                "concept_hits": tech_meta.get("hits", 0),
            }

            # sentiment
            if sent_axis is None or sent_meta["pos_hits"] < min_hits_axis or sent_meta["neg_hits"] < min_hits_axis:
                row["sentiment_raw"] = np.nan
                row["sentiment_delta"] = np.nan
            elif tech_vec is None or tech_meta.get("hits", 0) < min_hits_concept:
                row["sentiment_raw"] = np.nan
                row["sentiment_delta"] = np.nan
            else:
                raw = tech_assoc_to_axis(tech_vec, sent_axis)
                base = baseline_cache.get("sentiment", 0.0)
                row["sentiment_raw"] = raw
                row["sentiment_delta"] = raw - base

            # concerns
            for cn in concern_names:
                v_c = concern_vecs[cn]
                hits_c = concern_meta[cn]["hits"]
                if v_c is None or hits_c < min_hits_axis:
                    row[f"{cn}_raw"] = np.nan
                    row[f"{cn}_delta"] = np.nan
                elif tech_vec is None or tech_meta.get("hits", 0) < min_hits_concept:
                    row[f"{cn}_raw"] = np.nan
                    row[f"{cn}_delta"] = np.nan
                else:
                    raw = tech_assoc_to_axis(tech_vec, v_c)
                    base = baseline_cache.get(cn, 0.0)
                    row[f"{cn}_raw"] = raw
                    row[f"{cn}_delta"] = raw - base

            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["concept", "decade"])
    out_csv = out_dir / "tech_trends_layerA.csv"
    df.to_csv(out_csv, index=False)

    # write a small run manifest for reproducibility
    manifest = {
        "raw_dir": str(raw_dir),
        "reference_decade": ref_decade,
        "alignment": cfg_run["alignment"],
        "baseline": cfg_run["baseline"],
        "concepts": cfg_run["concepts"],
        "axes": cfg_run["axes"],
        "n_rows": int(df.shape[0]),
    }
    with open(out_dir / "run_manifest_layerA.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_dir / 'run_manifest_layerA.json'}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()