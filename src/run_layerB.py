import re
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


VocabType = Union[Dict[str, int], List[str], Tuple[str, ...]]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    if isinstance(vocab, dict):
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


def vocab_ordered_tokens(word_to_ix: Dict[str, int]) -> List[str]:
    return [w for w, _ in sorted(word_to_ix.items(), key=lambda kv: kv[1])]


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
    X = np.stack([W_cur[w2i_cur[w]] for w in anchors], axis=0)
    Y = np.stack([W_ref[w2i_ref[w]] for w in anchors], axis=0)

    Xn = normalize_rows(X)
    Yn = normalize_rows(Y)

    R, _ = orthogonal_procrustes(Xn, Yn)
    return W_cur @ R


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

    tokens = tech_cfg.get(version, tech_cfg.get("base", []))
    v, hits = mean_vec(tokens, W, w2i)
    meta = {"concept": concept_name, "type": "seed", "hits": hits, "n_seeds": len(tokens)}
    return v, meta


def top_neighbors(
    concept_vec: np.ndarray,
    W: np.ndarray,
    w2i: Dict[str, int],
    topn: int,
    tok_ok,
) -> List[Tuple[str, float]]:
    tokens = vocab_ordered_tokens(w2i)
    V = normalize_rows(W)
    c = normalize_vec(concept_vec)
    sims = V @ c  # cosine if rows normalized

    # over-generate then filter
    idx_sorted = np.argsort(-sims)[: max(topn * 5, topn)]
    out = []
    for idx in idx_sorted:
        tok = tokens[idx]
        if not tok_ok(tok):
            continue
        out.append((tok, float(sims[idx])))
        if len(out) >= topn:
            break
    return out


def cluster_neighbors(tokens: List[str], W: np.ndarray, w2i: Dict[str, int], k_min: int, k_max: int, seed: int):
    X = np.stack([W[w2i[t]] for t in tokens], axis=0)
    Xn = normalize_rows(X)

    best = None
    for k in range(k_min, k_max + 1):
        if k >= len(tokens):
            break
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(Xn)
        # silhouette needs >=2 clusters and enough points
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(Xn, labels, metric="cosine")
        if best is None or score > best["silhouette"]:
            best = {"k": k, "silhouette": float(score), "labels": labels.tolist()}
    return best


def parse_decade_from_filename(name: str) -> Optional[str]:
    m = re.match(r"^(\d{4})-w\.npy$", name)
    return m.group(1) if m else None


def main():
    root = Path(__file__).resolve().parents[1]
    cfg_run = load_yaml(root / "configs" / "run.yaml")
    cfg_seeds = load_yaml(root / "configs" / "seeds.yaml")

    raw_dir = Path(cfg_run["paths"]["raw_dir"]).expanduser()
    out_dir = (root / "results" / "emergent_topics").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover decades
    w_files = sorted(raw_dir.glob(cfg_run["data"]["decade_glob"]))
    decades = sorted({parse_decade_from_filename(p.name) for p in w_files if parse_decade_from_filename(p.name)})
    if not decades:
        raise RuntimeError("No decades found")

    # alignment setup
    ref_decade = str(cfg_run["alignment"]["reference_decade"])
    if ref_decade not in decades:
        raise RuntimeError(f"Reference decade {ref_decade} not found")

    tok_ok = build_token_filter(
        exclude_regex=cfg_run["neighbors"]["filter"].get("exclude_regex", []),
        min_len=int(cfg_run["neighbors"]["filter"].get("min_length", 2)),
    )

    anchors_ok = build_token_filter(
        exclude_regex=cfg_run["alignment"]["anchors"].get("exclude_regex", []),
        min_len=int(cfg_run["alignment"]["anchors"].get("min_token_length", 2)),
    )
    anchors_topk = int(cfg_run["alignment"]["anchors"].get("topk", 50000))

    # load reference
    W_ref = load_W(raw_dir / f"{ref_decade}{cfg_run['data']['vector_suffix']}")
    w2i_ref = load_vocab(raw_dir / f"{ref_decade}{cfg_run['data']['vocab_suffix']}")

    concepts = cfg_run["concepts"]["tech_concepts_to_run"]
    topn = int(cfg_run["neighbors"]["topn"])
    k_min, k_max = cfg_run["clustering"]["k_range"]
    seed = int(cfg_run["clustering"]["random_seed"])

    # plotting filter rules (apply here too, to avoid garbage topics)
    min_hits = 3

    records = []

    for decade in decades:
        W = load_W(raw_dir / f"{decade}{cfg_run['data']['vector_suffix']}")
        w2i = load_vocab(raw_dir / f"{decade}{cfg_run['data']['vocab_suffix']}")

        if cfg_run["alignment"]["enabled"] and decade != ref_decade:
            anchors = choose_anchors(w2i_ref, w2i, topk=anchors_topk, tok_ok=anchors_ok)
            W_use = align_to_reference(W, w2i, W_ref, w2i_ref, anchors)
        else:
            W_use = W

        for concept in concepts:
            # AI starts from 1950 for interpretability
            if concept == "ai" and int(decade) < 1950:
                continue

            cvec, meta = concept_centroid(concept, cfg_seeds, W_use, w2i, version="base")
            if cvec is None:
                continue

            # apply universal plotting threshold here to keep topics meaningful
            if meta.get("hits", 0) < min_hits:
                continue

            # phone_social requires both sides
            if concept == "phone_social":
                sh = meta.get("sub_hits", {})
                if sh.get("smartphone", 0) < min_hits or sh.get("social_media", 0) < min_hits:
                    continue

            neigh = top_neighbors(cvec, W_use, w2i, topn=topn, tok_ok=tok_ok)
            neigh_tokens = [t for t, _ in neigh]

            best = cluster_neighbors(neigh_tokens, W_use, w2i, k_min=k_min, k_max=k_max, seed=seed)
            if best is None:
                continue

            # build clusters
            labels = best["labels"]
            clusters = {}
            for t, lab in zip(neigh, labels):
                tok, sim = t
                clusters.setdefault(str(lab), []).append({"token": tok, "sim": sim})

            # sort tokens within clusters by similarity
            for lab in clusters:
                clusters[lab] = sorted(clusters[lab], key=lambda x: -x["sim"])

            payload = {
                "decade": int(decade),
                "concept": concept,
                "concept_hits": int(meta.get("hits", 0)),
                "sub_hits": meta.get("sub_hits", None),
                "topn": topn,
                "k": best["k"],
                "silhouette": best["silhouette"],
                "clusters": clusters,
            }

            # write JSON per decade per concept
            concept_dir = out_dir / concept
            concept_dir.mkdir(parents=True, exist_ok=True)
            out_json = concept_dir / f"{decade}_topics.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            records.append({
                "decade": int(decade),
                "concept": concept,
                "concept_hits": int(meta.get("hits", 0)),
                "k": best["k"],
                "silhouette": best["silhouette"],
                "json_path": str(out_json),
            })

    # summary index
    idx = pd.DataFrame(records).sort_values(["concept", "decade"])
    idx_path = out_dir / "topics_index.csv"
    idx.to_csv(idx_path, index=False)
    print(f"[OK] wrote {idx_path}")
    print(f"[OK] topics saved under {out_dir}/<concept>/YYYY_topics.json")


if __name__ == "__main__":
    main()