from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

# -------- settings you can edit --------
PAIRS = [
    ("nuclear", 1970),
    ("nuclear", 1980),
    ("nuclear", 1990),
    ("ai", 1980),
    ("ai", 1990),
    ("ai", 2000),
]
TOP_TOKENS_PER_CLUSTER = 12  # show top 12 tokens for each cluster
# --------------------------------------


def load_topics(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_cluster(cluster_tokens: list[dict], topk: int) -> str:
    toks = [x["token"] for x in cluster_tokens[:topk]]
    return ", ".join(toks)


def main():
    root = Path(__file__).resolve().parents[1]
    idx_path = root / "results" / "emergent_topics" / "topics_index.csv"
    out_dir = root / "results" / "emergent_topics"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(idx_path)

    rows = []
    md_lines = []
    md_lines.append("# Emergent topics summary\n")

    for concept, decade in PAIRS:
        hit = idx[(idx["concept"] == concept) & (idx["decade"] == decade)]
        if hit.empty:
            md_lines.append(f"## {concept} {decade}\n")
            md_lines.append("Not found in topics_index.csv\n")
            continue

        json_path = Path(hit.iloc[0]["json_path"])
        payload = load_topics(json_path)

        md_lines.append(f"## {concept} {decade}\n")
        md_lines.append(f"- concept_hits: {payload.get('concept_hits')}\n")
        md_lines.append(f"- k: {payload.get('k')}, silhouette: {payload.get('silhouette'):.3f}\n")
        if payload.get("sub_hits") is not None:
            md_lines.append(f"- sub_hits: {payload.get('sub_hits')}\n")

        clusters = payload["clusters"]
        # stable ordering by cluster id
        for lab in sorted(clusters.keys(), key=lambda x: int(x)):
            cluster_tokens = clusters[lab]
            text = format_cluster(cluster_tokens, TOP_TOKENS_PER_CLUSTER)
            md_lines.append(f"### Cluster {lab}\n")
            md_lines.append(text + "\n")

            rows.append({
                "concept": concept,
                "decade": decade,
                "cluster": int(lab),
                "top_tokens": text,
                "k": payload.get("k"),
                "silhouette": payload.get("silhouette"),
                "concept_hits": payload.get("concept_hits"),
                "json_path": str(json_path),
            })

    # write outputs
    out_md = out_dir / "selected_topics_summary.md"
    out_csv = out_dir / "selected_topics_summary.csv"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_md}")
    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()