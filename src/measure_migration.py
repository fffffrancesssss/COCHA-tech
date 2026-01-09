from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd


def load_selected_topics_csv(root: Path) -> pd.DataFrame:
    # you already generated selected_topics_summary.csv, but here we use the big index + parse json if needed.
    # We will use emergent_topics/<concept>/<decade>_topics.json via index when needed.
    idx = pd.read_csv(root / "results" / "emergent_topics" / "topics_index.csv")
    return idx


def load_cluster_tokens_from_selected(selected_csv: Path) -> pd.DataFrame:
    # If you have selected_topics_summary.csv, it already includes top_tokens per cluster.
    return pd.read_csv(selected_csv)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="results/inheritance/top_edges_concern_contrib.csv")
    ap.add_argument("--selected_topics", default="results/emergent_topics/selected_topics_summary.csv")
    ap.add_argument("--top_edges", type=int, default=3)
    ap.add_argument("--top_concerns_per_edge", type=int, default=2)
    ap.add_argument("--delta_decades", type=int, default=1, help="compare t-Δ to t, in decades steps (1=10yrs)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    contrib = pd.read_csv(root / args.edges)
    sel = pd.read_csv(root / args.selected_topics)

    # pick top edges
    top_edges = (
        contrib.groupby(["from", "to"])["weight"].sum().reset_index()
        .sort_values("weight", ascending=False)
        .head(args.top_edges)
    )

    rows = []
    for _, e in top_edges.iterrows():
        t1, t2 = e["from"], e["to"]
        sub = contrib[(contrib["from"] == t1) & (contrib["to"] == t2)].copy()
        sub = sub.sort_values("weight", ascending=False).head(args.top_concerns_per_edge)

        for _, r in sub.iterrows():
            concern = r["concern"]
            onset_from = int(r["onset_from"])
            onset_to = int(r["onset_to"])
            # choose a comparison decade near onset_to
            target_decade = onset_to
            prev_decade = target_decade - args.delta_decades * 10

            # We will measure "token migration" using the TECH cluster tokens (union across clusters)
            def tech_tokens(concept: str, decade: int) -> set[str]:
                tmp = sel[(sel["concept"] == concept) & (sel["decade"] == decade)]
                if tmp.empty:
                    return set()
                toks = set()
                for s in tmp["top_tokens"].tolist():
                    toks |= set([x.strip() for x in str(s).split(",") if x.strip()])
                return toks

            A_prev = tech_tokens(t1, prev_decade)
            B_now = tech_tokens(t2, target_decade)

            mig = jaccard(A_prev, B_now)

            rows.append({
                "from": t1,
                "to": t2,
                "concern": concern,
                "prev_decade": prev_decade,
                "target_decade": target_decade,
                "onset_from": onset_from,
                "onset_to": onset_to,
                "edge_weight": float(r["weight"]),
                "migration_jaccard": mig,
                "from_tokens_n": len(A_prev),
                "to_tokens_n": len(B_now),
            })

    out_dir = root / "results" / "inheritance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "migration_scores.csv"
    pd.DataFrame(rows).sort_values(["migration_jaccard", "edge_weight"], ascending=False).to_csv(out_path, index=False)
    print("[OK] wrote", out_path)


if __name__ == "__main__":
    main()