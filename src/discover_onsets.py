from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def infer_concerns(df: pd.DataFrame) -> list[str]:
    # columns like privacy_delta, health_delta...
    return sorted({c[:-6] for c in df.columns if c.endswith("_delta") and c not in ("sentiment_delta",)})


def onset_time(
    ts,                    # DataFrame: decade, A, concept_hits
    tau: float,
    min_hits: int,
    window: int,
    min_keep: int,
    earliest_decade: int,  # 新增
):
    ts = ts.sort_values("decade").reset_index(drop=True)

    # A: 先跳过技术出现前的 decade
    ts = ts[ts["decade"] >= earliest_decade].reset_index(drop=True)
    if ts.empty:
        return None

    decades = ts["decade"].to_list()

    for i in range(1, len(ts)):   # 注意从 1 开始，才能用到 t-10
        t = ts.loc[i, "decade"]
        prev = ts.loc[i-1, "decade"]

        # 只支持 decade 步长 10 的情况
        if t - prev != 10:
            continue

        if ts.loc[i, "concept_hits"] < min_hits:
            continue

        A_prev = ts.loc[i-1, "A"]
        A_now  = ts.loc[i, "A"]

        # B: crossing 条件
        if not (A_prev <= tau and A_now > tau):
            continue

        # window/min_keep：从 i 开始往后 window 个 decade，A>tau 且 hits>=min_hits 计数
        end = min(len(ts), i + window)
        keep = 0
        for j in range(i, end):
            if ts.loc[j, "concept_hits"] >= min_hits and ts.loc[j, "A"] > tau:
                keep += 1

        if keep >= min_keep:
            return int(t)

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/tech_trends_layerA.csv")
    ap.add_argument("--baseline", type=int, default=None, help="baseline decade, for labeling only (deltas are already baseline-adjusted)")
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--min_keep", type=int, default=2)
    ap.add_argument("--top_edges", type=int, default=10)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / args.csv)
    df["decade"] = df["decade"].astype(int)

    concerns = infer_concerns(df)
    techs = sorted(df["concept"].unique().tolist())

    # onset matrix: rows=concern, cols=tech
    onset_records = []
    for c in concerns:
        col = f"{c}_delta"
        for t in techs:
            ts = df[df["concept"] == t][["decade", "concept_hits", col]].dropna(subset=["decade"])
            if ts.empty or col not in ts.columns:
                onset = np.nan
            else:
                onset = onset_time(
                    ts,
                    value_col=col,
                    min_hits_col="concept_hits",
                    min_hits=args.min_hits,
                    tau=args.tau,
                    window=args.window,
                    min_keep=args.min_keep,
                )
            onset_records.append({"concern": c, "tech": t, "onset_decade": onset})

    onsets = pd.DataFrame(onset_records)
    onsets_wide = onsets.pivot(index="concern", columns="tech", values="onset_decade").reset_index()

    out_dir = root / "results" / "inheritance"
    out_dir.mkdir(parents=True, exist_ok=True)

    onsets_path = out_dir / "onsets_wide.csv"
    onsets_wide.to_csv(onsets_path, index=False)

    # concern clustering by onset pattern (simple: k-means on normalized ranks)
    # Convert decades to ranks within each concern, NA -> large rank
    mat = onsets_wide.drop(columns=["concern"]).to_numpy(dtype=float)
    # fill NA with +inf then rank; later replace inf ranks with max+1
    mat_filled = np.where(np.isfinite(mat), mat, np.inf)

    # rank across tech dimension for each concern
    ranks = np.zeros_like(mat_filled, dtype=float)
    for i in range(mat_filled.shape[0]):
        row = mat_filled[i]
        finite = np.isfinite(row)
        if finite.sum() == 0:
            ranks[i] = np.nan
            continue
        # smaller decade => smaller rank
        order = np.argsort(row)
        r = np.empty_like(row, dtype=float)
        r[order] = np.arange(len(row))
        # push inf to the end
        inf_mask = ~finite
        if inf_mask.any():
            r[inf_mask] = finite.sum() + 1
        # normalize
        ranks[i] = r / (np.nanmax(r) if np.nanmax(r) > 0 else 1.0)

    # Choose k automatically (2..4) by silhouette if enough concerns
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = ranks
    # drop all-nan
    keep = np.isfinite(X).any(axis=1)
    Xk = X[keep]
    concerns_kept = onsets_wide.loc[keep, "concern"].tolist()

    best = None
    if len(concerns_kept) >= 4:
        for k in [2, 3, 4]:
            if k >= len(concerns_kept):
                continue
            km = KMeans(n_clusters=k, random_state=0, n_init="auto")
            labels = km.fit_predict(Xk)
            if len(set(labels)) < 2:
                continue
            s = silhouette_score(Xk, labels)
            if best is None or s > best["sil"]:
                best = {"k": k, "sil": float(s), "labels": labels.tolist()}
    else:
        best = {"k": 1, "sil": np.nan, "labels": [0] * len(concerns_kept)}

    cluster_df = pd.DataFrame({"concern": concerns_kept, "cluster": best["labels"]})
    cluster_path = out_dir / "concern_clusters.csv"
    cluster_df.to_csv(cluster_path, index=False)

    # inheritance edges (pattern discovery): for each pair tech1->tech2 sum exp(-dt/lambda) over concerns
    lambda_decades = 2.0  # 20 years scale; tune later
    edges = []
    tech_cols = [c for c in onsets_wide.columns if c != "concern"]
    for t1 in tech_cols:
        for t2 in tech_cols:
            if t1 == t2:
                continue
            score = 0.0
            contrib = []
            for _, row in onsets_wide.iterrows():
                c = row["concern"]
                a = row[t1]
                b = row[t2]
                if not np.isfinite(a) or not np.isfinite(b):
                    continue
                if a < b:
                    dt = (b - a) / 10.0  # decades diff in decades units
                    w = float(np.exp(-dt / lambda_decades))
                    score += w
                    contrib.append((c, w, int(a), int(b)))
            if score > 0:
                edges.append({"from": t1, "to": t2, "score": score, "n_concerns": len(contrib)})

    edges_df = pd.DataFrame(edges).sort_values("score", ascending=False)
    edges_path = out_dir / "inheritance_edges.csv"
    edges_df.to_csv(edges_path, index=False)

    # Also output top-edge concern contributions for later deep dive
    top = edges_df.head(args.top_edges)[["from", "to"]]
    contrib_rows = []
    for _, e in top.iterrows():
        t1, t2 = e["from"], e["to"]
        for _, row in onsets_wide.iterrows():
            c = row["concern"]
            a = row[t1]
            b = row[t2]
            if not np.isfinite(a) or not np.isfinite(b):
                continue
            if a < b:
                dt = (b - a) / 10.0
                w = float(np.exp(-dt / lambda_decades))
                contrib_rows.append({"from": t1, "to": t2, "concern": c, "weight": w, "onset_from": int(a), "onset_to": int(b)})

    contrib_df = pd.DataFrame(contrib_rows).sort_values(["from", "to", "weight"], ascending=[True, True, False])
    contrib_path = out_dir / "top_edges_concern_contrib.csv"
    contrib_df.to_csv(contrib_path, index=False)

    print("[OK] wrote:")
    print(" -", onsets_path)
    print(" -", cluster_path, f"(k={best['k']}, silhouette={best['sil']})")
    print(" -", edges_path)
    print(" -", contrib_path)


if __name__ == "__main__":
    main()