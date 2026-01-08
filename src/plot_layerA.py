from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["decade"] = df["decade"].astype(int)
    return df


def filter_for_plot(df: pd.DataFrame, concept: str, min_hits: int = 3) -> pd.DataFrame:
    tmp = df[df["concept"] == concept].copy()

    # global rule: only plot decades with concept_hits >= min_hits
    tmp = tmp[tmp["concept_hits"] >= min_hits]

    # AI starts from 1950s
    if concept == "ai":
        tmp = tmp[tmp["decade"] >= 1950]

    # phone_social requires both sides >= min_hits if these columns exist
    if concept == "phone_social":
        if "smartphone_hits" in tmp.columns and "social_media_hits" in tmp.columns:
            tmp = tmp[(tmp["smartphone_hits"] >= min_hits) & (tmp["social_media_hits"] >= min_hits)]

    return tmp.sort_values("decade")


def plot_metric(
    df: pd.DataFrame,
    concepts: list[str],
    ycol: str,
    title: str,
    out_path: Path,
    min_hits: int = 3,
):
    plt.figure()
    for c in concepts:
        tmp = filter_for_plot(df, c, min_hits=min_hits)
        plt.plot(tmp["decade"], tmp[ycol], marker="o", label=c)

    plt.axhline(0, linewidth=1)
    plt.xlabel("Decade")
    plt.ylabel(ycol.replace("_", " "))
    plt.title(title)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "results" / "tech_trends_layerA.csv"
    out_dir = root / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(csv_path)

    concepts = ["nuclear", "gmo", "phone_social", "ai"]

    # 1) sentiment
    plot_metric(
        df=df,
        concepts=concepts,
        ycol="sentiment_delta",
        title="Tech sentiment shift (Δ vs baseline; filtered for plotting)",
        out_path=out_dir / "sentiment_delta.png",
        min_hits=3,
    )

    # 2) concerns
    concerns = ["unemployment", "health", "privacy", "childhood", "information", "tech_companies"]
    for k in concerns:
        plot_metric(
            df=df,
            concepts=concepts,
            ycol=f"{k}_delta",
            title=f"Concern shift: {k} (Δ vs baseline; filtered for plotting)",
            out_path=out_dir / f"{k}_delta.png",
            min_hits=3,
        )

    print(f"[OK] wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()