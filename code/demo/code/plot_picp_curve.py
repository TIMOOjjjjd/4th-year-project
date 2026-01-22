from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CURVE_FILES = {
    "Baseline": "picp_curve_baseline.csv",
    "GRU": "picp_curve_gru.csv",
    "GNN": "picp_curve_gnn.csv",
}
OUTPUT_PATH = "picp_reliability_diagram.png"


def _load_curve(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"expected_coverage", "observed_coverage"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df


def main() -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    any_plotted = False

    for label, filename in CURVE_FILES.items():
        df = _load_curve(Path(filename))
        if df.empty:
            continue
        df = df.sort_values("expected_coverage")
        ax.plot(
            df["expected_coverage"],
            df["observed_coverage"],
            marker="o",
            linewidth=1.6,
            label=label,
        )
        any_plotted = True

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_title("Reliability Diagram (PICP Curve)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    if any_plotted:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
