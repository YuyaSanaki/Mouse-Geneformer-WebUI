"""
ISP post-stats figures and tables (logic from isp_analysis.ipynb).

Writes PNGs to a figures directory and CSV summaries to the stats directory.
Uses pandas + matplotlib; optional cuDF is not required for this script.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def run_isp_figure_analysis(
    parquet_path: str | Path,
    figures_dir: str | Path,
    stats_dir: str | Path,
    *,
    label_start: str = "start",
    label_end: str = "goal",
) -> None:
    parquet_path = Path(parquet_path)
    figures_dir = Path(figures_dir)
    stats_dir = Path(stats_dir)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150

    df = pd.read_parquet(parquet_path)
    print(f"[isp_analysis] Loaded {len(df)} genes from {parquet_path}")
    if "Sig" in df.columns:
        print(f"[isp_analysis] Significant (Sig=1): {int(df['Sig'].sum())}")

    goal_label = f"{label_start} → {label_end}"

    # --- Summary to stdout ---
    print("=" * 60)
    print("Summary Statistics for Shift_to_goal_end")
    print("=" * 60)
    print(df["Shift_to_goal_end"].describe())
    if "Goal_end_FDR" in df.columns:
        print(f"\nGenes with FDR < 0.05: {(df['Goal_end_FDR'] < 0.05).sum()}")
        print(f"Genes with FDR < 0.10: {(df['Goal_end_FDR'] < 0.10).sum()}")

    # --- 1. Shift + p-value distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(df["Shift_to_goal_end"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero shift")
    axes[0].set_xlabel(f"Shift to goal ({label_end})")
    axes[0].set_ylabel("Number of genes")
    axes[0].set_title("Distribution of cosine similarity shifts")
    axes[0].legend()
    axes[1].hist(df["Goal_end_vs_random_pval"], bins=50, color="coral", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0.05, color="red", linestyle="--", linewidth=1.5, label="p=0.05")
    axes[1].set_xlabel("P-value (vs random)")
    axes[1].set_ylabel("Number of genes")
    axes[1].set_title("Distribution of p-values")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "shift_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # --- 2. Volcano ---
    df = df.copy()
    df["neg_log10_fdr"] = -np.log10(df["Goal_end_FDR"].clip(lower=1e-300))
    df["neg_log10_pval"] = -np.log10(df["Goal_end_vs_random_pval"].clip(lower=1e-300))
    colors = np.where(
        df["Goal_end_FDR"] < 0.05,
        "red",
        np.where(df["Goal_end_vs_random_pval"] < 0.05, "orange", "grey"),
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df["Shift_to_goal_end"], df["neg_log10_pval"], c=colors, alpha=0.5, s=20, edgecolors="none")
    n_label = 15
    top_genes = pd.concat(
        [df.nlargest(n_label, "Shift_to_goal_end"), df.nsmallest(n_label, "Shift_to_goal_end")]
    ).drop_duplicates()
    for _, row in top_genes.iterrows():
        ax.annotate(
            row["Gene_name"],
            (row["Shift_to_goal_end"], row["neg_log10_pval"]),
            fontsize=7,
            alpha=0.8,
            textcoords="offset points",
            xytext=(5, 3),
        )
    ax.axhline(y=-np.log10(0.05), color="blue", linestyle="--", alpha=0.5, label="p=0.05")
    ax.axvline(x=0, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Shift to goal end", fontsize=13)
    ax.set_ylabel("-log10(p-value)", fontsize=13)
    ax.set_title(f"Volcano plot: {goal_label}", fontsize=14)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="FDR < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=8, label="p < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=8, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    plt.tight_layout()
    fig.savefig(figures_dir / "volcano_plot.png", bbox_inches="tight")
    plt.close(fig)

    # --- 3. Top genes bar ---
    n_top = 25
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    top_pos = df.nlargest(n_top, "Shift_to_goal_end")
    colors_pos = ["red" if s == 1 else "steelblue" for s in top_pos["Sig"]]
    axes[0].barh(range(n_top), top_pos["Shift_to_goal_end"].values, color=colors_pos, edgecolor="white")
    axes[0].set_yticks(range(n_top))
    axes[0].set_yticklabels(top_pos["Gene_name"].values, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Shift to goal end")
    axes[0].set_title(f"Top {n_top} toward goal ({label_end})")
    top_neg = df.nsmallest(n_top, "Shift_to_goal_end")
    colors_neg = ["red" if s == 1 else "coral" for s in top_neg["Sig"]]
    axes[1].barh(range(n_top), top_neg["Shift_to_goal_end"].values, color=colors_neg, edgecolor="white")
    axes[1].set_yticks(range(n_top))
    axes[1].set_yticklabels(top_neg["Gene_name"].values, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Shift to goal end")
    axes[1].set_title(f"Top {n_top} away from goal ({label_end})")
    plt.suptitle("Red = significant (FDR)", fontsize=10, y=0.02, color="red")
    plt.tight_layout()
    fig.savefig(figures_dir / "top_genes_barplot.png", bbox_inches="tight")
    plt.close(fig)

    # --- 4. Waterfall ---
    df_sorted = df.sort_values("Shift_to_goal_end", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors_wf = ["steelblue" if v >= 0 else "coral" for v in df_sorted["Shift_to_goal_end"]]
    ax.bar(range(len(df_sorted)), df_sorted["Shift_to_goal_end"], color=colors_wf, width=1.0, edgecolor="none")
    ax.set_xlabel("Gene rank", fontsize=13)
    ax.set_ylabel("Shift to goal end", fontsize=13)
    ax.set_title(f"Waterfall (all genes ranked): {goal_label}", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5)
    top1 = df_sorted.iloc[0]
    bottom1 = df_sorted.iloc[-1]
    ax.annotate(
        top1["Gene_name"],
        (0, top1["Shift_to_goal_end"]),
        fontsize=9,
        fontweight="bold",
        xytext=(30, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.annotate(
        bottom1["Gene_name"],
        (len(df_sorted) - 1, bottom1["Shift_to_goal_end"]),
        fontsize=9,
        fontweight="bold",
        xytext=(-80, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    plt.tight_layout()
    fig.savefig(figures_dir / "waterfall_plot.png", bbox_inches="tight")
    plt.close(fig)

    # --- 5. CSV exports (same as notebook) ---
    top100_pos = df.nlargest(100, "Shift_to_goal_end")[
        ["Gene_name", "Ensembl_ID", "Shift_to_goal_end", "Goal_end_vs_random_pval", "Goal_end_FDR", "Sig"]
    ]
    top100_neg = df.nsmallest(100, "Shift_to_goal_end")[
        ["Gene_name", "Ensembl_ID", "Shift_to_goal_end", "Goal_end_vs_random_pval", "Goal_end_FDR", "Sig"]
    ]
    top100_pos.to_csv(stats_dir / "top100_positive_shifters.csv", index=False)
    top100_neg.to_csv(stats_dir / "top100_negative_shifters.csv", index=False)
    sig_genes = df[df["Sig"] == 1].sort_values("Shift_to_goal_end", ascending=False)
    sig_genes.to_csv(stats_dir / "significant_genes.csv", index=False)
    nominal_sig = df[df["Goal_end_vs_random_pval"] < 0.05].sort_values("Shift_to_goal_end", ascending=False)
    nominal_sig.to_csv(stats_dir / "nominal_significant_genes_p0.05.csv", index=False)

    print(f"[isp_analysis] Figures saved under: {figures_dir}")
    print(f"[isp_analysis] Tables saved under:   {stats_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="ISP parquet → figures + CSV tables.")
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--figures-dir", type=Path, required=True)
    p.add_argument("--stats-dir", type=Path, required=True, help="Directory for CSV exports (usually ispstats_results).")
    p.add_argument("--label-start", default="start")
    p.add_argument("--label-end", default="goal")
    args = p.parse_args()
    run_isp_figure_analysis(
        args.parquet,
        args.figures_dir,
        args.stats_dir,
        label_start=args.label_start,
        label_end=args.label_end,
    )


if __name__ == "__main__":
    main()
