#!/usr/bin/env python3
"""Annotate an existing ISP UMAP run with marker-gene cell-type predictions.

Reconstructs the same start-state subsample used by the run (via config or
explicit sampling args), scores marker tokens in each cell's input_ids, and
writes pred_cell_type / coarse_type onto per_cell_isp_shift.csv (and the
cluster_coexpr_analysis copy when present).

Example:
  docker compose run --rm --no-deps \\
    -v /path/to/AsanoPIPand3and5w:/app/data/AsanoPIPand3and5w \\
    webui python3 /app/scripts/annotate_isp_umap_celltypes.py \\
      --run-dir output/20260807/isp_umap_074258 \\
      --config /app/output/_rerun_configs/isp_umap_igfbp2_rerun.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_from_disk

_SCRIPTS = Path(__file__).resolve().parent
_CORE = _SCRIPTS.parent / "core"
for p in (_CORE, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from isp_umap_celltype import annotate_dataframe_with_cell_types  # noqa: E402
from run_isp_umap import subsample_dataset  # noqa: E402


def _resolve_start_input_ids(cfg: dict, n_expected: int, sample_ids: list[str] | None):
    dataset = load_from_disk(cfg["paths"]["dataset"])
    state_key = cfg["perturbation"]["state_key"]
    start_state = cfg["perturbation"]["start_state"]
    umap_cfg = cfg.get("umap", {})
    max_cells = umap_cfg.get("max_cells_per_state", 2000)
    method = umap_cfg.get("sampling", "head")
    sample_key = umap_cfg.get("sample_key", "sample_id")
    seed = umap_cfg.get("seed", 42)

    start = dataset.filter(lambda x: x.get(state_key) == start_state)
    start = subsample_dataset(start, max_cells, method, sample_key, seed, state_label=start_state)
    if len(start) != n_expected:
        raise ValueError(
            f"Reconstructed start-state size {len(start)} != CSV rows {n_expected}. "
            "Pass a config that matches the original run sampling."
        )
    if sample_ids is not None and sample_key in start.column_names:
        recon = [str(x) for x in start[sample_key]]
        if recon != sample_ids:
            raise ValueError(
                "Reconstructed sample_id order does not match the CSV; "
                "refusing to annotate (wrong config/sampling/seed)."
            )
    return start["input_ids"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="ISP UMAP YAML used for the run (needed to reconstruct the start-state subsample)",
    )
    parser.add_argument(
        "--refresh-overlays",
        action="store_true",
        help="Also regenerate joint overlays + L2-by-group plots after annotating",
    )
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else Path.cwd() / args.run_dir
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    per_cell_path = run_dir / "per_cell_isp_shift.csv"
    if not per_cell_path.exists():
        raise FileNotFoundError(per_cell_path)
    df = pd.read_csv(per_cell_path)
    sample_ids = df["sample_id"].astype(str).tolist() if "sample_id" in df.columns else None
    input_ids = _resolve_start_input_ids(cfg, len(df), sample_ids)

    annotated = annotate_dataframe_with_cell_types(df, input_ids)

    cluster_csv = run_dir / "cluster_coexpr_analysis" / "per_cell_cluster_l2_celltype.csv"
    if cluster_csv.exists():
        old = pd.read_csv(cluster_csv)
        if len(old) == len(annotated):
            for col in ("cluster", "umap1_joint", "umap2_joint", "umap1_isp", "umap2_isp", "umap1", "umap2"):
                if col in old.columns and col not in annotated.columns:
                    annotated[col] = old[col].values
            annotated.to_csv(cluster_csv, index=False)
            print(f"Updated {cluster_csv}", flush=True)
        else:
            print(f"Skipping {cluster_csv} (row count mismatch)", flush=True)

    annotated.to_csv(per_cell_path, index=False)
    print(f"Updated {per_cell_path}", flush=True)
    print("pred_cell_type:", annotated["pred_cell_type"].value_counts().to_dict(), flush=True)
    print("coarse_type:", annotated["coarse_type"].value_counts().to_dict(), flush=True)

    if args.refresh_overlays:
        from run_isp_umap import run_downstream_plots

        gene = cfg.get("perturbation", {}).get("gene_to_perturb")
        isp = list(run_dir.glob("*_ISP_*_embs.npy"))
        if isp:
            name = isp[0].name
            gene = name.split("_ISP_")[1].replace("_embs.npy", "")
        run_downstream_plots(run_dir, gene, cfg)

if __name__ == "__main__":
    main()
