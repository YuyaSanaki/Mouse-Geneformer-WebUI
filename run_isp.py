"""
In-silico perturbation driver (notebook logic as a script).

Configuration: YAML file (default /app/config/isp.yaml). Override path with --config or ISP_CONFIG.
CLI --forward-batch-size / --nproc override the YAML runtime section when passed.
Outputs: with paths.output_root, writes to {output_root}/{YYYYMMDD}/isp_results and .../ispstats_results (date: --output-date, ISP_OUTPUT_DATE, or today).
Also writes {output_root}/{YYYYMMDD}/isp_run.log (rotating; env ISP_LOG_MAX_BYTES, ISP_LOG_BACKUP_COUNT; ISP_DISABLE_RUN_LOG=1 to skip) mirroring stdout/stderr on the main process only.

Speed on DGX / large GPUs:
  - Imports apply TF32-friendly matmul settings (see in_silico_perturber._configure_cuda_performance).
  - Tune forward_batch_size in config until just below OOM.
  - Optional: GENEFORMER_TORCH_COMPILE=1 for torch.compile (PyTorch 2+).
  - Multi-GPU: accelerate launch --num_processes <ngpu> run_isp.py [--config ...]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from accelerate import Accelerator
from geneformer import InSilicoPerturber, InSilicoPerturberStats

from run_pipeline_log import format_isp_run_banner, install_rotating_stdio_tee


def _deep_get(m: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = m
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _ensure_dir_suffix(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def load_isp_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"ISP config not found: {path}. Create config/isp.yaml or pass --config."
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"ISP config must be a mapping at root: {path}")
    return data


def _write_run_provenance(
    run_root: Path,
    config_path: Path,
    forward_batch_size: int,
    nproc: int,
    cli_forward_batch_size: int | None,
    cli_nproc: int | None,
) -> None:
    """Copy the ISP YAML and record metadata next to dated outputs (reproducibility)."""
    run_root.mkdir(parents=True, exist_ok=True)
    dest_cfg = run_root / "isp_config_used.yaml"
    shutil.copy2(config_path, dest_cfg)

    meta: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source_path": str(config_path.resolve()),
        "effective_runtime": {
            "forward_batch_size": forward_batch_size,
            "nproc": nproc,
        },
        "cli_overrides": {
            k: v
            for k, v in (
                ("forward_batch_size", cli_forward_batch_size),
                ("nproc", cli_nproc),
            )
            if v is not None
        },
    }
    git_sha = os.environ.get("ISP_GIT_COMMIT", "").strip()
    if not git_sha:
        try:
            rev = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if rev.returncode == 0:
                git_sha = rev.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    if git_sha:
        meta["git_commit"] = git_sha

    with open(run_root / "isp_run_metadata.yaml", "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    accelerator = Accelerator()
    default_cfg = os.environ.get("ISP_CONFIG", "/app/config/isp.yaml")
    p = argparse.ArgumentParser(description="Run Geneformer in-silico perturbation + stats.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(default_cfg),
        help="YAML config path (default: ISP_CONFIG or /app/config/isp.yaml).",
    )
    p.add_argument(
        "--forward-batch-size",
        type=int,
        default=None,
        help="Override config runtime.forward_batch_size.",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Override config runtime.nproc.",
    )
    p.add_argument(
        "--output-date",
        default=None,
        metavar="YYYYMMDD",
        help="Date folder under output_root (default: today, or ISP_OUTPUT_DATE).",
    )
    p.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip isp_analysis.py figures/tables after stats.",
    )
    args = p.parse_args()

    cfg = load_isp_config(args.config)

    forward_batch_size = args.forward_batch_size
    if forward_batch_size is None:
        forward_batch_size = int(
            _deep_get(cfg, "runtime", "forward_batch_size", default=os.environ.get("ISP_FORWARD_BATCH_SIZE", "128"))
        )
    nproc = args.nproc
    if nproc is None:
        nproc = int(_deep_get(cfg, "runtime", "nproc", default=os.environ.get("ISP_NPROC", "8")))

    paths = cfg.get("paths") or {}
    dataset_name = paths.get("dataset")
    model_dir = paths.get("geneformer_model")
    output_root = paths.get("output_root")
    isp_out: str | None
    stats_out: str | None
    date_used: str | None = None
    if output_root:
        date_used = (
            (args.output_date or os.environ.get("ISP_OUTPUT_DATE") or datetime.now().strftime("%Y%m%d")).strip()
        )
        if len(date_used) != 8 or not date_used.isdigit():
            raise ValueError(
                "output date must be YYYYMMDD (8 digits). "
                "Set --output-date, ISP_OUTPUT_DATE, or use default today."
            )
        isp_out = os.path.join(output_root, date_used, "isp_results")
        stats_out = os.path.join(output_root, date_used, "ispstats_results")
    else:
        isp_out = paths.get("isp_results_dir")
        stats_out = paths.get("ispstats_results_dir")
    for key, val in (
        ("paths.dataset", dataset_name),
        ("paths.geneformer_model", model_dir),
        ("paths output (isp or legacy)", isp_out),
        ("paths output (stats or legacy)", stats_out),
    ):
        if not val:
            raise ValueError(
                f"Missing required config key: {key}. "
                "Set paths.output_root (recommended) or paths.isp_results_dir and paths.ispstats_results_dir."
            )

    pert = cfg.get("perturbation") or {}
    select_perturb_type = pert.get("type", "delete")
    start_state = pert.get("start_state")
    end_state = pert.get("end_state")
    if start_state is None or end_state is None:
        raise ValueError("config perturbation.start_state and perturbation.end_state are required.")
    alt_state = list(pert.get("alt_states") or [])
    organ_data = pert.get("organ_data", "experiment")
    genes_to_perturb_list = list(pert.get("genes_to_perturb") or [])
    state_key = pert.get("state_key", "disease")

    mdl = cfg.get("model") or {}
    use_model_type = mdl.get("type", "Pretrained")
    num_classes = int(mdl.get("num_classes", 0))

    isp_cfg = cfg.get("isp") or {}
    max_ncells = isp_cfg.get("max_ncells")
    emb_layer = int(isp_cfg.get("emb_layer", -1))
    emb_mode = isp_cfg.get("emb_mode", "cell")
    cell_emb_style = isp_cfg.get("cell_emb_style", "mean_pool")
    combos = int(isp_cfg.get("combos", 0))
    anchor_gene = isp_cfg.get("anchor_gene")
    perturb_rank_shift = isp_cfg.get("perturb_rank_shift")
    perturb_rank_direct_shift = isp_cfg.get("perturb_rank_direct_shift")
    filter_data = isp_cfg.get("filter_data")

    st = cfg.get("stats") or {}
    stats_mode = st.get("mode", "goal_state_shift")

    start_state_fn = start_state.replace(" ", "-")
    end_state_fn = end_state.replace(" ", "-")
    output_prefix = "output_in-silico_SE{}_OR{}_ST{}_EN{}".format(
        select_perturb_type, organ_data, start_state_fn, end_state_fn
    )

    isp_dir = _ensure_dir_suffix(str(isp_out))
    os.makedirs(isp_dir, exist_ok=True)

    if output_root and date_used:
        log_dir = Path(output_root) / date_used
    else:
        log_dir = Path(isp_out.rstrip(os.sep)).parent
    log_path = log_dir / "isp_run.log"

    if accelerator.is_main_process:
        install_rotating_stdio_tee(log_path, env_prefix="ISP")
        banner = format_isp_run_banner(
            cfg,
            extras={
                "config_path": str(args.config.resolve()),
                "date_used": date_used or "",
                "isp_out": isp_out,
                "stats_out": stats_out,
                "output_prefix": output_prefix,
                "forward_batch_size": forward_batch_size,
                "nproc": nproc,
                "skip_analysis": args.skip_analysis,
            },
        )
        print(banner, end="")
        print(f"  run log (rotating): {log_path}")

    print("Checking CUDA...", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    isp = InSilicoPerturber(
        perturb_type=select_perturb_type,
        perturb_rank_shift=perturb_rank_shift,
        perturb_rank_direct_shift=perturb_rank_direct_shift,
        genes_to_perturb="all" if len(genes_to_perturb_list) == 0 else genes_to_perturb_list,
        combos=combos,
        anchor_gene=anchor_gene,
        model_type=use_model_type,
        num_classes=num_classes,
        emb_mode=emb_mode,
        cell_emb_style=cell_emb_style,
        filter_data=filter_data,
        cell_states_to_model={
            "state_key": state_key,
            "start_state": start_state,
            "goal_state": end_state,
            "alt_states": alt_state,
        },
        max_ncells=max_ncells,
        emb_layer=emb_layer,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
    )

    if output_root and date_used:
        print(f"Output root: {output_root}  (date {date_used})")
        print(f"  → ISP results: {isp_out}")
        print(f"  → ISP stats:   {stats_out}")

    if output_root and date_used and accelerator.is_main_process:
        _write_run_provenance(
            Path(output_root) / date_used,
            args.config.resolve(),
            forward_batch_size,
            nproc,
            args.forward_batch_size,
            args.nproc,
        )
        print(
            f"  → Provenance: {Path(output_root) / date_used / 'isp_config_used.yaml'} "
            f"+ isp_run_metadata.yaml"
        )

    print("Starting perturbation...")
    isp.perturb_data(model_dir, dataset_name, isp_dir, output_prefix)

    # Wait for all processes to finish perturbing
    accelerator.wait_for_everyone()

    # Only run analysis on the main process to avoid race conditions
    if accelerator.is_main_process:
        print("Perturbation complete. Generating stats...")
        ispstats = InSilicoPerturberStats(
            mode=stats_mode,
            genes_perturbed="all" if len(genes_to_perturb_list) == 0 else genes_to_perturb_list,
            combos=combos,
            anchor_gene=anchor_gene,
            cell_states_to_model={
                "state_key": state_key,
                "start_state": start_state,
                "goal_state": end_state,
                "alt_states": alt_state,
            },
        )

        stats_dir = _ensure_dir_suffix(str(stats_out))
        os.makedirs(stats_dir, exist_ok=True)

        ispstats.get_stats(isp_dir, None, stats_dir, output_prefix)
        print("Stats generation complete. Check the parquet file.")

        analysis_cfg = cfg.get("analysis") or {}
        run_figures = analysis_cfg.get("enabled", True) and not args.skip_analysis
        if run_figures:
            from isp_analysis import run_isp_figure_analysis

            parquet_file = os.path.join(stats_dir.rstrip(os.sep), f"{output_prefix}.parquet")
            stats_parent = Path(stats_dir.rstrip(os.sep))
            figures_dir = stats_parent.parent / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            print("Running ISP figure analysis (isp_analysis.py)...")
            run_isp_figure_analysis(
                parquet_file,
                figures_dir,
                stats_parent,
                label_start=start_state,
                label_end=end_state,
            )
            print(f"Figures directory: {figures_dir}")


if __name__ == "__main__":
    main()
