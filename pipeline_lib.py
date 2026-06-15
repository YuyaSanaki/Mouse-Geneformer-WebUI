"""
Shared helpers for the Tokenize → Fine-tune → ISP end-to-end pipeline.

Resolves paths from a minimal pipeline config (chiefly data.input_dir) and builds
per-stage YAML configs by merging defaults from config/*.yaml with overrides.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from data_input_layout import resolve_single_cell_input_dir
from geneformer.gene_ids import resolve_genes_to_perturb

ROOT = Path(__file__).resolve().parent
DEFAULT_PRETRAINED = "/app/models/mouse-Geneformer/"
DEFAULT_OUTPUT_ROOT = "/app/output"


def pipeline_runtime(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    """Shared runtime knobs from config/pipeline.yaml (applied to all stages)."""
    raw = pipeline.get("runtime")
    return dict(raw) if isinstance(raw, dict) else {}


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge override into a copy of base (override wins)."""
    out: dict[str, Any] = copy.deepcopy(dict(base))
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def study_name_from_input_dir(input_dir: str) -> str:
    name = Path(input_dir.rstrip("/")).name
    if not name:
        raise ValueError(f"Cannot derive study name from input_dir: {input_dir!r}")
    safe = re.sub(r"[^\w.-]+", "_", name).strip("_")
    if not safe:
        raise ValueError(f"Invalid study name derived from input_dir: {input_dir!r}")
    return safe


def resolve_pipeline_paths(pipeline: Mapping[str, Any], run_dir: Path) -> dict[str, str]:
    """
    Compute all chained paths for one pipeline run.

    run_dir: e.g. /app/output/20260525/pipeline_120000_abc123/
    """
    data = pipeline.get("data") or {}
    paths_cfg = pipeline.get("paths") or {}

    input_dir = str(data.get("input_dir", "")).strip()
    if not input_dir:
        raise ValueError("pipeline data.input_dir is required.")

    study_root = resolve_single_cell_input_dir(input_dir)
    input_path = Path(study_root)
    prefix = data.get("output_prefix")
    study = str(prefix).strip() if prefix else study_name_from_input_dir(str(study_root))

    tokenized_dir = run_dir / "tokenized_dataset"
    loom_temp_dir = run_dir / "loom_files"
    dataset_path = tokenized_dir / f"{study}_0.dataset"
    finetune_run_dir = run_dir / "finetune"
    finetune_model_dir = finetune_run_dir / "all_run1"

    return {
        "input_dir": str(input_path),
        "study_name": study,
        "loom_temp_dir": str(loom_temp_dir),
        "tokenized_dir": str(tokenized_dir),
        "dataset_path": str(dataset_path),
        "pretrained_model": str(paths_cfg.get("pretrained_model") or DEFAULT_PRETRAINED),
        "output_root": str(paths_cfg.get("output_root") or DEFAULT_OUTPUT_ROOT),
        "pipeline_run_dir": str(run_dir),
        "finetune_run_dir": str(finetune_run_dir),
        "finetune_model_dir": str(finetune_model_dir),
    }


def build_tokenize_config(
    pipeline: Mapping[str, Any],
    resolved: Mapping[str, str],
    template: Mapping[str, Any],
) -> dict[str, Any]:
    data = pipeline.get("data") or {}
    overrides = (pipeline.get("stages") or {}).get("tokenize") or {}
    cfg = deep_merge(template, overrides)

    cfg.setdefault("data", {})
    cfg["data"].update(
        {
            "input_type": data.get("input_type", cfg["data"].get("input_type", "single-cell")),
            "input_dir": resolved["input_dir"],
            "loom_temp_dir": resolved["loom_temp_dir"],
            "output_dir": resolved["tokenized_dir"] + "/",
            "output_prefix": resolved["study_name"],
        }
    )
    rt = pipeline_runtime(pipeline)
    cfg.setdefault("tokenizer", {})
    if "nproc" in rt:
        cfg["tokenizer"]["nproc"] = int(rt["nproc"])
    if "max_cells" in rt:
        cfg["tokenizer"]["max_cells"] = int(rt["max_cells"])
    return cfg


def build_finetune_config(
    pipeline: Mapping[str, Any],
    resolved: Mapping[str, str],
    template: Mapping[str, Any],
) -> dict[str, Any]:
    overrides = (pipeline.get("stages") or {}).get("finetune") or {}
    cfg = deep_merge(template, overrides)

    cfg.setdefault("paths", {})
    cfg["paths"].update(
        {
            "dataset": resolved["dataset_path"],
            "geneformer_model": resolved["pretrained_model"],
            "output_root": resolved["finetune_run_dir"],
            "output_time_subdir": False,
            "run_dir": resolved["finetune_run_dir"],
        }
    )
    rt = pipeline_runtime(pipeline)
    if "nproc" in rt:
        cfg.setdefault("runtime", {})
        cfg["runtime"]["nproc"] = int(rt["nproc"])
    return cfg


def build_isp_config(
    pipeline: Mapping[str, Any],
    resolved: Mapping[str, str],
    template: Mapping[str, Any],
    *,
    finetune_model_dir: str,
    num_classes: int,
) -> dict[str, Any]:
    overrides = (pipeline.get("stages") or {}).get("isp") or {}
    pert_defaults = pipeline.get("perturbation") or {}
    cfg = deep_merge(template, overrides)

    cfg.setdefault("paths", {})
    cfg["paths"].update(
        {
            "dataset": resolved["dataset_path"],
            "geneformer_model": finetune_model_dir,
            "output_root": resolved["pipeline_run_dir"],
            "output_time_subdir": False,
            "output_date_subdir": False,
        }
    )
    cfg.setdefault("perturbation", {})
    for key in ("type", "organ_data", "state_key", "start_state", "end_state", "alt_states", "genes_to_perturb"):
        if key in pert_defaults and pert_defaults[key] is not None:
            val = pert_defaults[key]
            if key == "genes_to_perturb":
                val = resolve_genes_to_perturb(list(val))
            cfg["perturbation"][key] = val

    cfg.setdefault("model", {})
    cfg["model"]["type"] = cfg["model"].get("type", "CellClassifier")
    cfg["model"]["num_classes"] = num_classes
    rt = pipeline_runtime(pipeline)
    if rt:
        cfg.setdefault("runtime", {})
        if "nproc" in rt:
            cfg["runtime"]["nproc"] = int(rt["nproc"])
        if "forward_batch_size" in rt:
            cfg["runtime"]["forward_batch_size"] = int(rt["forward_batch_size"])
    return cfg


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(dict(data), f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def format_pipeline_banner(
    pipeline_path: Path,
    resolved: Mapping[str, str],
    stage_config_paths: Mapping[str, Path],
) -> str:
    lines = [
        "=" * 72,
        "End-to-end pipeline: Tokenize → Fine-tune → ISP",
        "=" * 72,
        f"  pipeline config:    {pipeline_path}",
        f"  run directory:      {resolved['pipeline_run_dir']}",
        f"  input_dir:          {resolved['input_dir']}",
        f"  study / prefix:     {resolved['study_name']}",
        f"  tokenized dataset:  {resolved['dataset_path']}",
        f"  finetune model:     {resolved['finetune_model_dir']}",
        "",
        "  stage configs:",
    ]
    for name, p in stage_config_paths.items():
        lines.append(f"    {name}: {p}")
    lines.append("=" * 72)
    return "\n".join(lines) + "\n"
