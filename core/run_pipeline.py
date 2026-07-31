"""
End-to-end pipeline: Tokenize → Fine-tune → ISP.

Reads config/pipeline.yaml (or PIPELINE_CONFIG), derives chained paths from
data.input_dir, writes per-stage configs under the run directory, and runs
each stage sequentially.

Usage:
  docker compose run --rm pipeline
  python3 run_pipeline.py --config /app/config/my_pipeline.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pipeline_lib import (
    ROOT,
    build_finetune_config,
    build_isp_config,
    build_tokenize_config,
    format_pipeline_banner,
    load_yaml,
    resolve_pipeline_paths,
    study_name_from_input_dir,
    write_yaml,
)
from run_pipeline_log import install_rotating_stdio_tee
from run_provenance import update_service_provenance, write_service_provenance


def _utc_pipeline_folder_name(now_utc: datetime) -> str:
    return f"pipeline_{now_utc.strftime('%H%M%S')}_{now_utc.microsecond:06d}Z"


def _run_subprocess(cmd: list[str], env: dict[str, str], label: str) -> None:
    print(f"\n>>> Starting {label}: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    print(f"\n>>> {label} finished OK.\n", flush=True)


def _num_classes_from_label_dict(model_dir: Path) -> int:
    path = model_dir / "label_dict.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Fine-tune checkpoint missing label_dict.json: {path}. "
            "Check finetune logs; ISP needs num_classes from this file."
        )
    with open(path, encoding="utf-8") as f:
        label_dict = json.load(f)
    if not isinstance(label_dict, dict) or not label_dict:
        raise ValueError(f"Invalid label_dict.json: {path}")
    return len(label_dict)


def main() -> None:
    default_cfg = os.environ.get("PIPELINE_CONFIG", str(ROOT / "config" / "pipeline.yaml"))
    p = argparse.ArgumentParser(description="Run Tokenize → Fine-tune → ISP pipeline.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(default_cfg),
        help="Pipeline YAML (default: PIPELINE_CONFIG or config/pipeline.yaml).",
    )
    p.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Skip tokenization (dataset must already exist at derived path).",
    )
    p.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip fine-tuning (use existing checkpoint under finetune/all_run1).",
    )
    p.add_argument(
        "--skip-isp",
        action="store_true",
        help="Stop after fine-tuning.",
    )
    args = p.parse_args()

    pipeline_path = args.config.expanduser().resolve()
    pipeline = load_yaml(pipeline_path)

    data = pipeline.get("data") or {}
    if not data.get("input_dir"):
        raise ValueError("pipeline data.input_dir is required.")

    if data.get("output_prefix") in (None, "", "null"):
        study = study_name_from_input_dir(str(data["input_dir"]))
        pipeline = dict(pipeline)
        pipeline["data"] = dict(data)
        pipeline["data"]["output_prefix"] = study

    date_used = datetime.now().strftime("%Y%m%d")
    paths_cfg = pipeline.get("paths") or {}
    output_root = Path(str(paths_cfg.get("output_root") or "/app/output"))
    now_utc = datetime.now(timezone.utc)
    run_dir = output_root / date_used / _utc_pipeline_folder_name(now_utc)
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_pipeline_paths(pipeline, run_dir)
    stage_dir = run_dir / "stage_configs"
    tokenize_cfg_path = stage_dir / "tokenize.yaml"
    finetune_cfg_path = stage_dir / "finetune.yaml"
    isp_cfg_path = stage_dir / "isp.yaml"

    tokenize_tpl = load_yaml(ROOT / "config" / "tokenize.yaml")
    finetune_tpl = load_yaml(ROOT / "config" / "finetune.yaml")
    isp_tpl = load_yaml(ROOT / "config" / "isp.yaml")

    tokenize_cfg = build_tokenize_config(pipeline, resolved, tokenize_tpl)
    finetune_cfg = build_finetune_config(pipeline, resolved, finetune_tpl)

    write_yaml(tokenize_cfg_path, tokenize_cfg)
    write_yaml(finetune_cfg_path, finetune_cfg)

    log_path = run_dir / "pipeline_run.log"
    install_rotating_stdio_tee(log_path, env_prefix="PIPELINE")

    stage_paths = {
        "tokenize": tokenize_cfg_path,
        "finetune": finetune_cfg_path,
        "isp": isp_cfg_path,
    }
    print(
        format_pipeline_banner(pipeline_path, resolved, stage_paths),
        end="",
    )
    print(f"  pipeline log: {log_path}")

    shutil.copy2(pipeline_path, run_dir / "pipeline_config_used.yaml")
    write_yaml(run_dir / "pipeline_resolved_paths.yaml", resolved)

    write_service_provenance(
        run_root=run_dir,
        service="pipeline",
        config_path=pipeline_path,
        extra_meta={
            "service": "pipeline",
            "resolved_paths": resolved,
            "stage_configs": {k: str(v) for k, v in stage_paths.items()},
        },
        input_paths={"input_dir": resolved["input_dir"]},
        fast_fingerprint=True,
    )

    pipeline_ok = False
    finetune_model = Path(resolved["finetune_model_dir"])

    try:
        if not args.skip_tokenize:
            env = os.environ.copy()
            env["TOKENIZE_CONFIG"] = str(tokenize_cfg_path)
            _run_subprocess(
                [sys.executable, str(ROOT / "execute_tokenizer_pipeline.py")],
                env,
                "Tokenize",
            )
        else:
            print("Skipping tokenize (--skip-tokenize).")

        dataset_path = Path(resolved["dataset_path"])
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"Tokenized dataset not found: {dataset_path}. "
                "Run tokenize or fix data.output_prefix / input_dir."
            )

        if not args.skip_finetune:
            env = os.environ.copy()
            env["FINETUNE_CONFIG"] = str(finetune_cfg_path)
            _run_subprocess(
                [sys.executable, str(ROOT / "run_finetune.py"), "--config", str(finetune_cfg_path)],
                env,
                "Fine-tune",
            )
        else:
            print("Skipping fine-tune (--skip-finetune).")

        if not finetune_model.is_dir():
            raise FileNotFoundError(
                f"Fine-tuned model directory not found: {finetune_model}. "
                "Expected all_run1/ under the pipeline finetune folder."
            )

        num_classes = _num_classes_from_label_dict(finetune_model)
        isp_cfg = build_isp_config(
            pipeline,
            resolved,
            isp_tpl,
            finetune_model_dir=str(finetune_model),
            num_classes=num_classes,
        )
        write_yaml(isp_cfg_path, isp_cfg)

        if not args.skip_isp:
            nproc = os.environ.get("ISP_NUM_GPUS", "1")
            env = os.environ.copy()
            env["ISP_CONFIG"] = str(isp_cfg_path)
            _run_subprocess(
                [
                    "accelerate",
                    "launch",
                    "--num_processes",
                    str(nproc),
                    str(ROOT / "run_isp.py"),
                    "--config",
                    str(isp_cfg_path),
                    "--no-output-time-subdir",
                    "--no-output-date-subdir",
                ],
                env,
                "ISP",
            )
        else:
            print("Skipping ISP (--skip-isp).")

        pipeline_ok = True
        print("\n" + "=" * 72)
        print("Pipeline complete.")
        print("=" * 72)
        print(f"  Run directory:     {run_dir}")
        print(f"  Tokenized data:    {resolved['dataset_path']}")
        print(f"  Fine-tuned model:  {finetune_model}")
        if not args.skip_isp:
            print(f"  ISP outputs:       {resolved['pipeline_run_dir']}/isp_results/")
        print("=" * 72)

    finally:
        update_service_provenance(
            run_dir,
            {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_status": "completed" if pipeline_ok else "failed",
                "finetune_model_dir": str(finetune_model),
                "dataset_path": resolved["dataset_path"],
            },
            service="pipeline",
        )

    if not pipeline_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
