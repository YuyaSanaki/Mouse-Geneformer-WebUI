import os
import shutil
import subprocess
import sys
import yaml
import scanpy as sc
import anndata as ad
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add current directory to path so geneformer and provenance can be imported
sys.path.append(os.getcwd())
from data_input_layout import (
    diagnose_input_dir,
    discover_sample_dirs,
    parse_sample_folder_name,
    resolve_single_cell_input_dir,
    sample_label,
    tenx_matrix_directory,
)
from geneformer import TranscriptomeTokenizer
from run_pipeline_log import format_tokenize_run_banner, install_rotating_stdio_tee
from run_provenance import write_service_provenance, update_service_provenance


def process_single_cell_to_loom(input_dir, loom_temp_dir, settings, tokenizer_cfg) -> int:
    """Convert subdirectories of (barcodes/features/matrix) to .loom files. Returns loom count."""
    os.makedirs(loom_temp_dir, exist_ok=True)
    input_path = Path(input_dir).resolve()
    loom_path = Path(loom_temp_dir).resolve()
    data_root = resolve_single_cell_input_dir(input_path, loom_path)
    if data_root != input_path:
        print(f"Resolved single-cell study root: {data_root} (from {input_path})")

    sample_dirs = discover_sample_dirs(data_root, exclude=loom_path)
    if sample_dirs:
        print(f"Found {len(sample_dirs)} sample(s): {', '.join(p.name for p in sample_dirs)}")

    if not sample_dirs:
        print(f"No 10x sample directories found under {data_root}")
        print(diagnose_input_dir(input_path, loom_path))
        return 0

    converted = 0
    for sample_dir in sample_dirs:
        sample_path = Path(sample_dir)
        folder_name = sample_path.name
        loom_stem = sample_label(sample_path, data_root)
        mtx_path_obj = tenx_matrix_directory(sample_path)
        if mtx_path_obj is None:
            print(f"Skipping {folder_name}: no 10x matrix files found.")
            continue
        mtx_path = str(mtx_path_obj)

        print(f"Converting {folder_name} → {loom_stem}.loom (from {mtx_path})...")
        try:
            # Read mtx and set Ensembl IDs
            adata = sc.read_10x_mtx(mtx_path, var_names='gene_ids', make_unique=True)
            # Strip version numbers from Ensembl IDs (e.g., ENSMUSG00000102693.2 -> ENSMUSG00000102693)
            adata.var['ensembl_id'] = adata.var_names.astype(str).str.split('.').str[0]
            adata.obs['n_counts'] = adata.X.sum(axis=1).A1 if hasattr(adata.X, "sum") else adata.X.sum(axis=1)

            if settings.get("extract_metadata_from_path"):
                meta = parse_sample_folder_name(folder_name)
                adata.obs["time"] = meta["time"]
                adata.obs["genotype"] = meta["genotype"]
                adata.obs["disease"] = meta["disease"]
                adata.obs["replicate"] = meta["replicate"]
                adata.obs["sample_id"] = meta["sample_id"]
            elif tokenizer_cfg.get("custom_attr_name_dict"):
                adata.obs["sample_id"] = folder_name

            if tokenizer_cfg.get("custom_attr_name_dict"):
                for attr in tokenizer_cfg["custom_attr_name_dict"].keys():
                    if attr not in adata.obs.columns:
                        adata.obs[attr] = ""

            if "sample_id" not in adata.obs.columns:
                adata.obs["sample_id"] = folder_name
            adata.write_loom(os.path.join(loom_temp_dir, f"{loom_stem}.loom"))
            converted += 1
        except Exception as e:
            print(f"Error converting {folder_name}: {e}")

    print(f"Converted {converted} sample(s) to .loom under {loom_temp_dir}")
    return converted


def main():
    config_path = Path(os.getenv("TOKENIZE_CONFIG", "/app/config/tokenize.yaml")).expanduser()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    tokenizer_cfg = config['tokenizer']
    tokenizer_nproc = int(tokenizer_cfg.get('nproc', 1))
    single_cell_settings = config.get('single_cell_settings', {}) or {}

    out_dir = Path(data_cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "tokenize_run.log"
    install_rotating_stdio_tee(log_path, env_prefix="TOKENIZE")
    print(
        format_tokenize_run_banner(
            config_path.resolve(),
            data_cfg,
            tokenizer_cfg,
            single_cell_settings,
        ),
        end="",
    )
    print(f"  run log (rotating): {log_path}")

    # Step 1: Handle Conversion if needed
    if data_cfg["input_type"] == "single-cell":
        print("Input type is single-cell. Converting to loom first...")
        study_root = resolve_single_cell_input_dir(
            data_cfg["input_dir"], data_cfg.get("loom_temp_dir")
        )
        if str(study_root) != str(Path(data_cfg["input_dir"]).resolve()):
            print(
                f"Note: data.input_dir was {data_cfg['input_dir']}; "
                f"using study root {study_root} (ExperimentName / Time-Condition-Replicate layout)."
            )
        data_cfg["input_dir"] = str(study_root)
        n_converted = process_single_cell_to_loom(
            data_cfg["input_dir"],
            data_cfg['loom_temp_dir'],
            single_cell_settings,
            tokenizer_cfg,
        )
        loom_dir = Path(data_cfg['loom_temp_dir'])
        n_looms = len(list(loom_dir.glob("*.loom")))
        if n_looms == 0:
            print(diagnose_input_dir(data_cfg['input_dir'], data_cfg['loom_temp_dir']))
            raise SystemExit(
                "No .loom files produced. Fix data.input_dir layout (see message above) and retry."
            )
        tokenizer_input_dir = str(loom_dir) + ("" if str(loom_dir).endswith(os.sep) else os.sep)
        file_format = "loom"
    else:
        print("Input type is loom. Skipping conversion.")
        tokenizer_input_dir = data_cfg['input_dir']
        file_format = "loom"

    # Step 2: Provenance Tracking & Fingerprinting
    input_paths = {
        "input_data": data_cfg['input_dir']
    }
    
    # Read provenance settings from config with sensible defaults
    prov_cfg = config.get('provenance') or {}
    enable_fp = prov_cfg.get('enable_input_fingerprint', False)
    # Environment variable still takes precedence if set
    if os.environ.get("ENABLE_INPUT_FINGERPRINT"):
        enable_fp = os.environ.get("ENABLE_INPUT_FINGERPRINT").lower() in ("1", "true", "yes")
    
    is_fast = prov_cfg.get('fingerprint_fast', True)
    if os.environ.get("FINGERPRINT_FAST"):
        is_fast = os.environ.get("FINGERPRINT_FAST").lower() in ("1", "true", "yes")
    
    # Temporarily set environment variable for run_provenance utility if enabled by config
    if enable_fp:
        os.environ["ENABLE_INPUT_FINGERPRINT"] = "true"

    write_service_provenance(
        run_root=out_dir,
        service="tokenize",
        config_path=config_path,
        extra_meta={
            "service": "tokenize",
            "tokenizer": {"nproc": tokenizer_nproc},
            "data": {
                "input_type": data_cfg.get("input_type"),
                "input_dir": data_cfg.get("input_dir"),
                "loom_temp_dir": data_cfg.get("loom_temp_dir"),
                "output_dir": data_cfg.get("output_dir"),
                "output_prefix": data_cfg.get("output_prefix"),
            },
        },
        input_paths=input_paths,
        fast_fingerprint=is_fast
    )
    print(f"  Provenance saved in: {out_dir}")

    # Step 3: Tokenize
    print(f"Initializing Tokenizer with nproc={tokenizer_nproc}...")
    
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=tokenizer_cfg.get('custom_attr_name_dict'), 
        nproc=tokenizer_nproc,
        max_cells=int(tokenizer_cfg.get('max_cells', 300_000))
    )
    
    print(f"Starting Tokenization of files in {tokenizer_input_dir}...")
    pipeline_ok = False
    try:
        tk.tokenize_data(
            data_directory=tokenizer_input_dir,
            output_directory=data_cfg['output_dir'],
            output_prefix=data_cfg['output_prefix'],
            file_format=file_format
        )
        pipeline_ok = True
        print("Pipeline Finished.")
    finally:
        update_service_provenance(
            out_dir,
            {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_status": "completed" if pipeline_ok else "failed",
            },
            service="tokenize",
        )

if __name__ == "__main__":
    main()
