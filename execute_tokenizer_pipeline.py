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
from geneformer import TranscriptomeTokenizer
from run_pipeline_log import format_tokenize_run_banner, install_rotating_stdio_tee
from run_provenance import write_service_provenance, update_service_provenance

def process_single_cell_to_loom(input_dir, loom_temp_dir, settings):
    """Convert subdirectories of (barcodes/features/matrix) to .loom files."""
    os.makedirs(loom_temp_dir, exist_ok=True)
    
    # List all subdirectories
    sample_dirs = sorted([os.path.join(input_dir, d) for d in os.listdir(input_dir) 
                  if os.path.isdir(os.path.join(input_dir, d)) 
                  and not d.startswith(".") 
                  and os.path.abspath(os.path.join(input_dir, d)) != os.path.abspath(loom_temp_dir)])
    
    if not sample_dirs:
        print(f"No sample directories found in {input_dir}")
        return

    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir.rstrip('/'))
        
        # Check for filtered_feature_bc_matrix/ (standard for many platforms)
        mtx_path = os.path.join(sample_dir, "filtered_feature_bc_matrix")
        if not os.path.exists(mtx_path):
            if os.path.exists(os.path.join(sample_dir, "matrix.mtx.gz")):
                mtx_path = sample_dir
            else:
                continue

        print(f"Converting {sample_name} to loom...")
        try:
            # Read mtx and set Ensembl IDs
            adata = sc.read_10x_mtx(mtx_path, var_names='gene_ids', make_unique=True)
            adata.var['ensembl_id'] = adata.var_names.astype(str)
            adata.obs['n_counts'] = adata.X.sum(axis=1).A1 if hasattr(adata.X, "sum") else adata.X.sum(axis=1)

            if settings.get('extract_metadata_from_path'):
                parts = sample_name.split('-')
                if len(parts) >= 3:
                    adata.obs['time'], adata.obs['genotype'], adata.obs['replicate'] = parts[0], parts[1], parts[2]
                    adata.obs['disease'] = parts[1]
                else:
                    adata.obs['disease'] = sample_name
            
            adata.obs['sample_id'] = sample_name
            adata.write_loom(os.path.join(loom_temp_dir, f"{sample_name}.loom"))
        except Exception as e:
            print(f"Error converting {sample_name}: {e}")

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
    if data_cfg['input_type'] == "single-cell":
        print("Input type is single-cell. Converting to loom first...")
        process_single_cell_to_loom(
            data_cfg['input_dir'], 
            data_cfg['loom_temp_dir'], 
            single_cell_settings,
        )
        tokenizer_input_dir = data_cfg['loom_temp_dir']
    else:
        print("Input type is loom. Skipping conversion.")
        tokenizer_input_dir = data_cfg['input_dir']

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
        nproc=tokenizer_nproc
    )
    
    print(f"Starting Tokenization of files in {tokenizer_input_dir}...")
    pipeline_ok = False
    try:
        tk.tokenize_data(
            data_directory=tokenizer_input_dir,
            output_directory=data_cfg['output_dir'],
            output_prefix=data_cfg['output_prefix']
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
        )

if __name__ == "__main__":
    main()
