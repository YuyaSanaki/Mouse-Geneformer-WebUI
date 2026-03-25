import os
import sys
import yaml
import scanpy as sc
import anndata as ad
from pathlib import Path

# Add current directory to path so geneformer can be imported
sys.path.append(os.getcwd())
from geneformer import TranscriptomeTokenizer

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
    config_path = os.getenv("TOKENIZE_CONFIG", "/app/config/tokenize.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    tokenizer_cfg = config['tokenizer']
    
    # Step 1: Handle Conversion if needed
    if data_cfg['input_type'] == "single-cell":
        print("Input type is single-cell. Converting to loom first...")
        process_single_cell_to_loom(
            data_cfg['input_dir'], 
            data_cfg['loom_temp_dir'], 
            config.get('single_cell_settings', {})
        )
        tokenizer_input_dir = data_cfg['loom_temp_dir']
    else:
        print("Input type is loom. Skipping conversion.")
        tokenizer_input_dir = data_cfg['input_dir']

    # Step 2: Tokenize
    os.makedirs(data_cfg['output_dir'], exist_ok=True)
    print(f"Initializing Tokenizer with nproc={tokenizer_cfg.get('nproc', 1)}...")
    
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=tokenizer_cfg.get('custom_attr_name_dict'), 
        nproc=tokenizer_cfg.get('nproc', 1)
    )
    
    print(f"Starting Tokenization of files in {tokenizer_input_dir}...")
    tk.tokenize_data(
        data_directory=tokenizer_input_dir,
        output_directory=data_cfg['output_dir'],
        output_prefix=data_cfg['output_prefix']
    )
    print("Pipeline Finished.")

if __name__ == "__main__":
    main()
