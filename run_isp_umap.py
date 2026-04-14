import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from datetime import datetime, timezone
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_mean_embs(hidden_state, length, max_len):
    """Mean pool hidden states based on actual non-padded sequence lengths."""
    # hidden_state: (batch, seq_len, hidden_dim)
    # length: (batch,)
    batch_size = hidden_state.size(0)
    device = hidden_state.device
    mask = torch.arange(max_len, device=device).unsqueeze(0) < length.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(hidden_state).float()
    masked_embs = hidden_state * mask
    mean_embs = masked_embs.sum(1) / length.view(-1, 1).float()
    return mean_embs

def extract_embeddings(model, dataset, device, batch_size, pad_token_id):
    """Run model forward passes to extract sequence embeddings."""
    all_embs = []
    model.eval()

    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        max_len = max(batch["length"])
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_id_list in batch["input_ids"]:
            length = len(input_id_list)
            padded_input_ids.append(input_id_list + [pad_token_id] * (max_len - length))
            padded_attention_masks.append([1] * length + [0] * (max_len - length))
            
        input_ids = torch.tensor(padded_input_ids).to(device)
        attention_mask = torch.tensor(padded_attention_masks).to(device)
        lengths = torch.tensor(batch["length"]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # using last hidden state 
            hidden_states = outputs.hidden_states[-1]
            embs = compute_mean_embs(hidden_states, lengths, max_len=input_ids.shape[1])
            all_embs.append(embs.cpu().numpy())
            
    if not all_embs:
        return np.array([])
    return np.concatenate(all_embs, axis=0)

def main():
    parser = argparse.ArgumentParser(description="ISP UMAP Plotter")
    default_cfg = os.environ.get("ISP_UMAP_CONFIG", "/app/config/isp_umap.yaml")
    parser.add_argument("--config", type=str, default=default_cfg, 
                        help="YAML config path (default: /app/config/isp_umap.yaml)")
    args = parser.parse_args()
    
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        return
        
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    time_str = datetime.now(timezone.utc).strftime("%H%M%S")
    out_dir = Path("/app/output") / date_str / f"isp_umap_{time_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loaded config from {cfg_path}")
    logger.info(f"Output directory initialized at {out_dir}")

    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load the model
    geneformer_model_path = cfg["paths"]["geneformer_model"]
    num_classes = cfg["runtime"].get("num_classes", 2)
    logger.info(f"Loading sequence classification model: {geneformer_model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        geneformer_model_path,
        num_labels=num_classes,
        output_hidden_states=True
    ).to(device)

    # 2. Get tokens
    from geneformer import tokenizer as gf_tokenizer
    from geneformer.in_silico_perturber_stats import GENE_NAME_ID_DICTIONARY_FILE
    
    with open(gf_tokenizer.TOKEN_DICTIONARY_FILE, 'rb') as f:
        token_dict = pickle.load(f)
    pad_token_id = token_dict.get("<pad>", 0)

    # Retrieve Gene Ensembl ID
    raw_gene = cfg["perturbation"]["gene_to_perturb"]
    ensembl_id = raw_gene
    
    # Check if raw_gene is a symbol, if it doesn't start with ENS
    if not str(raw_gene).startswith("ENS"):
        logger.info(f"Gene '{raw_gene}' looks like a symbol. Converting to Ensembl ID...")
        if GENE_NAME_ID_DICTIONARY_FILE.exists():
            with open(GENE_NAME_ID_DICTIONARY_FILE, "rb") as f:
                name_id_dict = pickle.load(f)
            if raw_gene in name_id_dict:
                ensembl_id = name_id_dict[raw_gene]
                logger.info(f"Successfully converted {raw_gene} -> {ensembl_id}")
            else:
                logger.error(f"Could not find Ensembl ID for symbol {raw_gene}. Stopping.")
                return
        else:
            logger.error(f"Gene name dictionary file not found at {GENE_NAME_ID_DICTIONARY_FILE}. Please supply pure Ensembl IDs.")
            return

    gene_token = token_dict.get(ensembl_id)
    if not gene_token:
        logger.error(f"Token for {ensembl_id} ({raw_gene}) not found inside the model token dictionary.")
        return
    logger.info(f"Target token for '{raw_gene}' ({ensembl_id}): {gene_token}")

    # 3. Load dataset
    dataset_path = cfg["paths"]["dataset"]
    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    state_key = cfg["perturbation"]["state_key"]
    start_state = cfg["perturbation"]["start_state"]
    end_state = cfg["perturbation"]["end_state"]
    max_cells = cfg["umap"].get("max_cells_per_state", 2000)

    logger.info(f"Filtering dataset for exactly {max_cells} '{start_state}' cells and '{end_state}' cells...")
    start_dataset = dataset.filter(lambda x: x.get(state_key) == start_state)
    end_dataset = dataset.filter(lambda x: x.get(state_key) == end_state)
    
    start_dataset = start_dataset.select(range(min(len(start_dataset), max_cells)))
    end_dataset = end_dataset.select(range(min(len(end_dataset), max_cells)))
    logger.info(f"Found {len(start_dataset)} {start_state} cells and {len(end_dataset)} {end_state} cells.")

    # 4. Perturb Dataset manually (currently only simulates deletion)
    perturb_type = cfg["perturbation"].get("type", "delete")
    if perturb_type != "delete":
        logger.warning(f"Currently, only 'delete' perturbation is built-in to UMAP extract. Enforcing 'delete'.")
        
    def delete_perturb_dataset(batch):
        new_input_ids = []
        new_lengths = []
        new_attention_masks = []
        for input_id_list in batch["input_ids"]:
            perturbed_ids = [tok for tok in input_id_list if tok != gene_token]
            length = len(perturbed_ids)
            new_input_ids.append(perturbed_ids)
            new_lengths.append(length)
            new_attention_masks.append([1] * length)
        return {"input_ids": new_input_ids, "length": new_lengths, "attention_mask": new_attention_masks}

    logger.info(f"Modifying the data to mathematically apply the perturbation [{raw_gene}]...")
    batch_size = cfg["runtime"].get("batch_size", 50)
    pert_start_dataset = start_dataset.map(delete_perturb_dataset, batched=True, batch_size=batch_size)

    # 5. Extract Embeddings
    logger.info("Pushing data through the model to extract embeddings... (this may take a minute)")
    logger.info(f"-> Extracting {end_state} embeddings...")
    end_embs = extract_embeddings(model, end_dataset, device, batch_size, pad_token_id)
    
    logger.info(f"-> Extracting {start_state} embeddings...")
    start_embs = extract_embeddings(model, start_dataset, device, batch_size, pad_token_id)
    
    logger.info(f"-> Extracting {start_state}+ISP({raw_gene}) embeddings...")
    pert_start_embs = extract_embeddings(model, pert_start_dataset, device, batch_size, pad_token_id)

    # 6. Save raw NumPy Arrays
    logger.info("Saving underlying embeddings to arrays...")
    np.save(str(out_dir / f"{end_state}_embs.npy"), end_embs)
    np.save(str(out_dir / f"{start_state}_embs.npy"), start_embs)
    np.save(str(out_dir / f"{start_state}_ISP_{raw_gene}_embs.npy"), pert_start_embs)

    all_embs = np.vstack([end_embs, start_embs, pert_start_embs])
    labels = (
        [end_state] * len(end_embs) + 
        [start_state] * len(start_embs) + 
        [f"{start_state}+ISP({raw_gene})"] * len(pert_start_embs)
    )

    # 7. UMAP Generation
    n_neighbors = cfg["umap"].get("n_neighbors", 15)
    min_dist = cfg["umap"].get("min_dist", 0.1)
    umap_seed = cfg["umap"].get("seed", 42)
    
    try:
        import cuml
        logger.info("Running RAPIDS cuML GPU UMAP...")
        reducer = cuml.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=umap_seed)
        umap_embs = reducer.fit_transform(all_embs)
    except Exception as e:
        logger.warning(f"RAPIDS cuML failed: {e}. Falling back to standard CPU UMAP...")
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=umap_seed)
        umap_embs = reducer.fit_transform(all_embs)

    df = pd.DataFrame(umap_embs, columns=["UMAP 1", "UMAP 2"])
    df["State"] = labels

    # 8. Plot Generation
    logger.info("Rendering visual plot...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x="UMAP 1", y="UMAP 2", hue="State", alpha=0.7, 
        palette={end_state: "steelblue", start_state: "coral", f"{start_state}+ISP({raw_gene})": "red"}
    )
    plt.title(f"UMAP of Embeddings: {end_state} vs {start_state} ({raw_gene} ISP)")

    # Draw trajectories
    num_arrows = cfg["umap"].get("num_trajectory_arrows", 100)
    start_idx = len(end_embs)
    pert_start_idx = len(end_embs) + len(start_embs)
    
    step = max(1, len(start_embs) // max(num_arrows, 1))
    
    for i in range(0, len(start_embs), step):
        plt.arrow(
            umap_embs[start_idx+i, 0], umap_embs[start_idx+i, 1],
            umap_embs[pert_start_idx+i, 0] - umap_embs[start_idx+i, 0],
            umap_embs[pert_start_idx+i, 1] - umap_embs[start_idx+i, 1],
            color='gray', alpha=0.3, width=0.01, head_width=0.1
        )

    out_file = out_dir / f"umap_{start_state}_vs_{end_state}_isp_{raw_gene}.png"
    plt.savefig(out_file, bbox_inches="tight")
    logger.info(f"Saved UMAP plot to {out_file}")
    logger.info("ISP UMAP pipeline finished successfully.")

if __name__ == "__main__":
    main()
