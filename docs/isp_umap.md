# in-sillico perturbation visualization on UMAP Plot Service

The **ISP UMAP** service calculates and visualizes the topological changes of perturbed cell embeddings in UMAP space. Unlike the mathematical cosine-similarity shift measured by the standard ISP scripts, this tool provides a visual representation of how the perturbation of a specific gene moves individual cells across conditions.

## Configuration

The service is fully configured via [`config/isp_umap.yaml`](../config/isp_umap.yaml). You may modify this file to point to different datasets, fine-tuned models, or target genes.

Key configurations to note:

| YAML area | What to set |
|-----------|-------------|
| `paths.dataset` | Tokenized `.dataset` directory containing your original and condition-annotated cells. |
| `paths.geneformer_model` | Path to your pre-trained or fine-tuned sequence classification model. |
| `umap.seed` | Seed for cuML/UMAP initialization (default `42`) to ensure reproducible projections. |
| `umap.num_trajectory_arrows` | The number of trajectories to draw. We limit this locally to prevent overplotting. |
| `perturbation.gene_to_perturb` | The Gene Symbol (e.g. `Igfbp2`) or Ensembl ID (e.g. `ENSMUSG00000039323`) to perturb. |
| `perturbation.state_key` | Label column that divides your cells (e.g., `disease`). |
| `perturbation.start_state` | Condition you are perturbing (e.g., `AD`). |
| `perturbation.end_state` | Condition you are trying to match or compare against (e.g., `WT`). |

### Gene Symbol Auto-Detection

The `gene_to_perturb` parameter supports both human-readable Gene Symbols (e.g., `Igfbp2`) and explicit Ensembl IDs. If a Gene Symbol is provided, the script will automatically consult internal Geneformer dictionaries (`GENE_NAME_ID_DICTIONARY_FILE`) to convert it mapping to the proper `.dataset` token.

## Running

```bash
docker compose run --rm isp_umap
```

## Outputs

All generated assets are safely routed to the `output/[DATE]/isp_umap_[UTC TIME]` directory.

### 1. UMAP Figure (`umap_*.png`)
A visually distinct seaborn scatterplot comparing:
- Target State cells (e.g. `WT` in blue)
- Start State cells (e.g. `AD` in orange)
- ISP Perturbed cells (e.g. `AD + ISP(Igfbp2)` in red).

Grey arrows signify the individual trajectory lines map tracking the progression of an exact single cell before perturbation to its specific position after perturbation.

### 2. Raw Embeddings arrays (`*.npy`)
The script exports native `.npy` representations of your intermediate matrix spaces (`WT_embs.npy`, `AD_embs.npy`, `AD_ISP..._.npy`) so that you can reuse them if you perform downstream tasks inside of Jupyter.

## Troubleshooting

- **`cuML failed: nvrtc...`**: By default, the script will aim for RAPIDS `cuML` GPU acceleration. If your CUDA hardware architecture is not inherently supported or fails compilation, it will gracefully fallback to multi-threaded CPU `umap-learn` without interrupting execution.
