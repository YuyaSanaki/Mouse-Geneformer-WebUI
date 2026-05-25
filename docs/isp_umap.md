# in-sillico perturbation visualization on UMAP Plot Service

The **ISP UMAP** service calculates and visualizes how **each cell** moves when a single gene is perturbed in silico. Unlike standard ISP (gene-level ranking via cosine similarity across the cohort), this service exports **per-cell shift metrics** and a UMAP trajectory plot for one target gene at a time.

**Core output:** [`per_cell_isp_shift.csv`](#3-per-cell-shift-table-per_cell_isp_shiftcsv) — one row per start-state cell (e.g. `Disease`) with `shift_l2` (embedding-space perturbation magnitude), `shift_toward_<end_state>` (movement toward e.g. `Ctrl`), and UMAP before/after coordinates. Use this table to identify which cells (and later, which cell types after you join annotations) are most affected by the perturbation.

## Configuration

The service is fully configured via [`config/isp_umap.yaml`](../config/isp_umap.yaml). You may modify this file to point to different datasets, fine-tuned models, or target genes.

Key configurations to note:

| YAML area | What to set |
|-----------|-------------|
| `paths.dataset` | Tokenized `.dataset` directory containing your original and condition-annotated cells. |
| `paths.geneformer_model` | Path to your pre-trained or fine-tuned sequence classification model. |
| `umap.seed` | Seed for cuML/UMAP initialization (default `42`) to ensure reproducible projections. |
| `umap.num_trajectory_arrows` | The number of trajectories to draw. We limit this locally to prevent overplotting. |
| `perturbation.gene_to_perturb` | Gene symbol (e.g. `TargetGene`) or Ensembl ID to perturb. |
| `perturbation.state_key` | Label column that divides your cells (e.g., `disease`). |
| `perturbation.start_state` | Condition you are perturbing (e.g. `Disease`). |
| `perturbation.end_state` | Condition you are comparing against (e.g. `Ctrl`). |

### Gene Symbol Auto-Detection

The `gene_to_perturb` parameter supports gene symbols (e.g. `TargetGene`) and Ensembl IDs. Symbols are resolved via internal Geneformer dictionaries (`GENE_NAME_ID_DICTIONARY_FILE`) to the proper `.dataset` token.

## Running

```bash
docker compose run --rm isp_umap
```

## Outputs

All generated assets are safely routed to the `output/[DATE]/isp_umap_[UTC TIME]` directory.

| File | Role |
|------|------|
| **`per_cell_isp_shift.csv`** | **Essential:** per-cell perturbation magnitude and direction (see below) |
| `umap_*.png` | Visual summary; grey arrows = same cells as `umap_shift_l2` in the CSV |
| `*_embs.npy` | Raw embedding matrices for custom downstream analysis |

### 1. UMAP Figure (`umap_*.png`)
A visually distinct seaborn scatterplot comparing:
- Target state cells (e.g. `Ctrl` in blue)
- Start state cells (e.g. `Disease` in orange)
- ISP perturbed cells (e.g. `Disease + ISP(TargetGene)` in red).

Grey arrows signify the individual trajectory lines map tracking the progression of an exact single cell before perturbation to its specific position after perturbation.

### 2. Raw Embeddings arrays (`*.npy`)
The script exports native `.npy` representations of intermediate embeddings (`Ctrl_embs.npy`, `Disease_embs.npy`, `Disease_ISP..._.npy`) for custom downstream analysis.

### 3. Per-cell shift table (`per_cell_isp_shift.csv`)
One row per **start-state** cell (e.g. each `Disease` cell in the run), with how much that cell moved under ISP:

| Column | Meaning |
|--------|---------|
| `shift_l2` | L2 distance between embeddings before vs after perturbation (larger = more perturbed in model space) |
| `shift_toward_<end_state>` | Reduction in distance to the end-state (e.g. `Ctrl`) centroid; positive values move closer to that reference |
| `umap1_before` / `umap2_before` | UMAP position before perturbation |
| `umap1_after` / `umap2_after` | UMAP position after perturbation |
| `umap_shift_l2` | L2 distance between before/after positions in UMAP space (matches the grey arrows on the plot) |

Dataset metadata columns present on the tokenized `.dataset` (e.g. `sample_id`, `disease`) are included so you can join cell-type labels later after classification.

## Troubleshooting

- **`cuML failed: nvrtc...`**: By default, the script will aim for RAPIDS `cuML` GPU acceleration. If your CUDA hardware architecture is not inherently supported or fails compilation, it will gracefully fallback to multi-threaded CPU `umap-learn` without interrupting execution.
