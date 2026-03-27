# In-silico perturbation (ISP) — new biology

This guide describes the **end-to-end workflow** to run Geneformer ISP on a **new experiment** (e.g. different tissue, genotype, or condition). Batch entrypoint: [`run_isp.py`](../run_isp.py). Configuration: [`config/isp.yaml`](../config/isp.yaml). Docker: [`docker-compose.yml`](../docker-compose.yml).

---

## 1. Define the question

- **Goal-state shift** evaluates how much in silico perturbation of a gene in start-labeled cells (e.g. wild type) shifts the model’s representation toward end-labeled cells (e.g. knockout).
- **What it measures:** changes in embedding similarity (based on ranked gene expression) in the model’s embedding space — not raw transcript counts or per-gene fold-change (i.e., whether the perturbed WT cell cluster moves towards the KO cell cluster or away from it).
- **Data requirements:** the tokenized `.dataset` generated from real scRNA-seq of both baseline and target states; label strings must exactly match `perturbation.start_state` and `perturbation.end_state` in `isp.yaml`.
- Decide **which cell types** proceed to the analysis (e.g. skeletal muscle only). You can enforce that **before tokenization** (subset cells) or **after**, using `isp.filter_data` in `isp.yaml` if those columns exist in the tokenized dataset.

---

## 2. Obtain expression data

You need **raw counts** (single-cell or single-nuc, DO NOT normalize the count) for both start-label (eg WT) and end-label (eg KO). Minimum 500 cells in the cell type of interest; a greater cell number is better. For example, it is ok if there are 1,000 cells of interest (eg muscle) in your single-cell data which contains 8,000 cells in your single-cell data (eg cardiac infarction).

### Acceptable file formats (for `tokenize service`)
The raw count must have three files (barcodes.tsv.gz, features.tsv.gz, and matrix.trx.gz) in a folder. The file names should be exactly same as "barcodes.tsv.gz, features.tsv.gz, and matrix.trx.gz" as originally generated from single-cell experiment.

The three files should be in a folder named representing your samples in three slot ranged by hyphen. The naming rule is ./data/your-expeimrnt-name/Time-Condition-Replicate/gz files.
This folder naming is used for following data processing as:

| Field	| Rule |
|-------|------|
| Time | First segment of the folder name: parts[0] (e.g. 1w, 3w, 5w).|
| Condition |	Second segment: parts[1] (e.g. AD, WT, Ccr2KO). |
| Replicate |	Third segment: parts[2] (e.g. 1st, 2nd, even if you have only 1 replicate, put "1" in the third field). |
| sample_id |	Full folder name: sample_name (e.g. 1w-AD-1st, 5w-Ccr2KO-AD). |


Alternative satating point is .loom file. should locate in ./data/your-experiment-name/loom_files/loom files.

| Format | Notes |
|--------|--------|
| **`.loom`** | Primary format described in the tokenizer. Put one or more `*.loom` files in a directory and call `tokenize_data(..., file_format="loom")` (default). |
| **`.h5ad` (AnnData)** | Supported via `tokenize_data(..., file_format="h5ad")`. Same biology requirements as loom; fields below must be present in the object. |


### Required fields (mouse tokenizer)

**Genes**

- **`ensembl_id`**: Ensembl gene ID per gene (loom: row attribute `ensembl_id`; AnnData: `adata.var["ensembl_id"]`). Values must match the **mouse** token / median dictionaries used by the tokenizer.

**Cells**

- **`n_counts`**: total UMI/read counts per cell (loom: column attribute `n_counts`; AnnData: `adata.obs["n_counts"]`). Used for normalization before ranking.

**Optional**

- **`filter_pass`**: binary flag per cell; if missing, **all** cells are tokenized (see tokenizer messages).

**Metadata for ISP / YAML**

- Any columns you want in the final **`.dataset`** (and in `isp.yaml`) must be passed through the tokenizer via **`custom_attr_name_dict`**: maps **names in the loom/h5ad** → **column names in the output dataset** (e.g. map your condition column to `disease` if that is your `perturbation.state_key`).
- Those values must match **`start_state` / `end_state` / `alt_states`** exactly.

### Expression assumptions

- Use **raw counts**, **without** prior feature selection (per tokenizer docstring).
- Genes are median-scaled and rank-tokenized inside the tokenizer; you do **not** hand ISP a CSV of ranks.

---

## 3. Tokenize your data

ISP does **not** read **loom / h5ad / CSV** directly. It expects a **Hugging Face [`datasets`](https://huggingface.co/docs/datasets)** object **saved on disk** as a **`.dataset` directory** with at least:

| Column / field | Role |
|----------------|------|
| `input_ids` | Rank-encoded gene tokens per cell |
| `length` | Sequence length (added by [`TranscriptomeTokenizer`](../geneformer/tokenizer.py)) |
| Your state column | Named in YAML as `perturbation.state_key`; values must match `start_state` / `end_state` / `alt_states` **exactly** |
| Other metadata | Optional (e.g. `cell_type`, `time`) if present on the input and listed in `custom_attr_name_dict`, for `isp.filter_data` or bookkeeping |

**In this repository**, tokenization is driven by **[`config/tokenize.yaml`](../config/tokenize.yaml)** and **[`execute_tokenizer_pipeline.py`](../execute_tokenizer_pipeline.py)** (Compose service **`tokenize`** in [`docker-compose.yml`](../docker-compose.yml)).

1. **Configure** [`config/tokenize.yaml`](../config/tokenize.yaml): set `data.input_dir`, `data.output_dir`, `data.output_prefix`, and `data.input_type` (`single-cell` or `loom`). Set **`tokenizer.custom_attr_name_dict`** so every **key** is a column attribute on each `.loom` (or `obs` column in `.h5ad`) and every **value** is the column name in the saved `.dataset` (see §2). Tune `tokenizer.nproc` if needed.
2. **`input_type: single-cell`**: the pipeline builds `.loom` files under `data.loom_temp_dir` from each sample subfolder (10x `filtered_feature_bc_matrix` or matrix in the folder), then runs `TranscriptomeTokenizer`. With **`single_cell_settings.extract_metadata_from_path: true`**, `time` / `genotype` / `replicate` / `disease` / `sample_id` are filled from the folder name as in §2; adjust naming or the script if your labels differ.
3. **`input_type: loom`**: put `*.loom` files in `data.input_dir` and skip conversion; ensure loom attributes match the **keys** in `custom_attr_name_dict`.
4. **Run** (from repo root, repo mounted at `/app` in the container):

```bash
docker compose run --rm tokenize
```

The image supplies the **mouse** token dictionary (`MLM-re_token_dictionary_v1.pkl`; see [README](../README.md)) and [`TranscriptomeTokenizer`](../geneformer/tokenizer.py). Output is written under `data.output_dir`, typically **`{output_prefix}_0.dataset`** (suffix increments if the tokenizer splits large runs).

**Notebook or custom scripts:** you can call `TranscriptomeTokenizer` and `save_to_disk` yourself (same column requirements). Example layout: [`run_tokenizer_ad.py`](../run_tokenizer_ad.py).

---

## 4. Configure  [`config/isp.yaml`](../config/isp.yaml))


Align the file with your **dataset column names** and **string labels**.

| YAML area | What to set |
|-----------|-------------|
| `paths.dataset` | Path to your **`.dataset`** directory |
| `paths.geneformer_model` | Mouse Geneformer checkpoint directory |
| `paths.output_root` | Parent directory; each run writes under **`{output_root}/{YYYYMMDD}/run_<UTC start>/isp_results`** and **`.../ispstats_results`** by default (`paths.output_time_subdir: true`), so same-day reruns do not overwrite. Disable with `output_time_subdir: false`, `ISP_OUTPUT_TIME_SUBDIR=0`, or `--no-output-time-subdir` for a flat **`{output_root}/{YYYYMMDD}/...`** layout. Override the time folder with `paths.output_subdir` / `ISP_OUTPUT_SUBDIR` / `--output-subdir`. Date defaults to today; override with `run_isp.py --output-date` or `ISP_OUTPUT_DATE`. Legacy: `isp_results_dir` and `ispstats_results_dir` instead of `output_root`. |
| `perturbation.state_key` | **Exact** name of the metadata column for states |
| `perturbation.start_state` / `end_state` | **Exact** strings as they appear in that column |
| `perturbation.alt_states` | List of alternate end-state labels, or `[]` |
| `isp.filter_data` | Optional, e.g. `{"cell_type": ["skeletal_muscle"]}` — keys must exist in the dataset |
| `isp.max_ncells` | Cap on cells (after filters) |
| `runtime.forward_batch_size` / `nproc` | GPU batch size and CPU workers for `datasets` |

`model.type: Pretrained` is the usual choice for the stock mouse model; other options require matching **fine-tuned** checkpoints (see Geneformer docs).

---

## 5. Run in Docker

**Run ISP (Single GPU - Default):**
```bash
docker compose run --rm isp
```
  This invokes `accelerate launch ... --num_processes 1 /app/run_isp.py --config /app/config/isp.yaml`.

**Run ISP (Multiple GPUs):**
To scale generation across multiple GPUs, pass the `ISP_NUM_GPUS` environment variable. For example, to run on 4 GPUs:
```bash
ISP_NUM_GPUS=4 docker compose run --rm isp
```
  *Note: The script safely multiplexes the processing across all GPUs using Accelerate, and ensures that the final figures and statistics are only consolidated once on the main process to avoid any race conditions.*

Alternative with Jupyter container already up:
- Single GPU: `docker exec -it mouse_geneformer_container accelerate launch --num_processes 1 /app/run_isp.py --config /app/config/isp.yaml`
- Multi-GPU (e.g. 4 GPUs): `docker exec -it mouse_geneformer_container accelerate launch --num_processes 4 /app/run_isp.py --config /app/config/isp.yaml`

---

## 6. Outputs and downstream analysis

- **Intermediate / perturbation outputs:** `{paths.output_root}/{YYYYMMDD}/[run_<UTC>/]isp_results` (unless using legacy path keys; omit `run_<UTC>/` when `output_time_subdir` is false).
- **Stats (e.g. parquet):** `{paths.output_root}/{YYYYMMDD}/[run_<UTC>/]ispstats_results`.
- **Figures + table exports:** after stats, `run_isp.py` runs [`isp_analysis.py`](../isp_analysis.py) (same logic as [`isp_analysis.ipynb`](../isp_analysis.ipynb)): PNGs under **`{paths.output_root}/{YYYYMMDD}/[run_<UTC>/]figures/`** (e.g. `shift_distribution.png`, `volcano_plot.png`, `top_genes_barplot.png`, `waterfall_plot.png`); CSV summaries stay next to the parquet in `ispstats_results`. Disable with `analysis.enabled: false` in `isp.yaml` or `--skip-analysis`.
- **Run log + provenance (per run folder):** `isp_run.log` (rotating text log of stdout/stderr from `run_isp.py` on the **main** process only), plus `isp_config_used.yaml` and `isp_run_metadata.yaml`, under the same `run_<UTC>/` directory as results when time subdirs are enabled. Tokenization writes analogous files under `data.output_dir` (`tokenize_run.log`, `tokenize_config_used.yaml`, `tokenize_run_metadata.yaml`). See **[README.md § Run provenance and logs](../README.md#run-provenance-config-summary-and-rotating-logs-isp)** for behavior, rotation, and env vars (`ISP_LOG_*`, `TOKENIZE_LOG_*`, disable flags).

---

## Do I need a new `.dataset`?

| Situation | Action |
|-----------|--------|
| New tissue, genotype, or comparison | **Yes** — build or obtain a **tokenized** `.dataset` with correct `input_ids` and state metadata, then point `paths.dataset` at it. |
| Same dataset; only batch size / cell cap changes | Adjust `runtime.*` and/or `isp.max_ncells` only. |
| Same experiment; paths on disk changed | Update `paths.*` in `isp.yaml` only. |

---

## Hard requirements

1. **Same vocabulary** as training: mouse **pretrained** Geneformer + token dictionary used to build `input_ids`.
2. **Exact string match** between dataset labels and `start_state` / `end_state` / `alt_states` in YAML (including spaces and casing).

---

## Memory and throughput

- Prefer **letting PyTorch manage GPU memory** between batches. Avoid calling `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`, or aggressive `gc.collect()` in hot loops or after every pipeline stage: that can serialize CPU/GPU work and hurt the caching allocator, lowering GPU utilization.
- If a forward pass hits **CUDA OOM**, the ISP code may **empty the cache once and retry** that forward; if it still fails, reduce `runtime.forward_batch_size` or `isp.max_ncells` and rerun.
- **CPU `nproc`**: tune for your host (see `config/isp.yaml`); it only speeds dataset map/filter steps, not the transformer forwards.

---

## Flow summary

**Design → raw counts + metadata → tokenize → save `.dataset` → edit `isp.yaml` → `docker compose run --rm isp` → outputs → analysis.**
