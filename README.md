
# Premise
All services run via Docker Compose (diverges from the original Mouse-Geneformer repo which used plain Jupyter).

## Requirements

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA GPU (any) with drivers installed |
| **Container runtime** | Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`nvidia-container-toolkit`) |
| **CPU / OS** | Linux on **x86\_64** (Intel or AMD) or **aarch64 / ARM64** (e.g. DGX Spark, Grace Hopper) |

Code modification and primary testing is on a **DGX Spark (aarch64)**. x86\_64 NVIDIA hosts (desktop RTX, server A/H-series, etc.) are also supported — the image arch is chosen automatically by Docker.

**Not supported:** CPU-only, non-NVIDIA GPUs (AMD ROCm, Intel Arc), or macOS GPU.


# Install

1. **Clone the repository** and enter it:
   ```bash
   git clone <repository-url>
   cd Mouse-Geneformer   # or your clone directory name
   ```

2. **Git LFS** (large files tracked by LFS):
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Build the Docker image** used by Jupyter, Streamlit web UI, ISP, and tokenization (same image for all Compose services):
   ```bash
   docker compose build mouse-geneformer
   ```
   The first build can take a while (NGC PyTorch base + Python deps). Re-run this after `Dockerfile` or dependency changes.

4. **Mouse-Genecorpus-20M data:** From the repo root, fetch LFS objects for the corpus checkout:
   ```bash
   cd data/Mouse-Genecorpus-20M
   git lfs pull
   cd ../..
   ```

5. **`MLM-re_token_dictionary_v1.pkl`:** This file is not always present after LFS pull. If it is missing, download it into `data/Mouse-Genecorpus-20M/` (same directory as the rest of that dataset layout):
   - Direct link: [MLM-re_token_dictionary_v1.pkl](https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main/MLM-re_token_dictionary_v1.pkl)
   - Example: `wget -O data/Mouse-Genecorpus-20M/MLM-re_token_dictionary_v1.pkl 'https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main/MLM-re_token_dictionary_v1.pkl'`



# in-silico perturbation (docker compose service)
For the details, see [**docs/in-silico pertabation.md**](docs/in-silico%20pertabation.md) (end-to-end workflow for new biology, tokenized `.dataset`, and config).


1. Settings **[`config/isp.yaml`](../config/isp.yaml)** (paths, perturbation, model, `runtime.forward_batch_size`, `runtime.nproc`, etc.). Edit that file on the host; it appears as `/app/config/isp.yaml` in the container.

2.
```bash
docker compose run --rm isp
```
This runs `accelerate launch --num_processes 1 /app/run_isp.py --config /app/config/isp.yaml`.
If you want to run multiple consecutively, ` docker compose run --rm isp accelerate launch --num_processes 1 /app/run_isp.py --config /app/config/isp_WT-AD.yaml && docker compose run --rm isp accelerate launch --num_processes 1 /app/run_isp.py --config /app/config/isp_AD-WT.yaml`.

Outputs go under `./output/<DATE>/isp_<UTC time>/isp_results` (default), `./output/<DATE>/isp_<UTC time>/ispstats_results`, and `./output/<DATE>/isp_<UTC time>/figures` when `paths.output_root` and `paths.output_time_subdir` are true in `config/isp.yaml`. Use `paths.output_time_subdir: false`, `ISP_OUTPUT_TIME_SUBDIR=0`, or `--no-output-time-subdir` for the previous flat layout under `<DATE>/`. There will also be CSV tables next to the stats.

### Run provenance, config summary, and rotating logs (ISP)

On the **main process** only, `run_isp.py` prints a multi-line **config summary** (dataset path, model path, perturbation type, start/end states, `isp.max_ncells`, effective `forward_batch_size` / `nproc`, etc.) and mirrors **stdout** and **stderr** into a **rotating** log file next to that day’s outputs:

| Artifact | Location (with `paths.output_root` + date + default run folder) | Role |
|----------|-----------------------------------------------------|------|
| `isp_config_used.yaml` | `./output/<DATE>/isp_<UTC>/` | Copy of the ISP YAML used for this run |
| `isp_run_metadata.yaml` | `./output/<DATE>/isp_<UTC>/` | `started_at_utc` at launch; **`finished_at_utc`** and **`run_status`** (`completed` / `failed`) appended when the process exits the perturbation+stats+analysis block (main process only) |
| `isp_run.log` (+ `.1`, `.2`, … on rotation) | `./output/<DATE>/isp_<UTC>/` | Full console capture from Python onward (see note below) |

Optional environment variables:

| Variable | Meaning |
|----------|---------|
| `ISP_LOG_MAX_BYTES` | Max size of `isp_run.log` before rotation (default `52428800`, 50 MiB) |
| `ISP_LOG_BACKUP_COUNT` | How many rotated backups to keep (default `5`; `0` truncates instead of renaming) |
| `ISP_DISABLE_RUN_LOG` | Set to `1` / `true` / `yes` to skip file logging (console unchanged) |
| `ISP_GIT_COMMIT` | If set, written into `isp_run_metadata.yaml` when `git` is unavailable in the container |
| `ISP_OUTPUT_TIME_SUBDIR` | `1` / `true` / `yes` (default follows YAML) or `0` / `false` / `no` to disable the automatic `isp_<UTC time>` folder under each `<DATE>` |
| `ISP_OUTPUT_SUBDIR` | Fixed folder name under `<DATE>` (overrides the time-based folder); must be a single path segment |

**Note:** Messages printed by `accelerate launch` *before* `run_isp.py` starts (for example default-parameter hints) appear only in the terminal unless you redirect the Compose command at the shell level.

#### About the run log files (`*.log`)

- **What is captured:** Everything written to **stdout** and **stderr** from the moment the tee is installed — including ordinary `print` output, HuggingFace `datasets` progress bars, and other libraries that write to the console. This is a **text** log (UTF-8, with replacement for invalid bytes).
- **What is not captured:** Anything emitted **before** Python attaches the tee (notably `accelerate launch` startup lines). For a full shell transcript, run Compose with your own redirection (e.g. `docker compose run ... 2>&1 | tee full_terminal.log`).
- **ISP and multiple GPUs:** Only the **main** process writes `isp_run.log`. Other ranks still print to the terminal but do not append to that file, so the log stays a single coherent stream without interleaved corruption.
- **Append vs. rotate:** A new run that reuses the **same** output directory (same `<DATE>` and same run folder, or flat layout under `<DATE>`) **appends** to the existing `isp_run.log` / `tokenize_run.log`. Default ISP layout uses a new `isp_<UTC time>` folder per launch, so logs normally do not mix across runs. When the file exceeds `*_LOG_MAX_BYTES`, it is rotated: the current file becomes `*.log.1`, the previous `.1` becomes `.2`, and so on, up to `*_LOG_BACKUP_COUNT`. With `*_LOG_BACKUP_COUNT=0`, the log is **truncated** when the size limit is hit instead of keeping numbered backups.
- **Provenance:** The YAML copies (`*_config_used.yaml`) and small metadata files (`*_run_metadata.yaml`) are the authoritative snapshot of **config**; the `.log` file is for **console history** and debugging long runs.

Implementation: [`run_isp.py`](run_isp.py), [`run_pipeline_log.py`](run_pipeline_log.py).

---
# Tokenization service (`docker compose run --rm tokenize`)

[`execute_tokenizer_pipeline.py`](execute_tokenizer_pipeline.py) reads [`config/tokenize.yaml`](config/tokenize.yaml) (override path with env `TOKENIZE_CONFIG`). It prints a **config summary**, then runs conversion (if `input_type: single-cell`) and `TranscriptomeTokenizer`.

| Artifact | Location | Role |
|----------|----------|------|
| `tokenize_config_used.yaml` | `data.output_dir` from YAML | Copy of the tokenize config used |
| `tokenize_run_metadata.yaml` | same | `started_at_utc` at provenance write; **`finished_at_utc`** and **`run_status`** when tokenization finishes or fails |
| `tokenize_run.log` (+ rotations) | same | Rotating mirror of stdout/stderr for the whole pipeline |

| Variable | Meaning |
|----------|---------|
| `TOKENIZE_LOG_MAX_BYTES` | Same idea as ISP (default 50 MiB) |
| `TOKENIZE_LOG_BACKUP_COUNT` | Same idea as ISP (default `5`) |
| `TOKENIZE_DISABLE_RUN_LOG` | Disable tee to `tokenize_run.log` |
| `TOKENIZE_GIT_COMMIT` | Optional commit string for metadata when `git` is missing |

The **tokenize** run log follows the same rules as in [About the run log files](#about-the-run-log-files-log) above (tee after the output directory exists, so conversion + tokenization are both recorded).
---

# Fine-tuning service (`docker compose run --rm finetune`)

For the details, see [**docs/fine-tuning.md**](docs/fine-tuning.md) (end-to-end workflow for fine-tuning Geneformer on cell or disease classification).

[`run_finetune.py`](run_finetune.py) reads [`config/finetune.yaml`](config/finetune.yaml) (override path with env `FINETUNE_CONFIG` or `--config`). It prints a **config summary**, applies metadata injection, then trains a `BertForSequenceClassification` model using the HuggingFace `Trainer`.

1. **Configure** [`config/finetune.yaml`](config/finetune.yaml):

| YAML area | What to set |
|-----------|-------------|
| `paths.dataset` | Tokenized `.dataset` directory |
| `paths.geneformer_model` | Pretrained model to fine-tune from |
| `paths.output_root` | Output root; writes to `{output_root}/{YYYYMMDD}/finetune_<UTC>/` |
| `metadata.add_columns` | Add missing columns (e.g. `{organ_major: brain}`) |
| `metadata.rename_columns` | Rename columns (e.g. `{genotype: cell_type}`) |
| `finetune.task_type` | `disease` or `cell_type` |
| `finetune.label_column` | Which column becomes the classification label |
| `finetune.organ_key` | Group by this column, or `null` for full dataset |
| `training.*` | Learning rate, batch size, epochs, scheduler, etc. |
| `umap.enabled` | Generate UMAP visualization (default: `true`) |

2. **Run:**
```bash
docker compose run --rm finetune
```

3. **Use with ISP** — the script prints the checkpoint path at the end. Update your `isp.yaml`:
```yaml
paths:
  geneformer_model: /app/output/YYYYMMDD/finetune_.../all_run1/
model:
  type: CellClassifier
  num_classes: 2
```

| Artifact | Location | Role |
|----------|----------|------|
| `finetune_config_used.yaml` | `output/{DATE}/finetune_{UTC}/` | Copy of config used |
| `finetune_run_metadata.yaml` | same | Timestamps, status, saved model paths |
| `finetune_run.log` (+ rotations) | same | Rotating stdout/stderr mirror |
| `{group}_run{N}/` | same | Model checkpoint, `results.csv`, `label_dict.json`, `preds.pkl` |
| `figures/` | same | UMAP PDFs (if enabled) |

| Variable | Meaning |
|----------|---------|
| `FINETUNE_CONFIG` | Override config path (default `/app/config/finetune.yaml`) |
| `FINETUNE_LOG_MAX_BYTES` | Max log size before rotation (default 50 MiB) |
| `FINETUNE_LOG_BACKUP_COUNT` | Rotated backups to keep (default `5`) |
| `FINETUNE_DISABLE_RUN_LOG` | Set to `1` to skip file logging |
| `WANDB_DISABLED` | Set to `true` (default in Compose) to disable W&B |

Implementation: [`run_finetune.py`](run_finetune.py), [`config/finetune.yaml`](config/finetune.yaml).

---

# ISP UMAP Visualization service (`docker compose run --rm isp_umap`)

For the details, see [**docs/isp_umap.md**](docs/isp_umap.md) (end-to-end info about the visual trajectory analysis generated).

[`run_isp_umap.py`](run_isp_umap.py) reads [`config/isp_umap.yaml`](config/isp_umap.yaml). It leverages your tokenized dataset and fine-tuned model to simulate an in-silico gene perturbation (e.g. `Igfbp2`) locally, writes **`per_cell_isp_shift.csv`** (how much each start-state cell moved), and plots embedding trajectories via UMAP.

1. **Configure** [`config/isp_umap.yaml`](config/isp_umap.yaml):
    - Set the `gene_to_perturb` (supports symbols like `Igfbp2` or explicit Ensembl IDs)
    - Specify `start_state` (original disease mode) and `end_state` (target healthy mode)
2. **Run:**
```bash
docker compose run --rm isp_umap
```

| Artifact | Location | Role |
|----------|----------|------|
| `per_cell_isp_shift.csv` | `output/{DATE}/isp_umap_{UTC}/` | Per-cell embedding/UMAP shift after ISP (see [docs/isp_umap.md](docs/isp_umap.md)) |
| `umap_*.png` | `output/{DATE}/isp_umap_{UTC}/` | Seaborn trajectory visualization graphic |
| `[STATE]_embs.npy` | same | Raw extracted intermediate embeddings arrays exported for standalone notebooks |

# Other types of Runs
## docker container

Build the image once, then use Compose services (repo is mounted at `/app` in the container).

```bash
docker compose build mouse-geneformer
```

## Jupyter Lab (interactive)
juoyter notebooks were not fully edited to match docker compose services.

```bash
docker compose up -d mouse-geneformer
# Jupyter: http://localhost:8888
```

## Streamlit Web UI (same container image as CLI)

For layout, workspace paths, uploads, and environment variables, see [**docs/web-ui.md**](docs/web-ui.md).

The `webui` service runs [Streamlit](https://streamlit.io/) in the **same** `mouse-geneformer` image. Jobs started from the browser execute as **subprocesses** in that container (`accelerate launch … run_isp.py`, `execute_tokenizer_pipeline.py`, etc.), matching the CLI entrypoints.

```bash
docker compose build mouse-geneformer
docker compose up -d webui
# UI: http://localhost:8501
```

Uploads and per-run configs/logs are written under `data/streamlit_workspace/` (ignored when `data/` is not tracked). Set `WEBUI_ROOT` or `WEBUI_WORKSPACE` in the service environment if you need non-default paths.

Large uploads: Streamlit’s default 200 MB cap is raised via [`.streamlit/config.toml`](.streamlit/config.toml) and the `webui` service env `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` (megabytes). For multi–hundred GB or TB-scale data, prefer placing files under the mounted repo (e.g. `data/…`) and referencing those paths in YAML instead of browser upload.

### 1. in silico pertubation jupyter notebook edit done
    docker compose up to open jupyterlab
    open in_silico_perturbation.ipynb
    edit the cell starting from "# in silico perturbation in deletion mode to determine genes whose. deletion in the dilated cardiomyopathy (dcm) state significantly shift. the embedding towards non-failing (nf) state"
    excute cells and the output ispstats_result will be made
    second analysis with isp_analysis.ipynb
