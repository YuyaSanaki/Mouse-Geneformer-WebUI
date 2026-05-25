# Tokenization

Convert raw single-cell expression (10x, loom, or AnnData) into a Hugging Face **`.dataset`** directory for fine-tuning and ISP.

Entrypoint: [`execute_tokenizer_pipeline.py`](../execute_tokenizer_pipeline.py). Configuration: [`config/tokenize.yaml`](../config/tokenize.yaml). Docker: `docker compose run --rm tokenize` in [`docker-compose.yml`](../docker-compose.yml).

Install prerequisites (Docker image, `MLM-re_token_dictionary_v1.pkl`): [README § Install](../README.md#install).

---

## 1. Input layout (10x / single-cell)

You need **raw counts** (single-cell or single-nucleus; **do not** normalize before tokenization) for all conditions you will compare in ISP (e.g. `Ctrl` and `Disease`). Minimum ~500 cells per cell type of interest; more is better.

### 10x Cell Ranger / DRAGEN output

Each sample must have all three files:

- `barcodes.tsv.gz`
- `features.tsv.gz`
- `matrix.mtx.gz`

Place them in a folder named with three hyphen-separated segments:

```text
{study_root}/ExperimentName/Time-Condition-Replicate/
  barcodes.tsv.gz
  features.tsv.gz
  matrix.mtx.gz
```

Example: `1w-Disease-SingleCell/`, `1w-Ctrl-SingleCell/` under one study root.

| Field | Rule |
|-------|------|
| Time | First segment (e.g. `1w`, `3w`, `5w`) |
| Condition | Second segment (e.g. `Ctrl`, `Disease`) — used as ISP state when mapped to `disease` |
| Replicate | Third segment (e.g. `Rep1`, `SingleCell`; use a placeholder if only one replicate) |
| `sample_id` | Full folder name |

With **`single_cell_settings.extract_metadata_from_path: true`** in `config/tokenize.yaml`, `time`, `genotype`, `replicate`, `disease`, and `sample_id` are filled from folder names. Adjust naming or the script if your labels differ.

**`data.input_dir`** is the **study root** (parent of sample folders), e.g. `/app/data/my_study/` or `/app/data/ExperimentName/`. See [`data_input_layout.py`](../data_input_layout.py) for discovery rules (flat samples, nested `ExperimentName/`, multiple experiments under `/data/`).

### Alternative formats

| Format | Notes |
|--------|--------|
| **`.loom`** | Put one or more `*.loom` in `data.input_dir`; set `data.input_type: loom`. |
| **`.h5ad` (AnnData)** | Supported in Geneformer APIs with `file_format="h5ad"`; same field requirements as loom. |

### Required fields (loom / h5ad)

**Genes**

- **`ensembl_id`**: per gene; must match the **mouse** token dictionary.

**Cells**

- **`n_counts`**: total UMI/read counts per cell.

**Optional**

- **`filter_pass`**: if missing, all cells are tokenized.

**Metadata for ISP**

- Pass columns through **`tokenizer.custom_attr_name_dict`** (loom/h5ad name → dataset column name).
- Values must match **`perturbation.start_state` / `end_state`** in ISP YAML **exactly** (including casing).

### Expression assumptions

- **Raw counts** only; no prior feature selection.
- Median scaling and rank tokenization happen inside [`TranscriptomeTokenizer`](../geneformer/tokenizer.py).

---

## 2. Configure `config/tokenize.yaml`

| YAML area | What to set |
|-----------|-------------|
| `data.input_type` | `single-cell` (10x folders) or `loom` |
| `data.input_dir` | Study root (see §1) |
| `data.loom_temp_dir` | Where `.loom` files are written when converting from 10x |
| `data.output_dir` | Parent directory for the tokenized `.dataset` |
| `data.output_prefix` | Base name → `{output_prefix}_0.dataset` |
| `tokenizer.custom_attr_name_dict` | Loom/dataset column mapping |
| `tokenizer.nproc` / `max_cells` | Parallelism and cell cap |
| `single_cell_settings.extract_metadata_from_path` | Parse sample folder names (default `true`) |

**`input_type: single-cell`**: builds `.loom` under `data.loom_temp_dir`, then runs `TranscriptomeTokenizer`.

**`input_type: loom`**: reads `*.loom` from `data.input_dir`; loom attributes must match keys in `custom_attr_name_dict`.

The image uses the mouse token dictionary (`MLM-re_token_dictionary_v1.pkl`) from [README § Install](../README.md#install).

Custom scripts: [`run_tokenizer_ad.py`](../run_tokenizer_ad.py) shows an alternate layout.

---

## 3. Run

```bash
docker compose run --rm tokenize
```

Override config:

```bash
TOKENIZE_CONFIG=/app/config/my_tokenize.yaml docker compose run --rm tokenize
```

**Streamlit:** run type **Tokenize**, or **Pipeline (E2E)** (tokenize is stage 1). See [README § Streamlit Web UI](../README.md#streamlit-web-ui).

---

## 4. Output `.dataset`

Saved under `data.output_dir`, typically **`{output_prefix}_0.dataset`** (suffix increments if the run is split).

| Column / field | Role |
|----------------|------|
| `input_ids` | Rank-encoded gene tokens per cell |
| `length` | Sequence length |
| State column | e.g. `disease` — must match ISP `perturbation.state_key` and state strings |
| Other metadata | e.g. `cell_type`, `time`, `sample_id` if mapped in `custom_attr_name_dict` |

Point **`paths.dataset`** in [`config/finetune.yaml`](../config/finetune.yaml) or [`config/isp.yaml`](../config/isp.yaml) at this directory.

---

## 5. Run provenance and logs

[`execute_tokenizer_pipeline.py`](../execute_tokenizer_pipeline.py) prints a **config summary**, then runs conversion (if `input_type: single-cell`) and tokenization.

| Artifact | Location | Role |
|----------|----------|------|
| `tokenize_config_used.yaml` | `data.output_dir` | Copy of the tokenize config used |
| `tokenize_run_metadata.yaml` | same | `started_at_utc`; **`finished_at_utc`** and **`run_status`** when the run ends |
| `tokenize_run.log` (+ `.1`, `.2`, …) | same | Rotating mirror of stdout/stderr |

| Variable | Meaning |
|----------|---------|
| `TOKENIZE_CONFIG` | Override config path |
| `TOKENIZE_LOG_MAX_BYTES` | Max log size before rotation (default 50 MiB) |
| `TOKENIZE_LOG_BACKUP_COUNT` | Rotated backups to keep (default `5`; `0` truncates) |
| `TOKENIZE_DISABLE_RUN_LOG` | Set to `1` / `true` / `yes` to skip file logging |
| `TOKENIZE_GIT_COMMIT` | Optional commit string when `git` is unavailable in the container |

The tee is attached after the output directory exists, so **loom conversion and tokenization** are both recorded. Shared behavior (what is captured, rotation, append vs. new folder): [in-silico pertabation.md § Run provenance and logs](in-silico%20pertabation.md#run-provenance-config-summary-and-rotating-logs).

Implementation: [`execute_tokenizer_pipeline.py`](../execute_tokenizer_pipeline.py), [`run_pipeline_log.py`](../run_pipeline_log.py).

---

## 6. Flow summary

**Raw counts + metadata → `config/tokenize.yaml` → `docker compose run --rm tokenize` → `{output_prefix}_0.dataset` → fine-tune or ISP.**
