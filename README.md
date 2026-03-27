
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
clone rep
cd ti the dir
git lfs pull
docker build

MLM-re_token_dictionary_v1.pkl was missed from the repo. Download it from https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main/MLM-re_token_dictionary_v1.pkl

cd /home/yuya-sanaki/20260321Mouse-Geneformer/data/Mouse-Genecorpus-20M
git lfs pull

# Push results to local mac
scp /path/to/file yuyasanaki@192.168.200.102:/Users/yuyasanaki/desktop


# in-silico perturbation (docker compose service)
For the details, see [**docs/in-silico pertabation.md**](docs/in-silico%20pertabation.md) (end-to-end workflow for new biology, tokenized `.dataset`, and config).


1. Settings **[`config/isp.yaml`](../config/isp.yaml)** (paths, perturbation, model, `runtime.forward_batch_size`, `runtime.nproc`, etc.). Edit that file on the host; it appears as `/app/config/isp.yaml` in the container.

2.
```bash
docker compose run --rm isp
```
This runs `accelerate launch --num_processes 1 /app/run_isp.py --config /app/config/isp.yaml`.

Outputs go under `./output/<DATE>/isp_results`, `./output/<DATE>/ispstats_results`, and `./output/<DATE>/figures` when `paths.output_root` is set in `config/isp.yaml`. There will also be CSV tables next to the stats.

### Run provenance, config summary, and rotating logs (ISP)

On the **main process** only, `run_isp.py` prints a multi-line **config summary** (dataset path, model path, perturbation type, start/end states, `isp.max_ncells`, effective `forward_batch_size` / `nproc`, etc.) and mirrors **stdout** and **stderr** into a **rotating** log file next to that day’s outputs:

| Artifact | Location (with `paths.output_root` + date folder) | Role |
|----------|-----------------------------------------------------|------|
| `isp_config_used.yaml` | `./output/<DATE>/` | Copy of the ISP YAML used for this run |
| `isp_run_metadata.yaml` | `./output/<DATE>/` | `started_at_utc` at launch; **`finished_at_utc`** and **`run_status`** (`completed` / `failed`) appended when the process exits the perturbation+stats+analysis block (main process only) |
| `isp_run.log` (+ `.1`, `.2`, … on rotation) | `./output/<DATE>/` | Full console capture from Python onward (see note below) |

Optional environment variables:

| Variable | Meaning |
|----------|---------|
| `ISP_LOG_MAX_BYTES` | Max size of `isp_run.log` before rotation (default `52428800`, 50 MiB) |
| `ISP_LOG_BACKUP_COUNT` | How many rotated backups to keep (default `5`; `0` truncates instead of renaming) |
| `ISP_DISABLE_RUN_LOG` | Set to `1` / `true` / `yes` to skip file logging (console unchanged) |
| `ISP_GIT_COMMIT` | If set, written into `isp_run_metadata.yaml` when `git` is unavailable in the container |

**Note:** Messages printed by `accelerate launch` *before* `run_isp.py` starts (for example default-parameter hints) appear only in the terminal unless you redirect the Compose command at the shell level.

#### About the run log files (`*.log`)

- **What is captured:** Everything written to **stdout** and **stderr** from the moment the tee is installed — including ordinary `print` output, HuggingFace `datasets` progress bars, and other libraries that write to the console. This is a **text** log (UTF-8, with replacement for invalid bytes).
- **What is not captured:** Anything emitted **before** Python attaches the tee (notably `accelerate launch` startup lines). For a full shell transcript, run Compose with your own redirection (e.g. `docker compose run ... 2>&1 | tee full_terminal.log`).
- **ISP and multiple GPUs:** Only the **main** process writes `isp_run.log`. Other ranks still print to the terminal but do not append to that file, so the log stays a single coherent stream without interleaved corruption.
- **Append vs. rotate:** A new run on the **same** `<DATE>` **appends** to the existing `isp_run.log` / `tokenize_run.log`. When the file exceeds `*_LOG_MAX_BYTES`, it is rotated: the current file becomes `*.log.1`, the previous `.1` becomes `.2`, and so on, up to `*_LOG_BACKUP_COUNT`. With `*_LOG_BACKUP_COUNT=0`, the log is **truncated** when the size limit is hit instead of keeping numbered backups.
- **Provenance:** The YAML copies (`*_config_used.yaml`) and small metadata files (`*_run_metadata.yaml`) are the authoritative snapshot of **config**; the `.log` file is for **console history** and debugging long runs.

Implementation: [`run_isp.py`](run_isp.py), [`run_pipeline_log.py`](run_pipeline_log.py).

### Tokenization service (`docker compose run --rm tokenize`)

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
### 1. in silico pertubation jupyter notebook edit done
    docker compose up to open jupyterlab
    open in_silico_perturbation.ipynb
    edit the cell starting from "# in silico perturbation in deletion mode to determine genes whose. deletion in the dilated cardiomyopathy (dcm) state significantly shift. the embedding towards non-failing (nf) state"
    excute cells and the output ispstats_result will be made
    second analysis with isp_analysis.ipynb
