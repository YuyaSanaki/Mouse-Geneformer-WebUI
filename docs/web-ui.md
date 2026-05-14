# Streamlit Web UI

The **web UI** is a [Streamlit](https://streamlit.io/) control panel that runs inside the same Docker image as the CLI pipelines. Jobs you start in the browser are **subprocesses** in the container (`accelerate launch … run_isp.py`, `execute_tokenizer_pipeline.py`, etc.), so behavior matches the Compose one-shot services.

- App entrypoint: [`streamlit_app/app.py`](../streamlit_app/app.py)
- Compose service: [`docker-compose.yml`](../docker-compose.yml) (`webui`)

---

## 1. Prerequisites

Follow the main [README prerequisites](../README.md#requirements): NVIDIA GPU, Docker with NVIDIA Container Toolkit, and a built image:

```bash
docker compose build mouse-geneformer
```

Paths in the YAML editor are **container paths** (e.g. `/app/config/isp.yaml` templates are copied from the repo; per-run configs are written under the workspace — see below).

---

## 2. Start the UI

Foreground (logs in the terminal):

```bash
docker compose up webui
```

Detached:

```bash
docker compose up -d webui
```

Open **http://localhost:8501** (port mapped in Compose).

The service sets `WANDB_DISABLED=true` by default. Multi-GPU ISP from the UI uses the same variable as the CLI service: set **`ISP_NUM_GPUS`** in the environment (or `.env` next to `docker-compose.yml`) if you need more than one process for `accelerate launch`.

---

## 3. Layout

| Area | Purpose |
|------|---------|
| **Data upload** | Saves files under the workspace `uploads/<session>/`. For tokenization, you can upload a `.zip` of a `single-cell` tree; zips are extracted safely into a sibling folder. Point `data.input_dir` (or loom paths) in your YAML at that folder. |
| **Run type** | **ISP**, **Tokenize**, **Fine-tune**, or **ISP UMAP**. Switching type reloads the default template from `config/isp.yaml`, `config/tokenize.yaml`, `config/finetune.yaml`, or `config/isp_umap.yaml`. |
| **Run configuration** | Editable YAML. Use **Reset YAML to template on disk** to discard edits and reload the file from the repo. |
| **Execute** | **Run job** writes the YAML to a new run folder, then starts the matching command. While a job runs, only one job is allowed at a time. |
| **Logs & status** | Tails `console.log` for the active or last run; shows exit code when finished. |
| **Outputs** | Lists guessed output roots from the YAML (e.g. `paths.output_root`, tokenize `data.output_dir`) and shows recent subdirectories. Per-run **config.yaml** and **console.log** are downloadable from the last run folder. |

---

## 4. Workspace and environment variables

| Variable | Meaning |
|----------|---------|
| **`WEBUI_ROOT`** | Repository root inside the container. Default: current working directory (`/app` when using Compose). |
| **`WEBUI_WORKSPACE`** | Base directory for uploads and per-run folders. Default: `{WEBUI_ROOT}/data/streamlit_workspace`. |

Each **Run job** creates:

`{WEBUI_WORKSPACE}/runs/<UTC timestamp>_<short id>/`

containing `config.yaml` (the YAML you edited) and `console.log` (stdout/stderr of the subprocess).

Uploads go under:

`{WEBUI_WORKSPACE}/uploads/<session id>/`

---

## 5. Deeper guides per pipeline

The UI does not replace the YAML reference; use these docs for end-to-end workflows and field meanings:

| Run type | Doc |
|----------|-----|
| ISP | [in-silico pertabation.md](in-silico%20pertabation.md) |
| Tokenize | [in-silico pertabation.md](in-silico%20pertabation.md) (dataset / tokenization) · [README](../README.md) (section *Tokenization service*) |
| Fine-tune | [fine-tuning.md](fine-tuning.md) |
| ISP UMAP | [isp_umap.md](isp_umap.md) |

---

## 6. Jupyter vs Web UI

**Jupyter Lab** (`mouse-geneformer` service, port 8888) is for interactive notebooks. The **Web UI** is for running the same scripted entrypoints with edited YAML and a simple log tail — useful when you prefer a browser panel over `docker compose run` in a shell.
