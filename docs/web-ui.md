# Mouse Geneformer WebUI (reference)

[Mouse-Geneformer-WebUI](https://github.com/YuyaSanaki/Mouse-Geneformer-WebUI) on GitHub.

The Web UI is documented in the main [README § Streamlit Web UI](../README.md#streamlit-web-ui). This page adds Compose and path details.

- App: [`streamlit_app/app.py`](../streamlit_app/app.py)
- Compose service: [`docker-compose.yml`](../docker-compose.yml) (`webui`)

---

## Start

```bash
docker compose build mouse-geneformer
docker compose up -d webui   # or: docker compose up webui
```

Open **http://localhost:8501**.

`WANDB_DISABLED=true` by default. For multi-GPU ISP from the UI, set **`ISP_NUM_GPUS`** in the environment or `.env`.

---

## Environment variables

| Variable | Meaning |
|----------|---------|
| `WEBUI_ROOT` | Repository root (Compose sets `/app`) |
| `WEBUI_WORKSPACE` | Uploads and per-run folders (default `{WEBUI_ROOT}/data/streamlit_workspace`) |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | Upload cap in megabytes (see `.streamlit/config.toml`) |

Per **Run job**:

`{WEBUI_WORKSPACE}/runs/<UTC>_<id>/config.yaml` and `console.log`

Uploads:

`{WEBUI_WORKSPACE}/uploads/<session>/<study_name>/`

After a successful **Pipeline (E2E)** or **ISP UMAP** job, the **Outputs** section offers **Download figures (.zip)** when figure files exist under `paths.output_root`.

---

## Guides per run type

| Run type | Doc |
|----------|-----|
| Pipeline (E2E) | [pipeline.md](pipeline.md) |
| ISP UMAP | [isp_umap.md](isp_umap.md) |

Tokenize, fine-tune, and standalone ISP: use CLI (`docker compose run --rm tokenize` / `finetune` / `isp`) — see [tokenization.md](tokenization.md), [fine-tuning.md](fine-tuning.md), [in-silico pertabation.md](in-silico%20pertabation.md).
