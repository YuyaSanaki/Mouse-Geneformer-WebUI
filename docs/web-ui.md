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

### Remote server access

If Docker runs on a **remote GPU host** (not your laptop), `localhost` in the browser refers to your local machine. Use one of:

| Method | How |
|--------|-----|
| **SSH port forwarding** | `ssh -L 8501:localhost:8501 <user>@<server>` — then open `http://localhost:8501` locally |
| **Direct IP** | `http://<server-lan-or-tailscale-ip>:8501` if the port is reachable on your network |
| **Cursor Remote** | Forward port **8501** in the IDE Ports panel |

Verify on the server: `curl -I http://localhost:8501` should return HTTP 200 when the container is up.

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

After a successful **Pipeline (E2E)** job, the **Outputs** section offers **Download pipeline run (.zip)** for that run only (`pipeline_<UTC>/`: finetune checkpoints, ISP outputs, figures, tokenized dataset, logs, stage configs). **ISP UMAP** still offers **Download figures (.zip)**.

---

## Guides per run type

| Run type | Doc |
|----------|-----|
| Pipeline (E2E) | [pipeline.md](pipeline.md) |
| ISP UMAP | [isp_umap.md](isp_umap.md) |

**Pipeline (E2E)** — set `perturbation.genes_to_perturb` in the Config YAML editor (mouse symbols e.g. `Ece1`, `Igfbp2`, or Ensembl IDs; empty `[]` = genome-wide ISP). See [pipeline.md § perturbation](pipeline.md#configure-configpipelineyaml).

Tokenize, fine-tune, and standalone ISP: use CLI (`docker compose run --rm tokenize` / `finetune` / `isp`) — see [tokenization.md](tokenization.md), [fine-tuning.md](fine-tuning.md), [in-silico pertabation.md](in-silico%20pertabation.md).
