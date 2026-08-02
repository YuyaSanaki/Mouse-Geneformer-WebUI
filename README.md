# Mouse Geneformer WebUI

This is a refactored clone of [mouse-Geneformer](https://github.com/machine-perception-robotics-group/Mouse-Geneformer). It runs on an NVIDIA GPU workstation or server with Docker. All services use Docker Compose (the upstream repo used Jupyter). You can run jobs from the **Web UI** or the **CLI**.

**Monorepo:** `core/` (Geneformer + CLI), `webui/` (Streamlit), `contracts/` (shared input layout). WebUI talks to core only via **subprocess + YAML**. Layout and contract: [docs/architecture.md](docs/architecture.md). This GitHub repo is the public surface of that monorepo, not a separate diverging codebase.



## Requirements

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA GPU with drivers installed |
| **Container runtime** | Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| **CPU / OS** | Linux **x86\_64** or **aarch64 / ARM64** (e.g. DGX Spark) |

Primary testing is on **DGX Spark (aarch64)**; x86\_64 NVIDIA hosts are supported. Docker selects the matching image architecture.

**Not supported:** CPU-only, non-NVIDIA GPUs, or macOS GPU.

<a id="install"></a>

## Install

1. **Clone** and enter the repository:
   ```bash
   git clone https://github.com/YuyaSanaki/Mouse-Geneformer-WebUI
   cd Mouse-Geneformer-WebUI
   ```

2. **Git Large Files:**
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Build the image** (Streamlit, ISP, tokenize, pipeline, fine-tune):
   ```bash
   docker compose build mouse-geneformer-webui
   ```
   Re-run after `Dockerfile` or dependency changes.

4. **Mouse-Genecorpus-20M:**
   ```bash
   cd data/Mouse-Genecorpus-20M
   git lfs pull
   cd ../..
   ```

5. **Token dictionaries** under `core/geneformer/dicts/` (required to import `geneformer` / run any pipeline job). If missing after LFS pull, download at least:
   - [MLM-re_token_dictionary_v1.pkl](https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main/MLM-re_token_dictionary_v1.pkl)
   - [MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl](https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl)
   ```bash
   mkdir -p core/geneformer/dicts
   wget -O core/geneformer/dicts/MLM-re_token_dictionary_v1.pkl \
     'https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main/MLM-re_token_dictionary_v1.pkl'
   wget -O core/geneformer/dicts/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl \
     'https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl'
   ```
   Also place `mouse_gene_median_dictionary.pkl` in the same folder when tokenizing (from the Mouse-Geneformer / Mouse-Genecorpus assets).

6. **Pretrained model** at `models/mouse-Geneformer/` (`config.json`, `pytorch_model.bin`). Required for fine-tune / E2E pipeline. Copy from an existing Mouse-Geneformer checkout, or download from the project’s Hugging Face / release assets into that directory.

<a id="streamlit-web-ui"></a>

## Quick start (Web UI — Pipeline E2E)

After [Install](#install), run the full **Tokenize → Fine-tune → ISP** pipeline from the browser (same as `core/run_pipeline.py` on the CLI).

```bash
docker compose up -d webui
```

Open **http://localhost:8501** ([Mouse-Geneformer-WebUI](https://github.com/YuyaSanaki/Mouse-Geneformer-WebUI)).

On a **remote GPU server**, `localhost` in your laptop browser does not reach the container. Use **SSH port forwarding** (keep the session open):

```bash
ssh -L 8501:localhost:8501 <user>@<server>
```

Then open **http://localhost:8501** locally. Or use the server LAN/Tailscale IP (e.g. `http://<server-ip>:8501`) if your network allows it. See [docs/web-ui.md](docs/web-ui.md).

1. **Study name** — your experiment name (e.g. `MyExperiment`; set **before** uploading the zip).
2. **Upload** — **data.zip** with sample folders (`Time-State-Suffix/`, e.g. `1w-Ctrl-SingleCell/`, `1w-Disease-SingleCell/`), each with `barcodes.tsv.gz`, `features.tsv.gz`, and `matrix.tsv.gz`.
3. **Run type** — **Pipeline (E2E)**.
4. **ISP states** — pick **start_state** / **end_state** (e.g. AD, WT) → **Apply setting to Config YAML**. To test specific genes instead of genome-wide ISP, set `perturbation.genes_to_perturb` in the YAML editor (mouse symbols e.g. `Ece1`, `Igfbp2`, or Ensembl IDs).
5. **GPU batch size** — leave on **Auto** to measure this GPU at ISP startup, or switch to **Manual** to pin `runtime.forward_batch_size`.
6. **Run job** — one E2E job at a time; follow **Logs & status** and **Outputs**. When finished, use **Download pipeline run (.zip)** to get that run’s checkpoints, figures, ISP outputs, logs, and configs.

Workflow and YAML fields: [docs/pipeline.md](docs/pipeline.md). Default ISP runs all genes (can take ~30 hours on DGX Spark). Use `genes_to_perturb: [Ece1]` (or Ensembl IDs) for a single-gene test run.

More detail: [docs/web-ui.md](docs/web-ui.md).

![Mouse Geneformer WebUI — study upload, Pipeline (E2E), ISP states, and run directory](docs/webui.png)
---

## CLI

**End-to-end** — edit [`core/config/pipeline.yaml`](core/config/pipeline.yaml), then:

```bash
docker compose run --rm pipeline
```

**Standalone ISP** — edit [`core/config/isp.yaml`](core/config/isp.yaml). Set `perturbation.genes_to_perturb` to mouse gene symbols (e.g. `[Ece1]`, `[Igfbp2]`) or Ensembl IDs; leave empty `[]` for genome-wide ISP:

```bash
docker compose run --rm isp
```

| Step | Command | Doc |
|------|---------|-----|
| Tokenize | `docker compose run --rm tokenize` | [tokenization.md](docs/tokenization.md) |
| Fine-tune | `docker compose run --rm finetune` | [fine-tuning.md](docs/fine-tuning.md) |
| ISP | `docker compose run --rm isp` | [in-silico pertabation.md](docs/in-silico%20pertabation.md) |
| ISP UMAP | `docker compose run --rm isp_umap` | [isp_umap.md](docs/isp_umap.md) |

---

## Documentation

| Topic | Guide |
|-------|--------|
| **Architecture / boundaries** | [docs/architecture.md](docs/architecture.md) |
| **Tokenization** | [docs/tokenization.md](docs/tokenization.md) |
| **Fine-tuning** | [docs/fine-tuning.md](docs/fine-tuning.md) |
| **ISP** | [docs/in-silico pertabation.md](docs/in-silico%20pertabation.md) |
| **E2E pipeline** | [docs/pipeline.md](docs/pipeline.md) |
| **ISP UMAP** | [docs/isp_umap.md](docs/isp_umap.md) |
| **Web UI (details)** | [docs/web-ui.md](docs/web-ui.md) |
