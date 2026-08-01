# Project Instructions

## Monorepo layout
- `core/` — geneformer library, CLI runners (`run_*.py`), `core/config/*.yaml`
- `webui/` — Streamlit only; jobs via **subprocess + YAML** (see `docs/architecture.md`)
- `contracts/` — shared layout rules (`data_input_layout.py`); no geneformer imports
- Public GitHub Mouse-Geneformer-WebUI is this same monorepo (mirror/release surface), not a divergent codebase

## Runtime Environment
- All executions (tokenization, ISP, etc.) MUST be performed using `docker compose` services to ensure consistent dependencies and environment variables.
- The project root is mapped to `/app` inside the containers. Use `/app/...` paths in configuration files.
- `PYTHONPATH=/app/core:/app/contracts:/app/webui`

## Configuration
- Use YAML files in `core/config/` to manage parameters for different tasks.
- **Tokenization:** `core/config/tokenize.yaml` → `docker compose run --rm tokenize`
- **In Silico Perturbation (ISP):** `core/config/isp.yaml` → `docker compose run --rm isp` (or `accelerate launch … /app/core/run_isp.py --config /app/core/config/your_config.yaml`)
- **Fine-Tuning:** `core/config/finetune.yaml` → `docker compose run --rm finetune`. Fine-tuned models can be used by ISP with `model.type: CellClassifier` and `paths.geneformer_model` pointing at the checkpoint.
- **E2E pipeline:** `core/config/pipeline.yaml` → `docker compose run --rm pipeline`

## Data Structure
- Input single-cell data is located under `data/` (e.g. `data/MyStudy/` with `1w-Ctrl-SingleCell/`, `1w-Disease-SingleCell/`).
- Tokenized datasets are stored in `data/*/tokenized_dataset/` (standalone) or under pipeline run dirs.
- ISP results are written to `output/YYYYMMDD/`.
- Fine-tuned models are written to `output/YYYYMMDD/finetune_*/` or pipeline run folders.
