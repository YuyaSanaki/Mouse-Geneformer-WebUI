# Project Instructions

## Runtime Environment
- All executions (tokenization, ISP, etc.) MUST be performed using `docker compose` services to ensure consistent dependencies and environment variables.
- The project root is mapped to `/app` inside the containers. Use `/app/...` paths in configuration files.

## Configuration
- Use YAML files in the `config/` directory to manage parameters for different tasks.
- **Tokenization:** Uses `config/tokenize.yaml`. Execute via `docker compose run --rm tokenize`.
- **In Silico Perturbation (ISP):** Uses `config/isp.yaml` (or specialized versions like `config/isp_1w_WT_AD.yaml`). Execute via `docker compose run --rm isp accelerate launch --num_processes 1 /app/run_isp.py --config /app/config/your_config.yaml`.

## Data Structure
- Input single-cell data is located in `data/AD/`.
- Tokenized datasets are stored in `data/AD/tokenized_dataset/`.
- ISP results are written to `output/YYYYMMDD/`.
