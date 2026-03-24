
# Premise
Development is ongoing on DGX spark (ARM64).
Supposed to work on Inte;/AMD/ARM CPU with NVIDIA GPU.
All services will be run by docker compose. Not jupyter (major chang from original mouse-geneformer).


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


1. Settings **`config/isp.yaml`** (paths, perturbation, model, `runtime.forward_batch_size`, `runtime.nproc`, etc.). Edit that file on the host; it appears as `/app/config/isp.yaml` in the container.

2.
```bash
docker compose run --rm isp
```
This runs `accelerate launch --num_processes 1 /app/run_isp.py --config /app/config/isp.yaml`.


**Alternative:** if a long-lived container is already running (`docker compose up -d mouse-geneformer`):

```bash
docker exec -it mouse_geneformer_container accelerate launch --num_processes 1 /app/run_isp.py --config /app/config/isp.yaml
```

Outputs go under `paths.output_root` in `config/isp.yaml`, in **date-stamped** subfolders: `{output_root}/{YYYYMMDD}/isp_results`, `{output_root}/{YYYYMMDD}/ispstats_results`, and **`{output_root}/{YYYYMMDD}/figures/`** (PNG plots from `isp_analysis.py` after stats). Disable extra analysis with `analysis.enabled: false` or `--skip-analysis`. Default date: today (`--output-date` / `ISP_OUTPUT_DATE`). On the host this is under your repo bind mount (e.g. `./output/...`).





# Other types of Runs
## docker environment

Build the image once, then use Compose services (repo is mounted at `/app` in the container).

```bash
docker compose build mouse-geneformer
```

## Jupyter Lab (interactive)

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
