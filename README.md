
# Premise
All services run via Docker Compose (diverges from the original Mouse-Geneformer repo which used plain Jupyter).

gemini --resume 76282c30-8ad3-4fdf-a65c-dcd411190261   


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

Outputs go under ./output/DATE/isp_results and ./output/DATE/isp_results. There will also be simple analysis figures.



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
