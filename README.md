
# Premise
DGX spark (ARM64) 
Docker env and use uv to install packages.

To adapt DGX spark GPU, many changes in requirements.txt.
optimize the Mouse-Geneformer pipeline for DGX Spark using Flash Attention 2, Hugging Face Accelerate, cuDF/cuML, and Parquet data formats.

# Install
clone rep
docker build

MLM-re_token_dictionary_v1.pkl was missed from the repo. Download it from https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main/MLM-re_token_dictionary_v1.pkl

cd /home/yuya-sanaki/20260321Mouse-Geneformer/data/Mouse-Genecorpus-20M
git lfs pull

# Push results to local mac
scp /path/to/file yuyasanaki@192.168.200.102:/Users/yuyasanaki/desktop

# Run juperter lab
docker compose up

# Run
1. in silico pertubation jupyter notebook edit done
    docker compose up to open jupyterlab
    open in_silico_perturbation.ipynb
    edit the cell starting from "# in silico perturbation in deletion mode to determine genes whose. deletion in the dilated cardiomyopathy (dcm) state significantly shift. the embedding towards non-failing (nf) state"
    excute cells and the output ispstats_result will be made
    second analysis with isp_analysis.ipynb
    
20260323 in silico pertubation acceleration (Single DGX Spark)
docker exec -it mouse_geneformer_container accelerate launch /app/run_isp.py --forward-batch-size 128

