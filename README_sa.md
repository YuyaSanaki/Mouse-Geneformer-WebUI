
# Premise
DGX spark (ARM64) 
Docker env and use uv to install packages.

To adapt DGX spark GPU, many changes in requirements.txt.

# Install
clone rep
docker build

# Run juperter lab
docker compose up

MLM-re_token_dictionary_v1.pkl was missed from the repo. Download it from https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main/MLM-re_token_dictionary_v1.pkl

cd /home/yuya-sanaki/20260321Mouse-Geneformer/data/Mouse-Genecorpus-20M
git lfs pull

changed 
# load disease dataset (xxx.dataset)
dataset_name = "/app/data/Mouse-Genecorpus-20M/eval_dataset/in_silico_perturbation/Cop1KO_isp_mouse_tokenize_dataset_v-n1.dataset"

#18
changed


# "delete": delete gene from rank value encoding
# "overexpress": move gene to front of rank value encoding
# "inhibit": move gene to lower quartile of rank value encoding
# "activate": move gene to higher quartile of rank value encoding


select_perturb_type = "delete"

start_state = "Cop1_WT"          # <-- change this
end_state = "Cop1_KO"            # <-- change this
alt_state = []

use_model_type = "Pretrained"    # <-- change this (you have the pretrained model, not a fine-tuned CellClassifier)

genes_to_perturb_list = []

isp = InSilicoPerturber(perturb_type=select_perturb_type,
                        perturb_rank_shift=None,
                        genes_to_perturb="all" if len(genes_to_perturb_list) == 0 else genes_to_perturb_list,
                        combos=0,
                        anchor_gene=None,
                        model_type=use_model_type,
                        num_classes=2,               # <-- 2 classes: Cop1_WT and Cop1_KO
                        emb_mode="cell",
                        cell_emb_style="mean_pool",
                        filter_data=None,
                        cell_states_to_model={'state_key': 'disease', 
                                              'start_state': start_state, 
                                              'goal_state': end_state, 
                                              'alt_states': alt_state}, 
                        max_ncells=2000,
                        emb_layer=0,
                        forward_batch_size=50,
                        nproc=6)