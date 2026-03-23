"""
In-silico perturbation driver (notebook logic as a script).

Speed on DGX / large GPUs:
  - Imports apply TF32-friendly matmul settings (see in_silico_perturber._configure_cuda_performance).
  - Tune --forward-batch-size upward until just below OOM; larger batches improve GPU utilization.
  - Optional: GENEFORMER_TORCH_COMPILE=1 for torch.compile (PyTorch 2+; first epochs may be slow).
  - Multi-GPU (shard cells): accelerate launch --num_processes <ngpu> run_isp.py [...]
    (uses Accelerate sharding in InSilicoPerturber when num_processes > 1).
"""
import argparse
import os
import sys

import numpy as np
import torch
from geneformer import InSilicoPerturber, InSilicoPerturberStats


def main():
    p = argparse.ArgumentParser(description="Run Geneformer in-silico perturbation + stats.")
    p.add_argument(
        "--forward-batch-size",
        type=int,
        default=int(os.environ.get("ISP_FORWARD_BATCH_SIZE", "128")),
        help="GPU minibatch size for transformer forwards (raise until memory limit).",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=int(os.environ.get("ISP_NPROC", "8")),
        help="CPU workers for HuggingFace datasets map/filter.",
    )
    args = p.parse_args()

    print("Checking CUDA...", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    dataset_name = "/app/data/Mouse-Genecorpus-20M/eval_dataset/in_silico_perturbation/Cop1KO_isp_mouse_tokenize_dataset_v-n1.dataset"

    select_perturb_type = "delete"
    start_state = "Cop1_WT"
    end_state = "Cop1_KO"
    alt_state = []
    use_model_type = "Pretrained"
    genes_to_perturb_list = []
    organ_data = "Cop1KO"

    isp = InSilicoPerturber(
        perturb_type=select_perturb_type,
        perturb_rank_shift=None,
        genes_to_perturb=(
            "all" if len(genes_to_perturb_list) == 0 else genes_to_perturb_list
        ),
        combos=0,
        anchor_gene=None,
        model_type=use_model_type,
        num_classes=2,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        filter_data=None,
        cell_states_to_model={
            "state_key": "disease",
            "start_state": start_state,
            "goal_state": end_state,
            "alt_states": alt_state,
        },
        max_ncells=2000,
        emb_layer=0,
        forward_batch_size=args.forward_batch_size,
        nproc=args.nproc,
    )

    start_state_fn = start_state.replace(" ", "-")
    end_state_fn = end_state.replace(" ", "-")

    DIR_NAME = "/app/output/isp_results"
    os.makedirs(DIR_NAME, exist_ok=True)

    print("Starting perturbation...")
    isp.perturb_data(
        "/app/models/mouse-Geneformer/",
        dataset_name,
        DIR_NAME + "/",
        "output_in-silico_SE{}_OR{}_ST{}_EN{}".format(
            select_perturb_type, organ_data, start_state_fn, end_state_fn
        ),
    )

    print("Perturbation complete. Generating stats...")
    ispstats = InSilicoPerturberStats(
        mode="goal_state_shift",
        genes_perturbed="all" if len(genes_to_perturb_list) == 0 else genes_to_perturb_list,
        combos=0,
        anchor_gene=None,
        cell_states_to_model={
            "state_key": "disease",
            "start_state": start_state,
            "goal_state": end_state,
            "alt_states": alt_state,
        },
    )

    STATS_DIR_NAME = "/app/output/ispstats_results"
    os.makedirs(STATS_DIR_NAME, exist_ok=True)

    ispstats.get_stats(
        DIR_NAME + "/",
        None,
        STATS_DIR_NAME + "/",
        "output_in-silico_SE{}_OR{}_ST{}_EN{}".format(
            select_perturb_type, organ_data, start_state_fn, end_state_fn
        ),
    )
    print("Stats generation complete. Check the parquet file.")


if __name__ == "__main__":
    main()
