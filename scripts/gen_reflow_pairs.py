
from __future__ import annotations

import argparse
import datetime
from math import e
import os
import os.path as osp
import shutil
import time
from typing import List
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger as log
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from enflow.commons.io import get_base_data_dir
from enflow.commons import save_pkl, load_pkl
from enflow.data import EuclideanDataset
from enflow.models import BaseFlow
from enflow.models.utils import center_of_mass
from utils import instantiate_model, read_yaml

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------

torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def get_datetime() -> str:
    """Return current datetime formatted for directory names."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def split_indices(n: int, world_size: int) -> List[np.ndarray]:
    """Split ``range(n)`` into ``world_size`` nearly equal parts."""
    return list(np.array_split(np.arange(n), world_size))

# -----------------------------------------------------------------------------
# Worker process
# -----------------------------------------------------------------------------

def worker(rank: int, world_size: int, args: argparse.Namespace, config: dict):
    """Executed in each subprocess (one per GPU)."""

    # 1. Device setup ----------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    log.info(f"[Rank {rank}] Using device: {device}")

    # 2. Dataset & model -------------------------------------------------
    train_set = EuclideanDataset(partition=config["datamodule_args"]["partition"], split="train")
    val_set = EuclideanDataset(partition=config["datamodule_args"]["partition"], split="val")

    model = instantiate_model(config["model"], config["model_args"])
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict)
    model: BaseFlow = model.to(device).eval()

    # 3. Indices for this rank ----------------------------------------
    train_indices = split_indices(len(train_set), world_size)[rank]
    val_indices = split_indices(len(val_set), world_size)[rank]

    # 4. Sampling loop ---------------------------------------------------
    num_pairs = args.num_reflow_pairs

    for dataset, indices in zip([train_set, val_set], [train_indices, val_indices]):

        reflow_data_dir = Path(get_base_data_dir()) / "reflow_processed" / dataset.partition.lower() / dataset.split
        os.makedirs(reflow_data_dir, exist_ok=True)

        for idx in tqdm(indices, position=rank, desc=f"Generate pairs in {dataset.split} [rank{rank}]", ncols=160):
            
            data = dataset[int(idx)]
            data_name = str(dataset.get_data_path(int(idx))).split("/")[-1]
            batched_data = Batch.from_data_list([data] * num_pairs).to(device)

            with torch.no_grad():
                z, pos, bond_index, node_attr, edge_attr, batch = (
                    batched_data["atomic_numbers"],
                    batched_data["pos"],
                    batched_data["edge_index"],
                    batched_data.get("node_attr", None),  # optional
                    batched_data.get("edge_attr", None),  # optional
                    batched_data.get("batch", None),  # optional
                )

                x0 = model.sample_base_dist(pos.shape, edge_index=bond_index, batch=batch).to(device)
                x0 = center_of_mass(x0, batch=batch)

                pos = model.simple_ode_sample(
                    x0=x0,
                    z=z,
                    bond_index=bond_index,
                    batch=batch,
                    node_attr=node_attr,
                    edge_attr=edge_attr,
                    n_timesteps=args.num_ode_steps,
                )

                x0_reflow = x0.view(num_pairs, -1, 3).cpu() # [num_pairs, num_atoms, 3]
                x1_reflow = pos.view(num_pairs, -1, 3).cpu() # [num_pairs, num_atoms, 3]
            
            # 给 data 添加 新的属性 x0_reflow x1_reflow
            data.x0_reflow = x0_reflow
            data.x1_reflow = x1_reflow

            data_path = reflow_data_dir / data_name

            torch.save(data, data_path)

# -----------------------------------------------------------------------------
# Main process
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi‑GPU Reflow Pairs Generation")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--checkpoint", "-k", required=True)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--num_reflow_pairs", type=int, default=32)
    parser.add_argument("--num_ode_steps", type=int, default=50)
    args = parser.parse_args()

    # Validate paths -----------------------------------------------------
    assert osp.exists(args.config), "Config file not found."
    assert osp.exists(args.checkpoint), "Checkpoint file not found."

    config = read_yaml(args.config)

    # GPU count ----------------------------------------------------------
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args.num_gpus = args.num_gpus or available_gpus or 1
    if args.num_gpus > available_gpus:
        log.warning(f"Requested {args.num_gpus} GPUs but only {available_gpus} available; using {available_gpus}.")
        args.num_gpus = available_gpus

    # run
    mp.spawn(worker, nprocs=args.num_gpus, args=(args.num_gpus, args, config))

    log.success("Reflow pairs generation complete.")

if __name__ == "__main__":
    main()
