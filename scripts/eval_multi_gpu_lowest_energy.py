from __future__ import annotations

import argparse
import datetime
import os
import os.path as osp
import shutil
import time
from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger as log
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from enflow.commons import save_pkl, load_pkl
from enflow.data import EuclideanDataset
from enflow.models import BaseFlow
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
    dataset = EuclideanDataset(
        partition=config["datamodule_args"]["partition"], split="test"
    )

    model = instantiate_model(config["model"], config["model_args"])
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict)
    model: BaseFlow = model.to(device).eval()

    # 3. Indices for this rank ------------------------------------------
    indices = split_indices(len(dataset), world_size)[rank]
    if args.debug and indices:
        indices = indices[:1]
    log.info(f"[Rank {rank}] Processing {len(indices)} molecules")

    # 4. Sampling loop ---------------------------------------------------
    eval_args = config.get("eval_args", {})
    num_samples = args.num_samples

    data_dict: dict[str, Data] = {}
    times: List[float] = []

    for idx in tqdm(indices, position=rank, desc=f"rank{rank}"):

        data = dataset[int(idx)]
        smiles = data.smiles
        pos_ref = torch.load(dataset.data_files[int(idx)], weights_only=False).pos.cpu().numpy()
        pos_ref = pos_ref[0]

        batch_data = Batch.from_data_list([data] * num_samples).to(device)
        tic = time.time()
        with torch.no_grad():
            res = model.sample(
                batch_data["atomic_numbers"],
                batch_data["edge_index"],
                batch_data["batch"],
                node_attr=batch_data["node_attr"],
                chiral_index=batch_data["chiral_index"],
                chiral_nbr_index=batch_data["chiral_nbr_index"],
                chiral_tag=batch_data["chiral_tag"],
                energy_guided=args.energy_guided,
                guide_scale=args.guide_scale,
                return_energy=args.return_energy,
                **eval_args.get("sampler_args", {}),
            )
        if args.return_energy:
            pos, E = res
        else:
            pos = res
        times.append((time.time() - tic) / num_samples)
        # generated_positions: (num_samples, num_atoms, 3)
        if num_samples > 1:
            # energy: (num_samples,)
            generated_positions = pos.view(num_samples, -1, 3).cpu().detach().numpy()
            energies = E.cpu().detach().numpy()
            # find lowest energy
            min_energy_idx = np.argmin(energies, axis=0)
            lowest_energy_positions = generated_positions[min_energy_idx]
        else: 
            lowest_energy_positions = pos.cpu().detach().numpy()

        data_dict[smiles] = Data(
            smiles=smiles,
            pos_ref=pos_ref,
            rdmol=data.mol,
            pos_gen=lowest_energy_positions,
        )

    # 5. Persist temporary results --------------------------------------
    tmp_dir = osp.join(args.output_dir, "tmp_pickles")
    os.makedirs(tmp_dir, exist_ok=True)
    save_pkl(osp.join(tmp_dir, f"generated_rank{rank}.pkl"), list(data_dict.values()))
    save_pkl(osp.join(tmp_dir, f"times_rank{rank}.pkl"), times)


# -----------------------------------------------------------------------------
# Main process
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi‑GPU inference (no W&B)")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--checkpoint", "-k", required=True)
    parser.add_argument("--output_dir", "-o", default="logs/")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--energy_guided", action="store_true")
    parser.add_argument("--guide_scale", type=float, default=2.0)
    parser.add_argument("--return_energy", action="store_true")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    # Validate paths -----------------------------------------------------
    assert osp.exists(args.config), "Config file not found."
    assert osp.exists(args.checkpoint), "Checkpoint file not found."

    config = read_yaml(args.config)
    task_name = config.get("task_name", "default")

    # GPU count ----------------------------------------------------------
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args.num_gpus = args.num_gpus or available_gpus or 1
    if args.num_gpus > available_gpus:
        log.warning(f"Requested {args.num_gpus} GPUs but only {available_gpus} available; using {available_gpus}.")
        args.num_gpus = available_gpus

    # Output directory ---------------------------------------------------
    args.output_dir = osp.join(args.output_dir, f"samples/{task_name}/{get_datetime()}")
    os.makedirs(args.output_dir, exist_ok=True)

    log.info(f"Spawning {args.num_gpus} workers (debug={args.debug}) → {args.output_dir}")
    mp.spawn(worker, nprocs=args.num_gpus, args=(args.num_gpus, args, config))

    # Merge results ------------------------------------------------------
    tmp_dir = osp.join(args.output_dir, "tmp_pickles")
    merged_data: dict[str, Data] = {}
    merged_times: List[float] = []

    for rank in range(args.num_gpus):
        gen_path = osp.join(tmp_dir, f"generated_rank{rank}.pkl")
        time_path = osp.join(tmp_dir, f"times_rank{rank}.pkl")
        if not osp.exists(gen_path):
            log.error(f"Missing pickle from rank {rank}: {gen_path}")
            continue
        part_data: List[Data] = load_pkl(gen_path)
        part_times: List[float] = load_pkl(time_path)
        merged_times.extend(part_times)
        for d in part_data:
            merged_data[d.smiles] = d

    save_pkl(osp.join(args.output_dir, "generated_files.pkl"), list(merged_data.values()))
    save_pkl(osp.join(args.output_dir, "times.pkl"), merged_times)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log.success("Inference complete – merged results saved.")


if __name__ == "__main__":
    main()
