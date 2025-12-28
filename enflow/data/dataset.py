from pathlib import Path
from platform import node

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from enflow.commons.covmat import build_conformer
from enflow.commons.featurization import MoleculeFeaturizer
from enflow.commons.io import get_base_data_dir


class EuclideanDataset(Dataset):
    """Returns 3D Graph for different datasets

    Usage
    -----
    ```python
    from etflow.data import EuclideanDataset
    # pass path to processed data_dir
    dataset = EuclideanDataset(
        data_dir="processed",
        split="train",  # "train", "val", or "test"
        partition="drugs",  # "drugs" or "qm9"
    )
    ```
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        split: str = "train",
        partition: str = "drugs",
        reflow_processed: bool = False,
        return_all_poses: bool = False, # for some testing purpose
    ):
        super().__init__()
        self.mol_feat = MoleculeFeaturizer()

        # Set up paths
        if data_dir is None:
            if not reflow_processed:
                self.data_dir = Path(get_base_data_dir()) / "processed"
            else:
                self.data_dir = Path(get_base_data_dir()) / "reflow_processed"
        else:
            self.data_dir = Path(data_dir)

        # Set split and partition
        self.split = split
        self.partition = partition
        self.reflow_processed = reflow_processed
        self.return_all_poses = return_all_poses    # for some testing purpose

        # Find all data files for the specified partition and split
        self.data_files = list((self.data_dir / partition.lower() / split).glob("*.pt"))

        if len(self.data_files) == 0:
            raise ValueError(
                f"No data files found for partition {partition} and split {split}"
            )

        # Sort files for reproducibility
        self.data_files.sort()

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        # Load the data file
        data_path = self.data_files[idx]
        try:
            data = torch.load(data_path, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {data_path}: {e}")

        # sample a random conformer
        if not self.reflow_processed:
            
            if self.return_all_poses:
                return data # for some testing purpose, can not make a batch in this way.
            
            pos_confs = data.pos
            energies = data.energy

            if energies.shape[0] != 1:
                mean, std = energies.mean().unsqueeze(-1), energies.std().unsqueeze(-1)
            else:
                mean, std = energies[0], torch.tensor([1e-8])

            energies = (energies - mean) / (std + 1e-8)

            bolzmann_weights = data.boltzmann_weight
            conf_idx = np.random.randint(0, pos_confs.shape[0])
            pos = pos_confs[conf_idx]
            energy = energies[conf_idx]
            boltzmann_weight = bolzmann_weights[conf_idx]

            mol = data.mol
            mol.AddConformer(build_conformer(pos))

            data.pos = pos
            data.energy = energy
            data.energy_mean = mean
            data.energy_std = std
            data.boltzmann_weight = boltzmann_weight
            data.mol = mol
        
        else:
            x0_reflow, x1_reflow = data.x0_reflow, data.x1_reflow
            pair_idx = np.random.randint(0, x0_reflow.shape[0])
            x0, x1 = x0_reflow[pair_idx], x1_reflow[pair_idx]   # [n, 3], [n, 3]
            del data.x0_reflow, data.x1_reflow
            data.pos_x0 = x0
            data.pos_x1 = x1
            
        return data

    def get_data_path(self, idx):
        return self.data_files[idx]
