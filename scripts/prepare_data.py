"""
Script to process GEOM dataset into individual PyG data files
===========================================================

Usage:
python prepare_data.py -d /path/to/geom/ where 'QM9' and 'GEOM' are subfolders
Data Download:
download from TorDiff[https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7]
===========================================================
"""

import os
import torch
import hashlib
import numpy as np
import pandas as pd
import datamol as dm

from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional
from loguru import logger as log
from torch_geometric.data import Data

from enflow.commons import get_base_data_dir, load_pkl, MoleculeFeaturizer


def check_disconnected_components(mol):
    """Check for disconnected components using Union-Find algorithm."""
    # Initialize parent array for union-find
    n_nodes = mol.GetNumAtoms()
    parent = list(range(n_nodes))

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Add reverse edge for undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Process all edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        union(src, dst)

    # Count unique components
    components = {}
    for node in range(n_nodes):
        root = find(node)
        if root not in components:
            components[root] = []
        components[root].append(node)

    return list(components.values())


def process_mol(
    pkl_path: Path,
    partition: str,
    split: str,
    keep_top_n: int = 30,
) -> Optional[Data]:
    """Process a single molecule with all its conformers into one PyG Data object."""
    mol_feat = MoleculeFeaturizer()
    try:
        # Load molecule data
        mol_dict = load_pkl(pkl_path)
        confs = mol_dict["conformers"]
        mols = [conf["rd_mol"] for conf in confs]

        # Get SMILES with atom indices (use first mol as they're all same)
        smiles = dm.to_smiles(
            mols[0],
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )

        # Get chemical information (same for all conformers)
        mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
        atomic_charges = torch.tensor([atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.long)
        atomic_numbers = mol_feat.get_atomic_numbers_from_mol(mol)
        node_attr = mol_feat.get_atom_features_from_mol(mol, use_ogb_feat=True)
        edge_index, edge_attr = mol_feat.get_edge_index_from_mol(mol, use_edge_feat=False)
        chiral_index, chiral_nbr_index, chiral_tag = mol_feat.get_chiral_centers_from_mol(mol)
        
        rd_mol = deepcopy(mol)
        rd_mol.RemoveAllConformers()

        if split in ["train", "val"]:
            if len(atomic_numbers) > 60:
                # remove mols with more than 60 atoms in train/val for <GPU memory> safety.
                log.warning(f"Skipping {smiles} due to too many atoms ({len(atomic_numbers)}) for GPU memory.")
                return None
        # sorted confs in both train/val/test but keep_top_n==1000 when test to save all confs.
        confs = sorted(confs, key=lambda x: x["boltzmannweight"], reverse=True)
        confs = confs[:keep_top_n]  # Keep top 30 conformers

        # Collect conformer positions and energies
        positions = []
        energies = []
        boltzmann_weights = []  # Also store the weights

        for conf in confs:
            mol = conf["rd_mol"]
            pos = torch.from_numpy(mol.GetConformer().GetPositions()).float()
            energy = torch.tensor([conf["totalenergy"]]).float()
            weight = torch.tensor([conf["boltzmannweight"]]).float()

            positions.append(pos)
            energies.append(energy)
            boltzmann_weights.append(weight)

        # Stack conformer data
        positions = torch.stack(positions)  # [num_conformers, num_atoms, 3]
        energies = torch.stack(energies)  # [num_conformers, 1]
        boltzmann_weights = torch.stack(boltzmann_weights)  # [num_conformers, 1]

        # Check for disconnected components
        components = check_disconnected_components(mol)
        if len(components) > 1:
            log.warning(f"Skipping {smiles} due to disconnected components")
            return None
        
        if '.' in smiles:
            log.warning(f"Skipping {smiles} due to multiple molecules")
            return None
        
        if len(positions) == 0:
            log.warning(f"Skipping {smiles} due to no conformers")
            return None
        
        if (
            atomic_charges is None
            or atomic_numbers is None
            or node_attr is None
            or edge_index is None
            or chiral_index is None
            or chiral_nbr_index is None
            or chiral_tag is None
        ):
            log.warning(f"Skipping {smiles} due to missing features")
            return None

        # Create single PyG Data object with all conformers
        data = Data(
            mol=rd_mol,  # Store the original molecule without conformers
            atomic_numbers=atomic_numbers,  # [num_atoms]
            node_attr=node_attr,
            edge_attr=edge_attr,
            edge_index=edge_index,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
            charges=atomic_charges,  # [num_atoms]
            pos=positions,  # [num_conformers, num_atoms, 3]
            energy=energies,  # [num_conformers, 1]
            boltzmann_weight=boltzmann_weights,  # [num_conformers, 1]
            smiles=smiles,
            subset=partition,
        )

        return data

    except Exception as e:
        log.warning(f"Skipping due to {e}")
        return None


def main(output_dir: Path, data_dir: Path) -> None:
    """Main function to process the GEOM dataset and save as PyG Data objects."""
    # Create output directories for each partition and split
    partitions = ["qm9", "drugs"]
    for partition in partitions:
        for split in ["train", "val", "test"]:
            (output_dir / partition / split).mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    stats = {
        partition: {
            split: {"mols": 0, "confs": 0} for split in ["train", "val", "test"]
        }
        for partition in partitions
    }
    skipped_mols = 0

    # Process each partition
    for partition in partitions:
        log.info(f"Processing {partition} partition")

        # Load molecule summary
        all_pkl_files = sorted(list((data_dir / partition.upper() / partition).glob("*")))
        all_smiles = sorted(os.listdir(os.path.join(data_dir, partition.upper(), partition)))
        splits = np.load(data_dir / partition.upper() / "split.npy", allow_pickle=True)
        train_split_indices = splits[0]
        val_split_indices = splits[1]
        test_split_indices = splits[2]
        # train/val pickle paths
        train_pkl_paths = [all_pkl_files[i] for i in train_split_indices]
        val_pkl_paths = [all_pkl_files[i] for i in val_split_indices]
        # test_pkl_paths = [all_pkl_files[i] for i in test_split_indices] # use 1k smiles in test_split given by cached files

        train_smiles = [all_smiles[i].replace('.pickle', '') for i in train_split_indices]
        val_smiles = [all_smiles[i].replace('.pickle', '') for i in val_split_indices]
        test_smiles = [all_smiles[i].replace('.pickle', '') for i in test_split_indices]

        # check 1000 test_smiles come from test_split
        test_smiles = pd.read_csv(data_dir / partition.upper() / "test_smiles.csv")["smiles"].values.tolist()
        test_smiles_from_train = [smiles for smiles in train_smiles if smiles in test_smiles]
        test_smiles_from_val = [smiles for smiles in val_smiles if smiles in test_smiles]
        test_smiles_from_test = [smiles for smiles in test_smiles if smiles in test_smiles]
        log.info(f"Test smiles in train split: {len(test_smiles_from_train)}")
        log.info(f"Test smiles in val split: {len(test_smiles_from_val)}")
        log.info(f"Test smiles in test split: {len(test_smiles_from_test)}")
        if len(test_smiles_from_train) > 0 or len(test_smiles_from_val) > 0:
            raise ValueError("Test smiles found in train/val split!")
        else:
            log.info(f"{partition} test smiles verified to be in test split.")

        # Process each molecule
        for split, pkl_paths in zip(["train", "val"], [train_pkl_paths, val_pkl_paths]):
            # Process each pickle file in the current split
            for idx, pkl_path in enumerate(tqdm(pkl_paths, desc=f"Processing {partition} {split}")):

                mol_id = idx  

                # skip if file already exists
                if (output_dir / partition / split / f"{mol_id}.pt").exists():
                    continue

                # Process molecule with all its conformers
                data = process_mol(pkl_path, partition, split)

                if data is None:
                    skipped_mols += 1
                    continue

                # Save molecule data in appropriate split directory
                save_path = output_dir / partition / split / f"{mol_id}.pt"

                # Save the PyG Data object
                torch.save(data, save_path)

                # Update statistics
                stats[partition][split]["mols"] += 1
                stats[partition][split]["confs"] += len(data.pos)

        # save test set data objects
        log.info(f"Processing test molecules for {partition}")
        test_mols: Dict[str, list[dm.Mol]] = load_pkl(data_dir / partition.upper() / "test_mols.pkl")
        for (smiles, test_mol), smiles_check in zip(test_mols.items(), test_smiles):
            # data = process_test_mol(test_mol, partition) # original way
            assert smiles == smiles_check, f"SMILES mismatch: {smiles} != {smiles_check}"
            pkl_path = os.path.join(data_dir, partition.upper(), partition, f"{smiles}.pickle")
            data = process_mol(pkl_path, partition, "test", keep_top_n=1000) 
            # use process_mol to ensure consistency, 'keep_top_n = 1000' to save all confs in test mol
            if data is None:
                skipped_mols += 1
                continue

            safe_test_mol_id = hashlib.md5(smiles.encode()).hexdigest()
            save_path = output_dir / partition / "test" / f"{safe_test_mol_id}.pt"
            torch.save(data, save_path)
            stats[partition]["test"]["mols"] += 1
            stats[partition]["test"]["confs"] += len(data.pos)

    # Log statistics
    log.info("Processing complete:")
    for partition in partitions:
        log.info(f"\n{partition.upper()} statistics:")
        for split in ["train", "val", "test"]:
            split_stats = stats[partition][split]
            log.info(
                f"- {split}: {split_stats['mols']} molecules, "
                f"{split_stats['confs']} conformers"
            )
    log.info(f"Skipped molecules: {skipped_mols}")


if __name__ == "__main__":

    data_dir = Path(get_base_data_dir())
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."
    # Verify the the splits are present
    for partition in ["QM9", "DRUGS"]:
        assert (data_dir / partition / "split.npy").exists(), f"Split not found in {partition}"
        assert (data_dir / partition / "test_smiles.csv").exists(), f"Train split smiles not found in {partition}"
        assert (data_dir / partition / "test_mols.pkl").exists(), f"Test split mols not found in {partition}"


    output_dir = Path(data_dir) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process the data
    main(output_dir, data_dir)
