import argparse
import os

from loguru import logger as log

import wandb
from enflow.commons import load_pkl
from enflow.commons.covmat import GroundStateEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, help="Path to the data file", required=True)
    parser.add_argument("--use_alignmol", "-a", action="store_true", default=False, help="Use alignmol for matching")
    args = parser.parse_args()

    path = args.path
    os.path.exists(path), f"Path {path} does not exist"
    packed_data_list = load_pkl(path)

    use_alignmol = args.use_alignmol
    ground_state_result = GroundStateEvaluator(use_alignmol=args.use_alignmol)
    log.info("Evaluation Started...")
    d_mae, d_rmse, c_rmsd = ground_state_result(packed_data_list)
    log.info(f"Distance MAE: {d_mae}")
    log.info(f"Distance RMSE: {d_rmse}")
    log.info(f"RMSD Results: {c_rmsd}")
    log.info("Evaluation finished...")
