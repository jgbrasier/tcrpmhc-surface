import pandas as pd
import numpy as np
from pathlib import Path
import uuid
import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from tcrpmhc_surface.loader import TCRpMHCDataset
from tcrpmhc_surface.dmasif.data.transforms import RandomRotationPairAtoms, NormalizeChemFeatures, CenterPairAtoms
from tcrpmhc_surface.dmasif.model import dMaSIF
from tcrpmhc_surface.dmasif.data.iteration import iterate, iterate_surface_precompute
from tcrpmhc_surface.dmasif.data.loader import iface_valid_filter
from tcrpmhc_surface.utils import hard_split_df

from Arguments import parser

import pykeops
# Clean up the already compiled files
pykeops.clean_pykeops()

# PDB_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/pdb/run330_results_for_jg"
# PROCESSED_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/surface/tcr_3d_mesh"
# TSV_PATH = "data/preprocessed/run330_results.tsv"

PROCESSED_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/surface/atlas_true_mesh"
TSV_PATH = "data/preprocessed/processed_atlas.tsv"

args = parser.parse_args()
model_path = "models/" + args.experiment_name
save_predictions_path = Path("preds/benchmark/" + args.experiment_name)

# Ensure reproducability:
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


# We load the train and test datasets.
# Random transforms, to ensure that no network/baseline overfits on pose parameters:
# currently NormalizeChemFeatures() not supported for tcr_pmhc
transformations = (
    Compose([CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else None
)


# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]

# test_pdb_ids = [f.split("_")[0] for f in os.listdir(PROCESSED_DIR)]
# test_df = pd.DataFrame({'uuid': test_pdb_ids}).drop_duplicates()

test_df = pd.read_csv(TSV_PATH, sep='\t')
test_pdb_ids = test_df['uuid'].to_list()
# Load the test dataset:
test_dataset = TCRpMHCDataset(
    df=test_df, processed_dir=PROCESSED_DIR, transform=transformations
)

print("Test len:", len(test_dataset))

# print("Preprocessing testing dataset")
# test_dataset = iterate_surface_precompute(test_loader, net, args)
# no shuffling to keep naming consistent with id list
test_loader = DataLoader(test_dataset, batch_size=1, follow_batch=batch_vars)

l1 = torch.nn.BCELoss()
l2 = torch.nn.BCELoss()

net = dMaSIF(args)
# net.load_state_dict(torch.load(model_path, map_location=args.device))
net.load_state_dict(
    torch.load(model_path, map_location=args.device)["model_state_dict"]
)
net = net.to(args.device)

# Perform one pass through the data:
info = iterate(
    net,
    test_loader,
    None,
    args,
    test=True,
    save_path=save_predictions_path,
    pdb_ids=test_pdb_ids,
    loss_fn1=l1,
    loss_fn2=l2
)

# save info dict as JSON
if not os.path.exists('timings'):
    os.makedirs('timings')
with open(f"timings/{args.experiment_name}_benchmark_out.json", "w") as outfile:
    json.dump(info, outfile)

# np.save(f"timings/{args.experiment_name}_convtime.npy", info["conv_time"])
#np.save(f"timings/{args.experiment_name}_memoryusage.npy", info["memory_usage"])
