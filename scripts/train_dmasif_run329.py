
import pandas as pd
import numpy as np
from pathlib import Path

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


PDB_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/pdb/run329_results_for_jg"
PROCESSED_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/surface/run_329"
TSV_PATH = "data/preprocessed/run329_results.tsv"


# Parse the arguments, prepare the TensorBoard writer:
args = parser.parse_args()
writer = SummaryWriter("runs/{}".format(args.experiment_name))
model_path = "models/" + args.experiment_name

if not Path("models/").exists():
    Path("models/").mkdir(exist_ok=False)

# Ensure reproducibility:
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Create the model, with a warm restart if applicable:
net = dMaSIF(args)
net = net.to(args.device)

# We load the train and test datasets.
# Random transforms, to ensure that no network/baseline overfits on pose parameters:
# currently NormalizeChemFeatures() not supported for tcr_pmhc
transformations = (
    Compose([CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else None
)

# Read in and split data
df = pd.read_csv(TSV_PATH, sep='\t')
# select only positive samples .sample(2000, random_state=args.seed)
df = df[df['binder']==1].copy()
train_df, test_df, selected_targets = hard_split_df(df, 'peptide', min_ratio=0.85, random_seed=args.seed, low=30, high=500)


# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
# Load the train dataset:
train_dataset = TCRpMHCDataset(
    df=train_df, pdb_dir=PDB_DIR, processed_dir=PROCESSED_DIR, transform=transformations
)
train_dataset.process()

# train_dataset = [data for data in train_dataset if iface_valid_filter(data)]

train_loader = DataLoader(
    train_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True
)
# print("Preprocessing training dataset")
# train_dataset = iterate_surface_precompute(train_loader, net, args)

# Train/Validation split:
train_nsamples = len(train_dataset)
val_nsamples = int(train_nsamples * args.validation_fraction)
train_nsamples = train_nsamples - val_nsamples
train_dataset, val_dataset = random_split(
    train_dataset, [train_nsamples, val_nsamples]
)

# Load the test dataset:
test_dataset = TCRpMHCDataset(
    df=test_df, pdb_dir=PDB_DIR, processed_dir=PROCESSED_DIR, transform=transformations
)
test_dataset.process()

# test_dataset = [data for data in test_dataset if iface_valid_filter(data)]
test_loader = DataLoader(
    test_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True
)
# print("Preprocessing testing dataset")
# test_dataset = iterate_surface_precompute(test_loader, net, args)


# PyTorch_geometric data loaders:
train_loader = DataLoader(
    train_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=1, follow_batch=batch_vars)
test_loader = DataLoader(test_dataset, batch_size=1, follow_batch=batch_vars)


# Baseline optimizer:
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, amsgrad=True)
best_loss = 1e10  # We save the "best model so far"

starting_epoch = 0
if args.restart_training != "":
    checkpoint = torch.load("models/" + args.restart_training)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    starting_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]

# Training loop (~100 times) over the dataset:
for i in range(starting_epoch, args.n_epochs):
    # Train first, Test second:
    for dataset_type in ["Train", "Validation", "Test"]:
        if dataset_type == "Train":
            test = False
        else:
            test = True

        suffix = dataset_type
        if dataset_type == "Train":
            dataloader = train_loader
        elif dataset_type == "Validation":
            dataloader = val_loader
        elif dataset_type == "Test":
            dataloader = test_loader
        
        # Perform one pass through the data:
        info = iterate(
            net,
            dataloader,
            optimizer,
            args,
            test=test,
            summary_writer=writer,
            epoch_number=i,
        )

        # Write down the results using a TensorBoard writer:
        for key, val in info.items():
            if key in [
                "Loss",
                "ROC-AUC",
                "Distance/Positives",
                "Distance/Negatives",
                "Matching ROC-AUC",
            ]:
                writer.add_scalar(f"{key}/{suffix}", np.mean(val), i)

            if "R_values/" in key:
                val = np.array(val)
                writer.add_scalar(f"{key}/{suffix}", np.mean(val[val > 0]), i)

        if dataset_type == "Validation":  # Store validation loss for saving the model
            val_loss = np.mean(info["Loss"])

    if True:  # Additional saves
        if val_loss < best_loss:
            print("Validation loss {}, saving model".format(val_loss))
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                model_path + "_epoch{}".format(i),
            )

            best_loss = val_loss
