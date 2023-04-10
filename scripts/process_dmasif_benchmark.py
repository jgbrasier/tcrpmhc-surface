import pandas as pd
import numpy as np
from pathlib import Path
import uuid
import os

from tqdm import tqdm
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from tcrpmhc_surface.loader import TCRpMHCDataset
from tcrpmhc_surface.dmasif.data.transforms import RandomRotationPairAtoms, NormalizeChemFeatures, CenterPairAtoms
from tcrpmhc_surface.dmasif.model import dMaSIF
from tcrpmhc_surface.dmasif.data.iteration import iterate, iterate_surface_precompute, process, extract_single, generate_matchinglabels
from tcrpmhc_surface.dmasif.data.loader import iface_valid_filter
from tcrpmhc_surface.utils import hard_split_df

from Arguments import parser

import pykeops
# Clean up the already compiled files
pykeops.clean_pykeops()


PDB_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/pdb/run330_results_for_jg"
PROCESSED_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/surface/tcr_3d"
SAVE_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/surface/tcr_3d_mesh"
TSV_PATH = "data/preprocessed/run330_results.tsv"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# Parse the arguments, prepare the TensorBoard writer:
args = parser.parse_args()
RUN_ID = args.restart_training.split("_")[-3] if args.restart_training else str(uuid.uuid4())[:8]
print("RUN ID:", RUN_ID)
writer = SummaryWriter("runs/{}_{}".format(args.experiment_name, RUN_ID))
model_path = "models/" + args.experiment_name + "_{}".format(RUN_ID)
stats_path = "stats/point_cloud_gen"

if not Path("models/").exists():
    Path("models/").mkdir(exist_ok=False)
if not os.path.exists(stats_path):
    os.makedirs(stats_path)

# Ensure reproducibility:
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Create the model, with a warm restart if applicable:
net = dMaSIF(args)
net = net.to(args.device)

# Read in and generate data
df = pd.read_csv(TSV_PATH, sep='\t')
test_pdb_ids = [f.split("_")[0] for f in os.listdir(PROCESSED_DIR)]
df = pd.DataFrame({'uuid': test_pdb_ids}).drop_duplicates()
print(f"Generating {len(df.index)} structures...")
dataset = TCRpMHCDataset(
    df=df, pdb_dir=PDB_DIR, processed_dir=PROCESSED_DIR, transform=None, include_mesh_data=False
)

# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]

# PyTorch_geometric data loaders:
dataloader = DataLoader(
    dataset, batch_size=1, follow_batch=batch_vars, shuffle=False
)
surface_times = []
label_times = []
for it, protein_pair in enumerate(tqdm(dataloader)):
    # , desc="Test " if test else "Train")):k,
    if os.path.exists(os.path.join(SAVE_DIR, protein_pair.name_p2[0][0]+'_xyz.npy')):
        continue

    protein_batch_size = protein_pair.atom_coords_p1_batch[-1].item() + 1
    protein_pair.to(args.device)

    # Generate the surface:
    torch.cuda.synchronize()
    surface_time = time.time()
    P1_batch, P2_batch = process(args, protein_pair, net)
    torch.cuda.synchronize()
    surface_time = time.time() - surface_time
    surface_times.append(surface_time)

    for protein_it in range(protein_batch_size):

        P1 = extract_single(P1_batch, protein_it)
        P2 = extract_single(P2_batch, protein_it)

        # generate labels
        torch.cuda.synchronize()
        label_time = time.time()
        P1, P2 = generate_matchinglabels(args, P1, P2)
        torch.cuda.synchronize()
        label_time = time.time() - label_time
        label_times.append(label_time)

        
        p1_name = P1['name'][0]
        p2_name = P2['name'][0]

        # np.save(os.path.join(stats_path, p1_name+'_xyz'), P1['xyz'].cpu().detach().numpy())
        # np.save(os.path.join(stats_path, p2_name+'_xyz'), P2['xyz'].cpu().detach().numpy())
        # np.save(os.path.join(stats_path, p1_name+'_iface_labels'), P1['labels'].cpu().detach().numpy())
        # np.save(os.path.join(stats_path, p2_name+'_iface_labels'), P2['labels'].cpu().detach().numpy())
        # np.save(os.path.join(stats_path, p1_name+'_normals'), P1['normals'].cpu().detach().numpy())
        # np.save(os.path.join(stats_path, p2_name+'_normals'), P2['normals'].cpu().detach().numpy())
        # np.save(os.path.join(stats_path, p1_name+'_atomxyz'), P1['atom_xyz'].cpu().detach().numpy())
        # np.save(os.path.join(stats_path, p2_name+'_atomxyz'), P2['atom_xyz'].cpu().detach().numpy())
        # np.save(os.path.join(stats_path, p1_name+'_atomtypes'), P1['atomtypes'].cpu().detach().numpy())
        # np.save(os.path.join(stats_path, p2_name+'_atomtypes'), P2['atomtypes'].cpu().detach().numpy())

        np.save(os.path.join(SAVE_DIR, p1_name+'_xyz'), P1['xyz'].cpu().detach().numpy())
        np.save(os.path.join(SAVE_DIR, p2_name+'_xyz'), P2['xyz'].cpu().detach().numpy())
        np.save(os.path.join(SAVE_DIR, p1_name+'_iface_labels'), P1['labels'].cpu().detach().numpy())
        np.save(os.path.join(SAVE_DIR, p2_name+'_iface_labels'), P2['labels'].cpu().detach().numpy())
        np.save(os.path.join(SAVE_DIR, p1_name+'_normals'), P1['normals'].cpu().detach().numpy())
        np.save(os.path.join(SAVE_DIR, p2_name+'_normals'), P2['normals'].cpu().detach().numpy())
        np.save(os.path.join(SAVE_DIR, p1_name+'_atomxyz'), P1['atom_xyz'].cpu().detach().numpy())
        np.save(os.path.join(SAVE_DIR, p2_name+'_atomxyz'), P2['atom_xyz'].cpu().detach().numpy())
        np.save(os.path.join(SAVE_DIR, p1_name+'_atomtypes'), P1['atomtypes'].cpu().detach().numpy())
        np.save(os.path.join(SAVE_DIR, p2_name+'_atomtypes'), P2['atomtypes'].cpu().detach().numpy())

np.save(os.path.join(stats_path, 'tcr_3d_surface_times'), surface_times)
np.save(os.path.join(stats_path, 'tcr_3d_label_times'), label_times)