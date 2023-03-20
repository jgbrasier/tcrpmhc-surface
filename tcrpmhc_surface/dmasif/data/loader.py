import pandas as pd
import os
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.transforms import Compose
import numpy as np
from scipy.spatial.transform import Rotation
import math
import urllib.request
import tarfile
from pathlib import Path
import requests
from tcrpmhc_surface.dmasif.data.convert import convert_pdbs



tensor = torch.FloatTensor
inttensor = torch.LongTensor

def load_protein_npy(pdb_id, data_dir, center=False, single_pdb=False):
    """Loads a protein point cloud and its features"""

    # Load the data, and read the connectivity information:
    # Normalize the point cloud, as specified by the user:
    points = None if single_pdb else tensor(np.load(data_dir / (pdb_id + "_xyz.npy")))
    center_location = None if single_pdb else torch.mean(points, axis=0, keepdims=True)

    atom_coords = tensor(np.load(data_dir / (pdb_id + "_atomxyz.npy")))
    atom_types = tensor(np.load(data_dir / (pdb_id + "_atomtypes.npy")))

    if center:
        points = points - center_location
        atom_coords = atom_coords - center_location

    # Interface labels
    iface_labels = (
        None
        if single_pdb
        else tensor(np.load(data_dir / (pdb_id + "_iface_labels.npy")).reshape((-1, 1)))
    )

    # Features
    chemical_features = (
        None if single_pdb else tensor(np.load(data_dir / (pdb_id + "_features.npy")))
    )

    # Normals
    normals = (
        None if single_pdb else tensor(np.load(data_dir / (pdb_id + "_normals.npy")))
    )

    protein_data = Data(
        xyz=points,
        chemical_features=chemical_features,
        y=iface_labels,
        normals=normals,
        center_location=center_location,
        num_nodes=None if single_pdb else points.shape[0],
        atom_coords=atom_coords,
        atom_types=atom_types,
    )
    return protein_data


class PairData(Data):
    def __init__(
        self,
        xyz_p1=None,
        xyz_p2=None,
        chemical_features_p1=None,
        chemical_features_p2=None,
        y_p1=None,
        y_p2=None,
        normals_p1=None,
        normals_p2=None,
        center_location_p1=None,
        center_location_p2=None,
        atom_coords_p1=None,
        atom_coords_p2=None,
        atom_types_p1=None,
        atom_types_p2=None,
        atom_center1=None,
        atom_center2=None,
        rand_rot1=None,
        rand_rot2=None,
    ):
        super().__init__()
        self.xyz_p1 = xyz_p1
        self.xyz_p2 = xyz_p2
        self.chemical_features_p1 = chemical_features_p1
        self.chemical_features_p2 = chemical_features_p2
        self.y_p1 = y_p1
        self.y_p2 = y_p2
        self.normals_p1 = normals_p1
        self.normals_p2 = normals_p2
        self.center_location_p1 = center_location_p1
        self.center_location_p2 = center_location_p2
        self.atom_coords_p1 = atom_coords_p1
        self.atom_coords_p2 = atom_coords_p2
        self.atom_types_p1 = atom_types_p1
        self.atom_types_p2 = atom_types_p2
        self.atom_center1 = atom_center1
        self.atom_center2 = atom_center2
        self.rand_rot1 = rand_rot1
        self.rand_rot2 = rand_rot2

    def __inc__(self, key, value, *args, **kwargs):
        if key == "face_p1":
            return self.xyz_p1.size(0)
        if key == "face_p2":
            return self.xyz_p2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if ("index" in key) or ("face" in key):
            return 1
        else:
            return 0

def load_protein_pair(pdb_id, data_dir,single_pdb=False):
    """Loads a protein surface mesh and its features"""
    pspl = pdb_id.split("_")
    p1_id = pspl[0] + "_" + pspl[1]
    p2_id = pspl[0] + "_" + pspl[2]

    p1 = load_protein_npy(p1_id, data_dir, center=False,single_pdb=single_pdb)
    p2 = load_protein_npy(p2_id, data_dir, center=False,single_pdb=single_pdb)
    # pdist = ((p1['xyz'][:,None,:]-p2['xyz'][None,:,:])**2).sum(-1).sqrt()
    # pdist = pdist<2.0
    # y_p1 = (pdist.sum(1)>0).to(torch.float).reshape(-1,1)
    # y_p2 = (pdist.sum(0)>0).to(torch.float).reshape(-1,1)
    y_p1 = p1["y"]
    y_p2 = p2["y"]

    protein_pair_data = PairData(
        xyz_p1=p1["xyz"],
        xyz_p2=p2["xyz"],
        chemical_features_p1=p1["chemical_features"],
        chemical_features_p2=p2["chemical_features"],
        y_p1=y_p1,
        y_p2=y_p2,
        normals_p1=p1["normals"],
        normals_p2=p2["normals"],
        center_location_p1=p1["center_location"],
        center_location_p2=p2["center_location"],
        atom_coords_p1=p1["atom_coords"],
        atom_coords_p2=p2["atom_coords"],
        atom_types_p1=p1["atom_types"],
        atom_types_p2=p2["atom_types"],
    )
    return protein_pair_data
