import pandas as pd
import os
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
import numpy as np
from scipy.spatial.transform import Rotation

from pathlib import Path
from tcrpmhc_surface.dmasif.data.convert import convert_pdbs


   
tensor = torch.FloatTensor
inttensor = torch.LongTensor

def iface_valid_filter(protein_pair):
    labels1 = protein_pair.y_p1.reshape(-1)
    labels2 = protein_pair.y_p2.reshape(-1)
    valid1 = (
        (torch.sum(labels1) < 0.75 * len(labels1))
        and (torch.sum(labels1) > 30)
        and (torch.sum(labels1) > 0.01 * labels2.shape[0])
    )
    valid2 = (
        (torch.sum(labels2) < 0.75 * len(labels2))
        and (torch.sum(labels2) > 30)
        and (torch.sum(labels2) > 0.01 * labels1.shape[0])
    )

    return valid1 and valid2

def load_protein_npy(pdb_id, data_dir, mesh=False, iface_label=False, chemical_features=False, normals=False):
    """Loads a protein point cloud and its features"""
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    # Load the data, and read the connectivity information:
    # Normalize the point cloud, as specified by the user:
    atom_coords = tensor(np.load(data_dir / (pdb_id + "_atomxyz.npy")))
    atom_types = tensor(np.load(data_dir / (pdb_id + "_atomtypes.npy")))

    # TODO: Load mesh
    mesh_xyz = tensor(np.load(data_dir / (pdb_id + "_xyz.npy"))) if mesh else None

    # Atom labels
    iface_labels = (
        tensor(np.load(data_dir / (pdb_id + "_iface_labels.npy")).reshape((-1, 1))) if iface_label else None
    )

    # Features
    chemical_features = (
        tensor(np.load(data_dir / (pdb_id + "_features.npy"))) if chemical_features else None
    )

    # Normals
    normals = (
        tensor(np.load(data_dir / (pdb_id + "_normals.npy"))) if normals else None
    )

    protein_data = Data(
        xyz=mesh_xyz,
        chemical_features=chemical_features,
        y=iface_labels,
        normals=normals,
        num_nodes= atom_coords.shape[0],
        atom_coords=atom_coords,
        atom_types=atom_types,
    )
    return protein_data


class PairData(Data):
    def __init__(
        self,
        name_p1=None,
        name_p2=None,
        xyz_p1=None,
        xyz_p2=None,
        chemical_features_p1=None,
        chemical_features_p2=None,
        y_p1=None,
        y_p2=None,
        labels_p1=None,
        labels_p2=None,
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
        self.name_p1=name_p1,
        self.name_p2=name_p2,
        self.xyz_p1 = xyz_p1
        self.xyz_p2 = xyz_p2
        self.chemical_features_p1 = chemical_features_p1
        self.chemical_features_p2 = chemical_features_p2
        self.y_p1 = y_p1
        self.y_p2 = y_p2
        self.labels_p1 = labels_p1,
        self.labels_p2 = labels_p2,
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

def load_protein_pair(pdb_id, pdb_id_2=None, data_dir=None, mesh=False, single_pdb=False, chemical_features=False, normals=False):
    """Loads a protein surface mesh and its features"""
    if pdb_id_2 is None:
        pspl = pdb_id.split("_")
        p1_id = pspl[0] + "_" + pspl[1]
        p2_id = pspl[0] + "_" + pspl[2]
    else:
        p1_id = pdb_id
        p2_id = pdb_id_2

    p1 = load_protein_npy(p1_id, data_dir, mesh=False, single_pdb=False, chemical_features=False, normals=False)
    p2 = load_protein_npy(p2_id, data_dir, mesh=False, single_pdb=False, chemical_features=False, normals=False)
    # pdist = ((p1['xyz'][:,None,:]-p2['xyz'][None,:,:])**2).sum(-1).sqrt()
    # pdist = pdist<2.0
    # y_p1 = (pdist.sum(1)>0).to(torch.float).reshape(-1,1)
    # y_p2 = (pdist.sum(0)>0).to(torch.float).reshape(-1,1)

    protein_pair_data = PairData(
        name_p1=p1_id,
        name_p2=p2_id,
        xyz_p1=p1["xyz"],
        xyz_p2=p2["xyz"],
        chemical_features_p1=p1["chemical_features"],
        chemical_features_p2=p2["chemical_features"],
        y_p1=p1["y"],
        y_p2=p2["y"],
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
