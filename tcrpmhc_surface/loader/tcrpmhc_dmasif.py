
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from torch_geometric.transforms import Compose

from typing import List, Dict, Union, Callable

from tcrpmhc_surface.preprocessing.pdb import (
    read_pdb_to_dataframe, 
    split_af2_tcrpmhc_df, 
)
from tcrpmhc_surface.preprocessing.graph import (
    flag_contact_atoms, 
    extract_atom_data
)
from tcrpmhc_surface.dmasif.data.loader import (
    load_protein_pair,
    load_protein_npy,
    PairData
)


class TCRpMHCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pdb_dir: str = None, processed_dir: str = None, transform: Union[Compose, Callable]=None, include_mesh_data=True): 

        # load data
        self.df = df
        self.pdb_dir = pdb_dir
        self.processed_dir = processed_dir

        # transformations (ex: random rotation etc..)
        self.transform = transform
        self.mesh_data = include_mesh_data

    def process(self, contact_threshold: float = 8.0, parse_header: bool = False) -> None:
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if len(os.listdir(self.processed_dir)) > 0 :
            print(f"PDBs already processed in {self.processed_dir}")
        else:
            print(f"Processing PDBs, saving to {self.processed_dir}")
            for idx in tqdm(range(len(self.df.index))):
                pdb_id = str(self.df.iloc[idx]['uuid'])
                pdb_path = os.path.join(self.pdb_dir, 'model_'+pdb_id+'.pdb')
                chain_seq = (self.df.iloc[idx]['chainseq']).split('/')

                raw_df, _ = read_pdb_to_dataframe(pdb_path=pdb_path, parse_header=parse_header)
                tcr_raw_df, pmhc_raw_df = split_af2_tcrpmhc_df(raw_df, chain_seq)
                tcr_df, pmhc_df = flag_contact_atoms(tcr_raw_df, pmhc_raw_df, deprotonate=False, threshold=contact_threshold)
                tcr_data = extract_atom_data(tcr_df, center=False)
                pmhc_data = extract_atom_data(pmhc_df, center=False)

                np.save(os.path.join(self.processed_dir, pdb_id+"_tcr_atomxyz.npy"), tcr_data['atom_xyz'])
                np.save(os.path.join(self.processed_dir, pdb_id+"_tcr_atomtypes.npy"), tcr_data['atom_types'])
                np.save(os.path.join(self.processed_dir, pdb_id+"_tcr_atom_labels.npy"), tcr_data['atom_labels'])

                np.save(os.path.join(self.processed_dir, pdb_id+"_pmhc_atomxyz.npy"), pmhc_data['atom_xyz'])
                np.save(os.path.join(self.processed_dir, pdb_id+"_pmhc_atomtypes.npy"), pmhc_data['atom_types'])
                np.save(os.path.join(self.processed_dir, pdb_id+"_pmhc_atom_labels.npy"), pmhc_data['atom_labels'])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> PairData:
        # P1 is always TCR, P2 is always pMHC
        p1_id = str(self.df.iloc[index]['uuid'])+"_tcr"
        p2_id = str(self.df.iloc[index]['uuid'])+"_pmhc"
        if self.mesh_data:
            p1 = load_protein_npy(p1_id, self.processed_dir, mesh=True, iface_label=False, chemical_features=False, normals=True)
            p2 = load_protein_npy(p2_id, self.processed_dir, mesh=True, iface_label=False, chemical_features=False, normals=True)
            protein_pair_data = PairData(
                name_p1=p1_id,
                name_p2=p2_id,
                labels_p1=p1["y"],
                labels_p2=p2["y"],
                atom_coords_p1=p1["atom_coords"],
                atom_coords_p2=p2["atom_coords"],
                atom_types_p1=p1["atom_types"],
                atom_types_p2=p2["atom_types"],
                normals_p1=p1["normals"],
                normals_p2=p2["normals"],
                xyz_p1=p1["xyz"],
                xyz_p2=p2["xyz"],
            )
        else:
            p1 = load_protein_npy(p1_id, self.processed_dir, mesh=False, single_pdb=False, chemical_features=False, normals=False)
            p2 = load_protein_npy(p2_id, self.processed_dir, mesh=False, single_pdb=False, chemical_features=False, normals=False)
            # if all flags set to False: atom_coords, atom_types, iface_labels
            protein_pair_data = PairData(
                name_p1=p1_id,
                name_p2=p2_id,
                atom_coords_p1=p1["atom_coords"],
                atom_coords_p2=p2["atom_coords"],
                atom_types_p1=p1["atom_types"],
                atom_types_p2=p2["atom_types"],
            )
        if self.transform:
            protein_pair_data = self.transform(protein_pair_data)
        return protein_pair_data
    

    

    

    

        

        

            