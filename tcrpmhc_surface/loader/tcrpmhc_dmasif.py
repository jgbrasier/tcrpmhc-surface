
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from typing import List, Dict

from tcrpmhc_surface.preprocessing.pdb import (
    read_pdb_to_dataframe, 
    split_af2_tcrpmhc_df, 
)
from tcrpmhc_surface.preprocessing.graph import flag_contact_atoms



def extract_atom_data(df: pd.DataFrame, center=False, 
                      xyz_columns: List[str] = ['x_coord', 'y_coord', 'z_coord'], 
                      type_column: str = 'element_symbol',
                      label_column: str = 'is_contact') -> Dict[str, np.ndarray]:
    # ele2num dict
    ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}
    # atom coordinates
    atoms_coords = np.array(df[xyz_columns])
    # OHE atom types
    type_idx = np.array(df[type_column].apply(lambda x: ele2num[x]))
    atom_types = np.eye(len(ele2num.keys()))[type_idx]
    # interface labels
    iface_labels = np.array(df[label_column])
    if center:
        atoms_coords = atoms_coords - np.mean(atoms_coords, axis=0, keepdims=True)
    return {"xyz": atoms_coords, "types": atom_types, "iface_labels": iface_labels}


class TCRpMHCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pdb_dir: str, processed_dir: str): 

        # load data
        self.df = df
        self.pdb_dir = pdb_dir
        self.processed_dir = processed_dir


    def process(self, contact_threshold: float = 8.0, parse_header: bool = False):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        for idx in tqdm(self.df.index):
            pdb_id = str(self.df.iloc[idx]['uuid'])
            pdb_path = os.path.join(self.pdb_dir, 'model_'+pdb_id+'.pdb')
            chain_seq = (self.df.iloc[idx]['chainseq']).split('/')

            raw_df, _ = read_pdb_to_dataframe(pdb_path=pdb_path, parse_header=parse_header)
            tcr_raw_df, pmhc_raw_df = split_af2_tcrpmhc_df(raw_df, chain_seq)
            tcr_df, pmhc_df = flag_contact_atoms(tcr_raw_df, pmhc_raw_df, deprotonate=False, threshold=contact_threshold)
            tcr_data = extract_atom_data(tcr_df)
            pmhc_data = extract_atom_data(pmhc_df)

            np.save(os.path.join(self.processed_dir, pdb_id+"_tcr_atomxyz.npy"), tcr_data['xyz'])
            np.save(os.path.join(self.processed_dir, pdb_id+"_tcr_atomtypes.npy"), tcr_data['types'])
            np.save(os.path.join(self.processed_dir, pdb_id+"_tcr_iface_labels.npy"), tcr_data['iface_labels'])

            np.save(os.path.join(self.processed_dir, pdb_id+"_pmhc_atomxyz.npy"), pmhc_data['xyz'])
            np.save(os.path.join(self.processed_dir, pdb_id+"_pmhc_atomtypes.npy"), pmhc_data['types'])
            np.save(os.path.join(self.processed_dir, pdb_id+"_pmhc_iface_labels.npy"), pmhc_data['iface_labels'])

        

        

            