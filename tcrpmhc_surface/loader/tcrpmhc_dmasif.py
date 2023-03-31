
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
    def __init__(self, df: pd.DataFrame, pdb_dir: str, processed_dir: str, train=True, transform: Union[Compose, Callable]=None): 

        # load data
        self.df = df
        self.pdb_dir = pdb_dir
        self.processed_dir = processed_dir+"_train" if train else processed_dir+"_test"

        # transformations (ex: random rotation etc..)
        self.transform = transform

    def process(self, contact_threshold: float = 8.0, parse_header: bool = False) -> None:
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if len(os.listdir(self.processed_dir)) == len(self.df.index)*6: # x6 because each file 
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
                tcr_data = extract_atom_data(tcr_df)
                pmhc_data = extract_atom_data(pmhc_df)

                np.save(os.path.join(self.processed_dir, pdb_id+"_tcr_atomxyz.npy"), tcr_data['xyz'])
                np.save(os.path.join(self.processed_dir, pdb_id+"_tcr_atomtypes.npy"), tcr_data['types'])
                np.save(os.path.join(self.processed_dir, pdb_id+"_tcr_iface_labels.npy"), tcr_data['iface_labels'])

                np.save(os.path.join(self.processed_dir, pdb_id+"_pmhc_atomxyz.npy"), pmhc_data['xyz'])
                np.save(os.path.join(self.processed_dir, pdb_id+"_pmhc_atomtypes.npy"), pmhc_data['types'])
                np.save(os.path.join(self.processed_dir, pdb_id+"_pmhc_iface_labels.npy"), pmhc_data['iface_labels'])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> PairData:
        p1_id = str(self.df.iloc[index]['uuid'])+"_tcr"
        p2_id = str(self.df.iloc[index]['uuid'])+"_pmhc"
        p1 = load_protein_npy(p1_id, self.processed_dir, mesh=False, single_pdb=False, chemical_features=False, normals=False)
        p2 = load_protein_npy(p2_id, self.processed_dir, mesh=False, single_pdb=False, chemical_features=False, normals=False)
        # if all flags set to False: atom_coords, atom_types, iface_labels
        protein_pair_data = PairData(
            y_p1=p1["y"],
            y_p2=p2["y"],
            atom_coords_p1=p1["atom_coords"],
            atom_coords_p2=p2["atom_coords"],
            atom_types_p1=p1["atom_types"],
            atom_types_p2=p2["atom_types"],
        )
        if self.transform:
            protein_pair_data = self.transform(protein_pair_data)
        return protein_pair_data
    

    

    

    

        

        

            