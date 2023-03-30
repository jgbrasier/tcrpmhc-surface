
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
from tcrpmhc_surface.preprocessing.graph import (
    flag_contact_atoms, 
    extract_atom_data
)
from tcrpmhc_surface.dmasif.data.loader import (
    load_protein_pair
)


class TCRpMHCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pdb_dir: str, processed_dir: str): 

        # load data
        self.df = df
        self.pdb_dir = pdb_dir
        self.processed_dir = processed_dir


    def process(self, contact_threshold: float = 8.0, parse_header: bool = False):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if not os.listdir(self.processed_dir):
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
        else:
            print(f"Files already processed in {self.processed_dir}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pdb_id = str(self.df.iloc[index]['uuid'])
        protein_pair_data = load_protein_pair(pdb_id+'_tcr_pmhc', data_dir=self.processed_dir)
        return protein_pair_data
    

    

    

    

        

        

            