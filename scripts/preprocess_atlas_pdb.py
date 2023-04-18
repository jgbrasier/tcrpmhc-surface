import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from tcrpmhc_surface.preprocessing.graph import (
    flag_contact_atoms, 
    extract_atom_data
)
from tcrpmhc_surface.preprocessing.pdb import (
    read_pdb_to_dataframe,
)

"""This is the README file for the TCR docking benchmark, a set of unbound and bound
PDB files to develop and test predictive algorithms for TCR/pMHC recognition. If
you use this benchmark in your work, please cite:

Pierce BG, Weng Z.  "A flexible docking approach for prediction of T cell receptor-
peptide-MHC complexes".  Protein Science, In Press.



Key:
 - l: ligand (pMHC) treated as ligand here
 - r: reception (TCR)
 - u: unbound
 - b: bound
"""

PDB_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/pdb/atlas_true_pdb"
PROCESSED_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/surface/atlas_true"

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

for filename in os.listdir(PDB_DIR):
    pdb_id = filename.split(".")[0]
    try:
        df, _ = read_pdb_to_dataframe(pdb_path=os.path.join(PDB_DIR, filename), parse_header=False)
        chains = list(df['chain_id'].unique())
        if "B" in chains:
            tcr_df, pmhc_df = df[df['chain_id'].isin(['D', 'E'])], df[df['chain_id'].isin(['A', 'B', 'C'])]
        else:
            tcr_df, pmhc_df = df[df['chain_id'].isin(['D', 'E'])], df[df['chain_id'].isin(['A', 'C'])]
        tcr_data = extract_atom_data(tcr_df, label_column=None)
        pmhc_data = extract_atom_data(pmhc_df, label_column=None)

        np.save(os.path.join(PROCESSED_DIR, pdb_id+"_tcr_atomxyz.npy"), tcr_data['atom_xyz'])
        np.save(os.path.join(PROCESSED_DIR, pdb_id+"_tcr_atomtypes.npy"), tcr_data['atom_types'])
        # np.save(os.path.join(PROCESSED_DIR, pdb_id+"_tcr_iface_labels.npy"), tcr_data['iface_labels'])

        np.save(os.path.join(PROCESSED_DIR, pdb_id+"_pmhc_atomxyz.npy"), pmhc_data['atom_xyz'])
        np.save(os.path.join(PROCESSED_DIR, pdb_id+"_pmhc_atomtypes.npy"), pmhc_data['atom_types'])
    except:
        print("ERROR processing", pdb_id)
    # np.save(os.path.join(PROCESSED_DIR, pdb_id+"_pmhc_iface_labels.npy"), pmhc_data['iface_labels'])
