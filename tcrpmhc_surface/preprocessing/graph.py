
from typing import Callable, Dict, Generator, List, Optional, Union, Tuple

import numpy as np


import pandas as pd
from biopandas.pdb import PandasPdb

from graphein.protein.graphs import (
    deprotonate_structure, 
)

def get_contact_atoms(df1: pd.DataFrame, df2:pd.DataFrame = None , threshold:float = 8., deprotonate: bool=True, coord_names=['x_coord', 'y_coord', 'z_coord']):

    if df2 is not None:
        assert all(df1.columns == df2.columns), "DataFrame column names must match"
        if deprotonate:
            df1 = deprotonate_structure(df1)
            df2 = deprotonate_structure(df2)

        # Extract coordinates from dataframes
        coords1 = df1[coord_names].to_numpy()
        coords2 = df2[coord_names].to_numpy()

        # Compute pairwise distances between atoms
        dist_matrix = np.sqrt(((coords1[:, None] - coords2) ** 2).sum(axis=2))

        # Create a new dataframe containing pairs of atoms whose distance is below the threshold
        pairs = np.argwhere(dist_matrix < threshold)
        atoms1, atoms2 = df1.iloc[pairs[:, 0]], df2.iloc[pairs[:, 1]]
        atoms1_id = atoms1['chain_id'].map(str) + ":" + atoms1['residue_name'].map(str) + ":" + atoms1['residue_number'].map(str)
        atoms2_id = atoms2['chain_id'].map(str) + ":" + atoms2['residue_name'].map(str) + ":" + atoms2['residue_number'].map(str)
        node_pairs = np.vstack((atoms1_id.values, atoms2_id.values)).T
        result = pd.concat([df1.iloc[np.unique(pairs[:, 0])], df2.iloc[np.unique(pairs[:, 1])]])
    else:
        # TODO: Case where only 1 df is passed.
        raise NotImplementedError
    return result, node_pairs


def flag_contact_atoms(df1: pd.DataFrame, df2:pd.DataFrame = None , threshold:float = 8., deprotonate: bool=True, coord_names=['x_coord', 'y_coord', 'z_coord']):
    if df2 is not None:
        assert all(df1.columns == df2.columns), "DataFrame column names must match"
        if deprotonate:
            df1 = deprotonate_structure(df1)
            df2 = deprotonate_structure(df2)

        # Extract coordinates from dataframes
        coords1 = df1[coord_names].to_numpy()
        coords2 = df2[coord_names].to_numpy()

        # Compute pairwise distances between atoms
        dist_matrix = np.sqrt(((coords1[:, None] - coords2) ** 2).sum(axis=2))

        # Create a new dataframe containing pairs of atoms whose distance is below the threshold
        pairs = np.argwhere(dist_matrix < threshold)
        df1['is_contact'] = 0
        df2['is_contact'] = 0
        df1.loc[df1.index[pairs[:, 0]], 'is_contact'] = 1
        df2.loc[df2.index[pairs[:, 1]], 'is_contact'] = 1
        return df1, df2
    else:
        # TODO: Case where only 1 df is passed.
        raise NotImplementedError
        return df1
    

def get_all_residue_atoms(partial_df: pd.DataFrame, full_df: pd.DataFrame):
    assert all(partial_df.columns == full_df.columns), "DataFrame column names must match"
    res = pd.DataFrame(columns=full_df.columns)
    for chain in full_df['chain_id'].unique():
        chain_full_df = full_df[full_df['chain_id']==chain].copy()
        chain_partial_df = partial_df[partial_df['chain_id']==chain].copy()
        add = chain_full_df[chain_full_df['residue_number'].isin(chain_partial_df['residue_number'])]
        res = pd.concat((res, add))
    return res

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