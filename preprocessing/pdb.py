from typing import Callable, Dict, Generator, List, Optional, Union, Tuple
import string
import re
import os

import numpy as np

from tqdm import tqdm

import pandas as pd
from biopandas.pdb import PandasPdb

import networkx as nx

from prody import parsePDBHeader

from functools import partial
from graphein.protein.utils import (
    compute_rgroup_dataframe,
    filter_dataframe,
    get_protein_name_from_filename,
    three_to_one_with_mods,
)

def find_chain_names(header: dict):
    flag_dict = {
    'tra': {'base': ['tcr', 't-cell', 't cell', 't'], 'variant': ['alpha', 'valpha', 'light']},
    'trb': {'base': ['tcr', 't-cell', 't cell', 't'], 'variant': ['beta', 'vbeta', 'heavy']},
    'b2m': ['beta-2-microglobulin', 'beta 2 microglobulin', 'b2m'],
    'epitope': ['peptide', 'epitope', 'protein', 'self-peptide', 'nuclear'],
    'mhc': ['mhc', 'hla', 'hla class i', 'mhc class i']
    }

    chain_key_dict = {k: list() for k in flag_dict}

    chain_keys = [key for key in header.keys() if len(key)==1]

    for chain in chain_keys:
        name = re.split(';|,| ', str(header[chain].name).lower())
        for key in flag_dict:
            if key in ['tra', 'trb']:
                if bool(set(name) & set(flag_dict[key]['base'])) & bool(set(name) & set(flag_dict[key]['variant'])):
                    chain_key_dict[key].append(chain)
            else:
                if bool(set(name) & set(flag_dict[key])):
                    chain_key_dict[key].append(chain)

    for k, v in chain_key_dict.items():
        if len(v)==0:
            raise ValueError('Header parsing error for key: {} in protein {}'.format(k, header['identifier']))
    return chain_key_dict

def read_pdb_to_dataframe(
    pdb_path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    model_index: int = 1,
    parse_header: bool = False,
    ) -> pd.DataFrame:
    """
    Read a PDB file or fetch it from the RCSB PDB database, and return a Pandas DataFrame
    containing the atomic coordinates and metadata.

    Args:
        pdb_path (str, optional): Path to a local PDB file to read. Defaults to None.
        pdb_code (str, optional): 4-letter PDB code to fetch from the RCSB PDB database.
            Defaults to None.
        uniprot_id (str, optional): UniProt accession number to fetch from the RCSB PDB
            database using the AlphaFold2 pipeline. Defaults to None.
        model_index (int, optional): Index of the model to extract from the PDB file, in case
            it contains multiple models. Defaults to 1.
        parse_header (bool, optional): Whether to parse the PDB header and extract metadata.
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the atomic coordinates and metadata, with one row
            per atom.

    Raises:
        NameError: If none of `pdb_path`, `pdb_code`, or `uniprot_id` is specified.
        ValueError: If no model is found for the specified `model_index`.

    Examples:
        >>> df = read_pdb_to_dataframe(pdb_code="1crn")
        >>> df.shape
        (327, 21)

    """
    if pdb_code is None and pdb_path is None and uniprot_id is None:
        raise NameError(
            "One of pdb_code, pdb_path or uniprot_id must be specified!"
        )

    if pdb_path is not None:
        atomic_df = PandasPdb().read_pdb(pdb_path)
        if parse_header:
            header = parsePDBHeader(pdb_path)
            header['chain_key_dict'] = find_chain_names(header)
        else:
            header = None
    elif uniprot_id is not None:
        atomic_df = PandasPdb().fetch_pdb(
            uniprot_id=uniprot_id, source="alphafold2-v2"
        )
    else:
        atomic_df = PandasPdb().fetch_pdb(pdb_code)

    atomic_df = atomic_df.get_model(model_index)
    if len(atomic_df.df["ATOM"]) == 0:
        raise ValueError(f"No model found for index: {model_index}")

    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]]), header

def split_af2_tcrpmhc_df(df: pd.DataFrame, chain_seq: List[str], rescale_residue_number: Optional[bool] = False):
    """
    Split a Pandas DataFrame containing the atomic coordinates of an AlphaFold2-generated
    TCR:pMHC complex into separate DataFrames for the TCR and pMHC chains, according to their
    amino acid sequences.

    Args:
        df (pd.DataFrame): A DataFrame containing the atomic coordinates of the TCR:pMHC complex.
            It should have columns for 'residue_name', 'residue_number', and 'chain_id'.
        chain_seq (List[str]): A list of amino acid sequences for the TCR:pMHC complex chains,
            in the order: pMHC, epitope, TCR alpha, TCR beta. Each sequence should correspond
            to the chain with the same index in the DataFrame.
        rescale_residue_number (bool, optional): Whether to rescale the residue numbers of each
            chain so that they start at 0. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two DataFrames, one for the TCR chain and
            one for the pMHC chain. Each DataFrame contains the atomic coordinates and metadata
            for the corresponding chain, with one row per atom.

    Raises:
        AssertionError: If the amino acid sequences in `chain_seq` do not match the sequences
            in the DataFrame.

    Examples:
        >>> df = read_pdb_to_dataframe(pdb_code="7jnl")
        >>> tcr_df, pmhc_df = split_af2_tcrpmhc_df(df, ["MKWSRALIVILVLVAGSVMAG", "SIINFEKL", "ETYYNQSE", "SGYINEQV"])
        >>> tcr_df.shape
        (748, 22)
        >>> pmhc_df.shape
        (578, 22)

    """
    d = []
    out = []
    for res in df.groupby('residue_number'):
        aa = three_to_one_with_mods(res[1]['residue_name'].drop_duplicates().values[0])
        d.append((aa, res[1]['residue_name'].index.tolist()))

    for idx, seq in enumerate(chain_seq):
        slice = []
        z = list(zip(seq, d)).copy()
        assert [x[0] for x in z] == [x[1][0] for x in z]
        for i, x in enumerate(z):
            slice += x[1][1]
            del(d[0])
        seq_df = df.iloc[slice]
        seq_df['chain_id'] = string.ascii_uppercase[idx]
        if rescale_residue_number:
            seq_df['residue_number'] = seq_df['residue_number'] - seq_df['residue_number'].min()
        out.append(seq_df)
    # from finetuned AF2 model: sequence are in order: pmhc, epitope, tra, trb
    # TODO: make this more generalizable
    pmhc_df = pd.concat((out[0], out[1]))
    tcr_df = pd.concat((out[2], out[3]))
    return tcr_df, pmhc_df

def _init_PandasPdb(df: pd.DataFrame):
    """
    from: https://github.com/BioPandas/biopandas/blob/main/biopandas/pdb/pandas_pdb.py
    ```
    def df(self, value):
        "Assign a new value to the pandas DataFrame"
        raise AttributeError(
            "Please use `PandasPdb._df = ... ` instead\n"
            "of `PandasPdb.df = ... ` if you are sure that\n"
            "you want to overwrite the `df` attribute."
        )
        # self._df = value
    ```
    """
    pdb_df = PandasPdb()
    pdb_df._df = df
    return pdb_df

def save_tcrpmc_pdb(tcr_df: pd.DataFrame, pmhc_df: pd.DataFrame, dir: str, pdb_code: str) -> None:
    """
    Save the atomic coordinates of a TCR:pMHC complex to separate PDB files for the TCR and pMHC
    chains.

    Args:
        tcr_df (pd.DataFrame): A DataFrame containing the atomic coordinates of the TCR chain.
            It should have columns for 'atom_name', 'residue_name', 'chain_id', 'residue_number',
            'x_coord', 'y_coord', and 'z_coord'.
        pmhc_df (pd.DataFrame): A DataFrame containing the atomic coordinates of the pMHC chain.
            It should have the same columns as `tcr_df`.
        dir (str): Path to the directory where the PDB files will be saved.
        pdb_code (str): 4-letter PDB code or UUID to use as the prefix for the output files.

    Returns:
        None

    Examples:
        >>> df = read_pdb_to_dataframe(pdb_code="7jnl")
        >>> tcr_df, pmhc_df = split_af2_tcrpmhc_df(df, ["MKWSRALIVILVLVAGSVMAG", "SIINFEKL", "ETYYNQSE", "SGYINEQV"])
        >>> save_tcrpmc_pdb(tcr_df, pmhc_df, "output", "7jnl")

    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    tcr_biopdb = _init_PandasPdb(tcr_df)
    pmhc_biopdb = _init_PandasPdb(pmhc_df)
    tcr_biopdb.to_pdb(os.path.join(dir, pdb_code+"_tcr.pdb"))
    pmhc_biopdb.to_pdb(os.path.join(dir, pdb_code+"_pmhc.pdb"))


if __name__ == "__main__":
    path = "data/pdb/run329_results_for_jg/model_0a0d6e5a-b7a7-4e72-b7da-de39ea0ecbfb.pdb"
    df, header = read_pdb_to_dataframe(path)
    
    print(type(df))