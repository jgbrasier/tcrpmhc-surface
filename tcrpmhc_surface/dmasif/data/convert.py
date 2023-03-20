from typing import Dict, List, Union, Tuple
from pathlib import Path
from Bio.PDB import PDBParser
import numpy as np
from tqdm import tqdm

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def load_structure_np(fname: str, center: bool) -> Dict[str, np.ndarray]:
    """Loads a pdb file to return a point cloud and connectivity.

    Args:
        fname: The file name of the PDB structure to be loaded.
        center: Whether to normalize the coordinates by centering the point cloud.

    Returns:
        A dictionary containing the coordinates of the atoms in the PDB structure 
        and their corresponding atomic types.
    """
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords: List[List[float]] = []
    types: List[int] = []
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num[atom.element])

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}


def convert_pdbs(pdb_dir: Union[str, Path], npy_dir: Union[str, Path]) -> None:
    """Converts PDB files to numpy arrays.

    Args:
        pdb_dir: The directory containing the PDB files.
        npy_dir: The directory to save the resulting numpy arrays.

    Returns:
        None
    """
    pdb_dir = Path(pdb_dir)
    npy_dir = Path(npy_dir)

    print("Converting PDBs")
    for p in tqdm(list(pdb_dir.glob("*.pdb"))):
        protein = load_structure_np(p, center=False)
        np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
        np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])

