from pykeops.torch import LazyTensor
from tcrpmhc_surface.dmasif.data.geometry import (
    atoms_to_points_normals,
)
from tcrpmhc_surface.dmasif.utils import diagonal_ranges

def project_iface_labels(P, threshold=2.0):

    queries = P["xyz"]
    batch_queries = P["batch"]
    source = P["mesh_xyz"]
    batch_source = P["mesh_batch"]
    labels = P["mesh_labels"]
    x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(source[None, :, :])  # (1, M, D)

    D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # (N, M)
    D_ij.ranges = diagonal_ranges(batch_queries, batch_source)
    nn_i = D_ij.argmin(dim=1).view(-1)  # (N,)
    nn_dist_i = (
        D_ij.min(dim=1).view(-1, 1) < threshold
    ).float()  # If chain is not connected because of missing densities MaSIF cut out a part of the protein
    query_labels = labels[nn_i] * nn_dist_i
    P["labels"] = query_labels

    return P

def preprocess_surface(P, resolution=1.0, sup_sampling=20, distance=1.05):
    P["xyz"], P["normals"], P["batch"] = atoms_to_points_normals(
        P["atoms"],
        P["batch_atoms"],
        atomtypes=P["atomtypes"],
        resolution=resolution,
        sup_sampling=sup_sampling,
        distance=distance,
    )
    if P['mesh_labels'] is not None:
        P = project_iface_labels(P)
    return P


def process_single(protein_pair, chain_idx=1):
    """Turn the PyG data object into a dict."""

    P = {}
    with_mesh = "face_p1" in protein_pair.keys
    preprocessed = "gen_xyz_p1" in protein_pair.keys

    if chain_idx == 1:
        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p1 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p1_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p1 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p1 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p1
        P["batch_atoms"] = protein_pair.atom_coords_p1_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p1
        P["atomtypes"] = protein_pair.atom_types_p1

        P["xyz"] = protein_pair.gen_xyz_p1 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p1 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p1 if preprocessed else None
        P["labels"] = protein_pair.gen_labels_p1 if preprocessed else None

    elif chain_idx == 2:
        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p2 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p2_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p2 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p2 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p2
        P["batch_atoms"] = protein_pair.atom_coords_p2_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p2
        P["atomtypes"] = protein_pair.atom_types_p2

        P["xyz"] = protein_pair.gen_xyz_p2 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p2 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p2 if preprocessed else None
        P["labels"] = protein_pair.gen_labels_p2 if preprocessed else None

    return P

def process(args, protein_pair, net):
    P1 = process_single(protein_pair, chain_idx=1)
    if not "gen_xyz_p1" in protein_pair.keys:
        preprocess_surface(P1)
        #if P1["mesh_labels"] is not None:
        #    project_iface_labels(P1)
    P2 = None
    if not args.single_protein:
        P2 = process_single(protein_pair, chain_idx=2)
        if not "gen_xyz_p2" in protein_pair.keys:
            preprocess_surface(P2)
            #if P2["mesh_labels"] is not None:
            #    project_iface_labels(P2)

    return P1, P2