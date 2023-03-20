import torch
from scipy.spatial.transform import Rotation


tensor = torch.FloatTensor
inttensor = torch.LongTensor

class RandomRotationPairAtoms(object):
    r"""Randomly rotate a protein"""

    def __call__(self, data):
        R1 = tensor(Rotation.random().as_matrix())
        R2 = tensor(Rotation.random().as_matrix())

        data.atom_coords_p1 = torch.matmul(R1, data.atom_coords_p1.T).T
        data.xyz_p1 = torch.matmul(R1, data.xyz_p1.T).T
        data.normals_p1 = torch.matmul(R1, data.normals_p1.T).T

        data.atom_coords_p2 = torch.matmul(R2, data.atom_coords_p2.T).T
        data.xyz_p2 = torch.matmul(R2, data.xyz_p2.T).T
        data.normals_p2 = torch.matmul(R2, data.normals_p2.T).T

        data.rand_rot1 = R1
        data.rand_rot2 = R2
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class CenterPairAtoms(object):
    r"""Centers a protein"""

    def __call__(self, data):
        atom_center1 = data.atom_coords_p1.mean(dim=-2, keepdim=True)
        atom_center2 = data.atom_coords_p2.mean(dim=-2, keepdim=True)

        data.atom_coords_p1 = data.atom_coords_p1 - atom_center1
        data.atom_coords_p2 = data.atom_coords_p2 - atom_center2

        data.xyz_p1 = data.xyz_p1 - atom_center1
        data.xyz_p2 = data.xyz_p2 - atom_center2

        data.atom_center1 = atom_center1
        data.atom_center2 = atom_center2
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class NormalizeChemFeatures(object):
    r"""Centers a protein"""

    def __call__(self, data):
        pb_upper = 3.0
        pb_lower = -3.0

        chem_p1 = data.chemical_features_p1
        chem_p2 = data.chemical_features_p2

        pb_p1 = chem_p1[:, 0]
        pb_p2 = chem_p2[:, 0]
        hb_p1 = chem_p1[:, 1]
        hb_p2 = chem_p2[:, 1]
        hp_p1 = chem_p1[:, 2]
        hp_p2 = chem_p2[:, 2]

        # Normalize PB
        pb_p1 = torch.clamp(pb_p1, pb_lower, pb_upper)
        pb_p1 = (pb_p1 - pb_lower) / (pb_upper - pb_lower)
        pb_p1 = 2 * pb_p1 - 1

        pb_p2 = torch.clamp(pb_p2, pb_lower, pb_upper)
        pb_p2 = (pb_p2 - pb_lower) / (pb_upper - pb_lower)
        pb_p2 = 2 * pb_p2 - 1

        # Normalize HP
        hp_p1 = hp_p1 / 4.5
        hp_p2 = hp_p2 / 4.5

        data.chemical_features_p1 = torch.stack([pb_p1, hb_p1, hp_p1]).T
        data.chemical_features_p2 = torch.stack([pb_p2, hb_p2, hp_p2]).T

        return data