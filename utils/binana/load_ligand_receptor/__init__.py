# This file is part of BINANA, released under the Apache 2.0 License. See
# LICENSE.md or go to https://opensource.org/licenses/Apache-2.0 for full
# details. Copyright 2021 Jacob D. Durrant.

"""This module contains functions for loading ligands and receptors. Note that
while BINANA can process PDB files, the PDB format lacks some information
required for a full BINANA analysis. PDBQT recommended."""

import binana
from math import sqrt, pi
from binana._structure.point import Point as _Point
from binana._structure.mol import Mol as _Mol
from binana._utils import _math_functions

_ligand_receptor_dists_cache = {}
_ligand_receptor_aromatic_dists = None

# Contains all pi-pi interactions (of all types).
# pi_interactions = None


def from_texts(ligand_text, receptor_text, max_cutoff=None):
    """Loads a ligand and receptor from a PDBQT- or PDB-formatted string
    (text). PDBQT recommended.

    Args:
        ligand_text (str): The ligand text to load. Preferably PDBQT formatted,
            though BINANA and perform most analyses on PDB files as well.
        receptor_text (str): The receptor text to load. Preferably PDBQT
            formatted, though BINANA and perform most analyses on PDB files as
            well.
        max_cutoff (float, optional): If specified, will only load receptor
            atoms that fall within a cube extending this many angstroms beyond
            the ligand in the x, y, and z directions. Can dramatically speed
            calculations on large proteins, if an appropriate max_cutoff is
            known beforehand. On the other hand, may prevent proper assignment
            of secondary structure. Defaults to None, meaning load all receptor
            atoms.

    Returns:
        list: A list of binana._structure.mol.Mol objects, for the ligand and
        receptor, respectively.
    """

    ligand = _Mol()
    ligand.load_pdb_from_text(ligand_text)

    receptor = _Mol()
    if max_cutoff is None:
        # Load the full receptor (all atoms).
        receptor.load_pdb_from_text(receptor_text)
    else:
        receptor.load_pdb_from_text(
            receptor_text,
            None,
            ligand.min_x - max_cutoff,
            ligand.max_x + max_cutoff,
            ligand.min_y - max_cutoff,
            ligand.max_y + max_cutoff,
            ligand.min_z - max_cutoff,
            ligand.max_z + max_cutoff,
        )

    receptor.assign_secondary_structure()

    # Clears the cache
    _clear_cache()

    return ligand, receptor


def from_files(ligand_filename, receptor_filename, max_cutoff=None):
    """Loads a ligand and receptor from PDBQT or PDB files. PDBQT recommended.

    Args:
        ligand_pdbqt_filename (str): The ligand filename to load. Preferably
            PDBQT formatted, though BINANA and perform most analyses on PDB
            files as well.
        receptor_pdbqt_filename (str): The receptor filename to load.
            Preferably PDBQT formatted, though BINANA and perform most analyses
            on PDB files as well.
        max_cutoff (float, optional): If specified, will only load receptor
            atoms that fall within a cube extending this many angstroms beyond
            the ligand in the x, y, and z directions. Can dramatically speed
            calculations on large proteins, if an appropriate max_cutoff is
            known beforehand. On the other hand, may prevent proper assignment
            of secondary structure. Defaults to None, meaning load all receptor
            atoms.

    Returns:
        list: A list of binana._structure.mol.Mol objects, for the ligand and
        receptor, respectively.
    """

    # import pdb; pdb.set_trace()

    ligand = _Mol()
    ligand.load_pdb_file(ligand_filename)

    receptor = _Mol()
    if max_cutoff is None:
        # Load the full receptor (all atoms).
        receptor.load_pdb_file(receptor_filename)
    else:
        receptor.load_pdb_file(
            receptor_filename,
            ligand.min_x - max_cutoff,
            ligand.max_x + max_cutoff,
            ligand.min_y - max_cutoff,
            ligand.max_y + max_cutoff,
            ligand.min_z - max_cutoff,
            ligand.max_z + max_cutoff,
        )

    receptor.assign_secondary_structure()

    # Clears the cache
    _clear_cache()

    return ligand, receptor


def _clear_cache():
    global _ligand_receptor_dists_cache
    global _ligand_receptor_aromatic_dists
    # global pi_interactions

    _ligand_receptor_dists_cache = {}
    _ligand_receptor_aromatic_dists = None
    # pi_interactions = None


# cum_time = 0
# import time


def _get_coor_mol_dists(atom, coor, mol_all_atoms, max_dist_sqr, dist_inf_list):
    """Gets the distances between an atom/coordinate and the atoms of a
    molecule.

    Args:
        atom ([type]): The atom. If mol is receptor, atom is a ligand atom, and
            visa versa.
        coor ([type]): The corresponding coordinate. Should be the same as
            atom.coordinates, but I think there's a speed up by not retrieving
            the value every time.
        mol_all_atoms ([type]): All the atoms in the molecule (e.g., receptor).
        max_dist_sqr ([type]): The square of the maximum distance to consider.
        dist_inf_list ([type]): A list to store the information about the
            distances.
    """

    # if mol is receptor, atom is a ligand atom.
    for mol_atom in mol_all_atoms:
        # Try to get it from the cache. In benchmarks, the cache system
        # actually slows things down quite a bit. Especially in JavaScript,
        # but even here in Python.
        # key = (ligand_atom_index, receptor_atom_index)
        # if key in _ligand_receptor_dists_cache_keys:
        #     val = _ligand_receptor_dists_cache[key]
        #     if val[2] < max_dist:
        #         ligand_receptor_dists.append(val)
        #         continue

        # It's not in the cache, so keep looking
        mol_coor = mol_atom.coordinates

        # Doing as below because benchmarks suggestit is faster than
        # things like math.pow, math.fabs, math.abs, etc.
        delta_x = mol_coor.x - coor.x
        summed = delta_x * delta_x

        if summed > max_dist_sqr:
            continue

        delta_y = mol_coor.y - coor.y
        summed += delta_y * delta_y

        if summed > max_dist_sqr:
            continue

        delta_z = mol_coor.z - coor.z
        summed += delta_z * delta_z

        if summed > max_dist_sqr:
            continue

        dist = sqrt(summed)

        val = (atom, mol_atom, dist)
        # _ligand_receptor_dists_cache[key] = val
        dist_inf_list.append(val)


def _get_ligand_receptor_dists(ligand, receptor, max_dist, elements=None):
    # global cum_time
    # t1 = time.time()

    # global _ligand_receptor_dists_cache

    # Get all the atoms
    ligand_all_atoms_dict = ligand.all_atoms
    receptor_all_atoms_dict = receptor.all_atoms
    ligand_atom_indexes = ligand_all_atoms_dict.keys()
    receptor_atom_indexes = receptor_all_atoms_dict.keys()
    ligand_all_atoms = [ligand_all_atoms_dict[i] for i in ligand_atom_indexes]
    receptor_all_atoms = [receptor_all_atoms_dict[i] for i in receptor_atom_indexes]

    # Filter the atoms by element if needed.
    if elements is not None:
        # So elements are specified. Filter by those.
        ligand_all_atoms = [a for a in ligand_all_atoms if a.element in elements]
        receptor_all_atoms = [a for a in receptor_all_atoms if a.element in elements]

    # Use max_dist to go faster
    ligand_receptor_dists = []
    max_dist_sqr = max_dist * max_dist
    # _ligand_receptor_dists_cache_keys = _ligand_receptor_dists_cache.keys()

    for ligand_atom in ligand_all_atoms:
        ligand_coor = ligand_atom.coordinates
        _get_coor_mol_dists(
            ligand_atom,
            ligand_coor,
            receptor_all_atoms,
            max_dist_sqr,
            ligand_receptor_dists,
        )

    # cum_time += time.time() - t1
    # print(cum_time)

    return ligand_receptor_dists


def _get_ligand_receptor_aromatic_dists(ligand, receptor, pi_pi_general_dist_cutoff):
    global _ligand_receptor_aromatic_dists

    # Get it from the cache
    if _ligand_receptor_aromatic_dists is not None:
        return _ligand_receptor_aromatic_dists

    _ligand_receptor_aromatic_dists = []

    for ligand_aromatic in ligand.aromatic_rings:
        for receptor_aromatic in receptor.aromatic_rings:
            dist = ligand_aromatic.center.dist_to(receptor_aromatic.center)
            if dist < pi_pi_general_dist_cutoff:
                # so there could be some pi-pi interactions. first, let's
                # check for stacking interactions. Are the two pi's
                # roughly parallel?
                ligand_aromatic_norm_vector = _Point(
                    ligand_aromatic.plane_coeff[0],
                    ligand_aromatic.plane_coeff[1],
                    ligand_aromatic.plane_coeff[2],
                )

                receptor_aromatic_norm_vector = _Point(
                    receptor_aromatic.plane_coeff[0],
                    receptor_aromatic.plane_coeff[1],
                    receptor_aromatic.plane_coeff[2],
                )

                angle_between_planes = (
                    _math_functions.angle_between_points(
                        ligand_aromatic_norm_vector, receptor_aromatic_norm_vector
                    )
                    * 180.0
                    / pi
                )

                _ligand_receptor_aromatic_dists.append(
                    (
                        ligand_aromatic,
                        receptor_aromatic,
                        dist,
                        angle_between_planes,
                    )
                )

    return _ligand_receptor_aromatic_dists
