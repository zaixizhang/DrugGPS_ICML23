# This file is part of BINANA, released under the Apache 2.0 License. See
# LICENSE.md or go to https://opensource.org/licenses/Apache-2.0 for full
# details. Copyright 2021 Jacob D. Durrant.

"""
This module contains functions that return information about various types of
interactions between a protein receptor and a small-molecule ligand.
"""

from binana.interactions import default_params  # leave this for javascript
from binana._utils.shim import _set_default
from binana.interactions.default_params import (
    ACTIVE_SITE_FLEXIBILITY_DIST_CUTOFF,
    CATION_PI_DIST_CUTOFF,
    CLOSE_CONTACTS_DIST1_CUTOFF,
    CLOSE_CONTACTS_DIST2_CUTOFF,
    ELECTROSTATIC_DIST_CUTOFF,
    HYDROGEN_HALOGEN_BOND_ANGLE_CUTOFF,
    HYDROGEN_BOND_DIST_CUTOFF,
    HALOGEN_BOND_DIST_CUTOFF,
    HYDROPHOBIC_DIST_CUTOFF,
    PI_PADDING_DIST,
    PI_PI_INTERACTING_DIST_CUTOFF,
    PI_STACKING_ANGLE_TOLERANCE,
    SALT_BRIDGE_DIST_CUTOFF,
    T_STACKING_ANGLE_TOLERANCE,
    T_STACKING_CLOSEST_DIST_CUTOFF,
)
import binana
import binana.interactions

from binana.interactions import _cat_pi
from binana.interactions import _salt_bridges
from binana.interactions import _pi_pi
from binana.interactions import _ligand_atom_types
from binana.interactions import _hydrogen_halogen_bonds
from binana.interactions import _hydrophobics
from binana.interactions import _flexibility
from binana.interactions import _electrostatic_energies
from binana.interactions import _close
from binana.interactions import _closest
from binana.interactions import _metal_coordination
from binana.interactions.default_params import METAL_COORDINATION_DIST_CUTOFF


def get_cation_pi(ligand, receptor, cutoff=None, pi_padding=None):
    """Identifies and counts the number of pi-cation interactions between the
    protein and ligand. Output is formatted like this::

        {
            'counts': {
                'PI-CATION_LIGAND-CHARGED_BETA': 2,
                'PI-CATION_LIGAND-CHARGED_OTHER': 2,
                'PI-CATION_RECEPTOR-CHARGED_OTHER': 1
            },
            'labels': [
                (
                    '[A:CHT(1):N1(2) / A:CHT(1):C5(1) / A:CHT(1):C6(3) / A:CHT(1):C6(4) / A:CHT(1):C7(9)]',
                    '[A:TRP(43):CG(28) / A:TRP(43):CD1(29) / A:TRP(43):NE1(31) / A:TRP(43):CE2(32) / A:TRP(43):CD2(30)]',
                    {'distance': 4.403228947034208}
                ),
                (
                    '[A:CHT(1):N1(2) / A:CHT(1):C5(1) / A:CHT(1):C6(3) / A:CHT(1):C6(4) / A:CHT(1):C7(9)]',
                    '[A:TRP(43):CE2(32) / A:TRP(43):CD2(30) / A:TRP(43):CE3(33) / A:TRP(43):CZ3(35) / A:TRP(43):CH2(36) / A:TRP(43):CZ2(34)]',
                    {'distance': 3.608228947034208}
                )
            ],
            'mol': <binana._structure.mol.Mol instance at 0x7feb20488128>
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        cutoff (float, optional): The distance cutoff. Defaults to
            CATION_PI_DIST_CUTOFF.
        pi_padding (float, optional): The amount by which the radius of each pi
            ring should be artificially expanded, to be sure to catch the
            interactions. Defaults to PI_PADDING_DIST.

    Returns:
        dict: Contains the atom tallies ("counts"),
        the binana._structure.mol.Mol object with the participating atoms
        ("mol"), and the labels to use in the log file ("labels").
    """

    cutoff = _set_default(cutoff, CATION_PI_DIST_CUTOFF)
    pi_padding = _set_default(pi_padding, PI_PADDING_DIST)

    return _cat_pi.get_cation_pi(ligand, receptor, cutoff, pi_padding)


def get_salt_bridges(ligand, receptor, cutoff=None):
    """Identifies and counts the number of salt-bridge interactions between the
    protein and ligand. Output is formatted like this::

        {
            'counts': {
                'SALT-BRIDGE_OTHER': 1,
                'SALT-BRIDGE_ALPHA': 2
            },
            'labels': [
                (
                    '[A:CHT(1):N1(2) / A:CHT(1):C5(1) / A:CHT(1):C6(3) / A:CHT(1):C6(4) / A:CHT(1):C7(9)]',
                    '[A:ASP(45):CG(53) / A:ASP(45):OD1(54) / A:ASP(45):OD2(55)]',
                    {'distance': 4.403228947034208}
                ),
                (
                    '[A:CHT(1):N1(14) / A:CHT(1):C4(13) / A:CHT(1):H2(15) / A:CHT(1):H1(16) / A:CHT(1):C2(17)]',
                    '[A:ASP(157):CG(283) / A:ASP(157):OD1(284) / A:ASP(157):OD2(285)]',
                    {'distance': 3.608228947034208}
                )
            ],
            'mol': <binana._structure.mol.Mol instance at 0x7feb20494098>
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        cutoff (float, optional): The distance cutoff. Defaults to
            SALT_BRIDGE_DIST_CUTOFF.

    Returns:
        dict: Contains the atom tallies ("counts"),
        the binana._structure.mol.Mol object with the participating atoms
        ("mol"), and the labels to use in the log file ("labels").
    """
    cutoff = _set_default(cutoff, SALT_BRIDGE_DIST_CUTOFF)

    return _salt_bridges.get_salt_bridges(ligand, receptor, cutoff)


def get_pi_pi(
    ligand,
    receptor,
    pi_pi_general_dist_cutoff=None,
    pi_stacking_angle_tol=None,
    t_stacking_angle_tol=None,
    t_stacking_closest_dist_cutoff=None,
    pi_padding=None,
):
    """Identifies and counts the number of pi-pi stacking and T-shaped
    interactions between the protein and ligand. Output is formatted like
    this::

        {
            'labels': {
                'T_stacking': [
                    (
                        '[A:CHT(1):C6(4) / A:CHT(1):C7(5) / A:CHT(1):C8(6) / A:CHT(1):C9(7) / A:CHT(1):O2(8)]',
                        '[A:PHE(233):CG(657) / A:PHE(233):CD1(658) / A:PHE(233):CE1(660) / A:PHE(233):CZ(662) / A:PHE(233):CE2(661) / A:PHE(233):CD2(659)]',
                        {'distance': 3.2176272313616425, 'angle': 78.66902972009667}
                    ),
                    (
                        '[A:CHT(1):C2(17) / A:CHT(1):O1(18) / A:CHT(1):C5(19) / A:CHT(1):C4(20) / A:CHT(1):C3(21)]',
                        '[A:TRP(43):CG(28) / A:TRP(43):CD1(29) / A:TRP(43):NE1(31) / A:TRP(43):CE2(32) / A:TRP(43):CD2(30)]',
                        {'distance': 3.8236272313616425, 'angle': 91.23102972009667}
                    )
                ],
                'pi_stacking': [
                    (
                        '[A:CHT(1):C6(4) / A:CHT(1):C7(5) / A:CHT(1):C8(6) / A:CHT(1):C9(7) / A:CHT(1):O2(8)]',
                        '[A:TRP(90):CG(100) / A:TRP(90):CD1(101) / A:TRP(90):NE1(103) / A:TRP(90):CE2(104) / A:TRP(90):CD2(102)]',
                        {'distance': 4.296775339716984, 'angle': 17.315362614715923}
                    ),
                    (
                        '[A:CHT(1):C6(4) / A:CHT(1):C7(5) / A:CHT(1):C8(6) / A:CHT(1):C9(7) / A:CHT(1):O2(8)]',
                        '[A:TRP(90):CE2(104) / A:TRP(90):CD2(102) / A:TRP(90):CE3(105) / A:TRP(90):CZ3(107) / A:TRP(90):CH2(108) / A:TRP(90):CZ2(106)]',
                        {'distance': 3.256775339716984, 'angle': 7.323362614715923}
                    )
                ]
            },
            'counts': {
                'STACKING_BETA': 2,
                'T-SHAPED_OTHER': 3
            },
            'mols': {
                'T_stacking': <binana._structure.mol.Mol instance at 0x7feb20478fc8>,
                'pi_stacking': <binana._structure.mol.Mol instance at 0x7feb20478f80>
            }
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        pi_pi_general_dist_cutoff (float, optional): The distance cutoff used
            for all pi-pi interactions (stacking and T-shaped). Defaults to
            PI_PI_INTERACTING_DIST_CUTOFF.
        pi_stacking_angle_tol (float, optional): The angle tolerance for the
            pi-pi stacking interactions. Defaults to
            PI_STACKING_ANGLE_TOLERANCE.
        t_stacking_angle_tol (float, optional): The angle tolerance for the
            T-shaped interactions. Defaults to T_STACKING_ANGLE_TOLERANCE.
        t_stacking_closest_dist_cutoff (float, optional): The distance cutoff
            for T-shaped interactions specifically. Defaults to
            T_STACKING_CLOSEST_DIST_CUTOFF.
        pi_padding (float, optional): The amount by which the radius of each pi
            ring should be artificially expanded, to be sure to catch the
            interactions. Defaults to PI_PADDING_DIST.

    Returns:
        dict: Contains the atom tallies ("counts"), the
        binana._structure.mol.Mol objects with the participating atoms
        ("mols"), and the labels to use in the log file ("labels").
    """

    pi_pi_general_dist_cutoff = _set_default(
        pi_pi_general_dist_cutoff, PI_PI_INTERACTING_DIST_CUTOFF
    )
    pi_stacking_angle_tol = _set_default(
        pi_stacking_angle_tol, PI_STACKING_ANGLE_TOLERANCE
    )
    t_stacking_angle_tol = _set_default(
        t_stacking_angle_tol, T_STACKING_ANGLE_TOLERANCE
    )
    t_stacking_closest_dist_cutoff = _set_default(
        t_stacking_closest_dist_cutoff, T_STACKING_CLOSEST_DIST_CUTOFF
    )
    pi_padding = _set_default(pi_padding, PI_PADDING_DIST)

    return _pi_pi.get_pi_pi(
        ligand,
        receptor,
        pi_pi_general_dist_cutoff,
        pi_stacking_angle_tol,
        t_stacking_angle_tol,
        t_stacking_closest_dist_cutoff,
        pi_padding,
    )


def get_ligand_atom_types(ligand):
    """Tallies the ligand atoms by atom type. Output is formatted like this::

        {
            'counts': {
                'A': 8,
                'C': 5,
                'HD': 3,
                'OA': 5,
                'N': 2
            }
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.

    Returns:
        dict: Contains the atom tallies ("counts").
    """

    return _ligand_atom_types.get_ligand_atom_types(ligand)


def get_hydrogen_bonds(ligand, receptor, dist_cutoff=None, angle_cutoff=None):
    """Identifies and counts the number of hydrogen bonds between the protein
    and ligand. Output is formatted like this::

        {
            'counts': {
                'HDONOR_RECEPTOR_SIDECHAIN_OTHER': 1,
                'HDONOR_LIGAND_SIDECHAIN_OTHER': 2
            },
            'labels': [
                ('A:CHT(1):N1(14)', 'A:CHT(1):H1(16)', 'A:ASP(157):OD2(285)', 'LIGAND'),
                ('A:CHT(1):O6(22)', 'A:ASN(156):2HD2(276)', 'A:ASN(156):ND2(274)', 'RECEPTOR'),
                ('A:CHT(1):O6(22)', 'A:CHT(1):HO6(23)', 'A:ASP(157):OD1(284)', 'LIGAND')
            ],
            'mol': <binana._structure.mol.Mol instance at 0x7feb20478518>
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        dist_cutoff (float, optional): The distance cutoff. Defaults to
            HYDROGEN_BOND_DIST_CUTOFF.
        angle_cutoff (float, optional): The angle cutoff. Defaults to
            HYDROGEN_HALOGEN_BOND_ANGLE_CUTOFF.

    Returns:
        dict: Contains the atom tallies ("counts"), a binana._structure.mol.Mol
        object with the participating atoms ("mol"), and the labels to use in
        the log file ("labels").
    """

    dist_cutoff = _set_default(dist_cutoff, HYDROGEN_BOND_DIST_CUTOFF)
    angle_cutoff = _set_default(angle_cutoff, HYDROGEN_HALOGEN_BOND_ANGLE_CUTOFF)

    return _hydrogen_halogen_bonds.get_hydrogen_bonds(
        ligand, receptor, dist_cutoff, angle_cutoff
    )


def get_halogen_bonds(ligand, receptor, dist_cutoff=None, angle_cutoff=None):
    """Identifies and counts the number of halogen bonds between the protein
    and ligand. Output is formatted like this::

        {
            'counts': {
                'HDONOR_RECEPTOR_SIDECHAIN_OTHER': 1,
                'HDONOR_LIGAND_SIDECHAIN_OTHER': 2
            },
            'labels': [
                ('A:CHT(1):N1(14)', 'A:CHT(1):H1(16)', 'A:ASP(157):OD2(285)', 'LIGAND'),
                ('A:CHT(1):O6(22)', 'A:ASN(156):2HD2(276)', 'A:ASN(156):ND2(274)', 'RECEPTOR'),
                ('A:CHT(1):O6(22)', 'A:CHT(1):HO6(23)', 'A:ASP(157):OD1(284)', 'LIGAND')
            ],
            'mol': <binana._structure.mol.Mol instance at 0x7feb20478518>
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        dist_cutoff (float, optional): The distance cutoff. Defaults to
            HALOGEN_BOND_DIST_CUTOFF.
        angle_cutoff (float, optional): The angle cutoff. Defaults to
            HYDROGEN_HALOGEN_BOND_ANGLE_CUTOFF.

    Returns:
        dict: Contains the atom tallies ("counts"), a binana._structure.mol.Mol
        object with the participating atoms ("mol"), and the labels to use in
        the log file ("labels").
    """

    dist_cutoff = _set_default(dist_cutoff, HALOGEN_BOND_DIST_CUTOFF)
    angle_cutoff = _set_default(angle_cutoff, HYDROGEN_HALOGEN_BOND_ANGLE_CUTOFF)

    return _hydrogen_halogen_bonds.get_halogen_bonds(
        ligand, receptor, dist_cutoff, angle_cutoff
    )


def get_hydrophobics(ligand, receptor, cutoff=None):
    """Identifies and counts the number of hydrophobic (C-C) interactions
    between the protein and ligand. Output is formatted like this::

        {
            'counts': {
                'SIDECHAIN_OTHER': 43,
                'SIDECHAIN_BETA': 29,
                'BACKBONE_OTHER': 2
            },
            'labels': [
                (
                    'A:CHT(1):C5(1)',
                    'A:TRP(43):CD2(30)',
                    {'distance': 4.403228947034208}
                ),
                (
                    'A:CHT(1):C5(1)',
                    'A:TRP(43):CE2(32)',
                    {'distance': 3.923228947034208}
                ),
                (
                    'A:CHT(1):C5(1)',
                    'A:TRP(43):CE3(33)',
                    {'distance': 4.123228947034208}
                )
            ],
            'mol': <binana._structure.mol.Mol instance at 0x7feb000acc68>
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        cutoff (float, optional): The distance cutoff. Defaults to
            HYDROPHOBIC_DIST_CUTOFF.

    Returns:
        dict: Contains the atom tallies ("counts"), a binana._structure.mol.Mol
        object with the participating atoms ("mol"), and the labels to use in
        the log file ("labels").
    """

    cutoff = _set_default(cutoff, HYDROPHOBIC_DIST_CUTOFF)

    return _hydrophobics.get_hydrophobics(ligand, receptor, cutoff)


def get_active_site_flexibility(ligand, receptor, cutoff=None):
    """Categorizes ligand-adjacent receptor atoms as belonging to a sidechain
    or backbone, as well as an alpha helix, beta sheet, or other secondary
    structure. Output is formatted like this::

        {

            'counts': {
                'SIDECHAIN_OTHER': 136,
                'SIDECHAIN_BETA': 72,
                'BACKBONE_OTHER': 7,
                'BACKBONE_BETA': 3,
                'SIDECHAIN_ALPHA': 18
            },
            'mols': {
                'alpha_helix': <binana._structure.mol.Mol instance at 0x7feb20438170>,
                'beta_sheet': <binana._structure.mol.Mol instance at 0x7feb204381b8>,
                'side_chain': <binana._structure.mol.Mol instance at 0x7feb20438368>,
                'other_2nd_structure': <binana._structure.mol.Mol instance at 0x7feb20438248>,
                'back_bone': <binana._structure.mol.Mol instance at 0x7feb20438320>
            }
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        cutoff (float, optional): The distance cutoff. Defaults to
            ACTIVE_SITE_FLEXIBILITY_DIST_CUTOFF.

    Returns:
        dict: Contains the atom tallies ("counts"), as well as a list of
        binana._structure.mol.Mol objects ("mols"), each with the participating
        atoms that belong to alpha helixes, beta sheets, and other,
        respectively.
    """

    cutoff = _set_default(cutoff, ACTIVE_SITE_FLEXIBILITY_DIST_CUTOFF)

    return _flexibility.get_flexibility(ligand, receptor, cutoff)


def get_electrostatic_energies(ligand, receptor, cutoff=None):
    """Calculates and tallies the electrostatic energies between receptor and
    ligand atoms that come within a given distance of each other. Output is
    formatted like this::

        {
            'counts': {
                'C_C': 49372.61585423234,
                'A_OA': -311243.9243779809
            }
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        cutoff (float, optional): The distance cutoff. Defaults to
            ELECTROSTATIC_DIST_CUTOFF.

    Returns:
        dict: Contains the tallies ("counts") of the energies by atom-type
        pair.
    """

    cutoff = _set_default(cutoff, ELECTROSTATIC_DIST_CUTOFF)

    return _electrostatic_energies.get_electrostatic_energies(ligand, receptor, cutoff)


def get_close(ligand, receptor, cutoff=None):
    """Identifies and counts the number of close protein/ligand contacts.
    Output is formatted like this::

        {
            'counts': {
                'C_C': 5,
                'A_OA': 29,
                'HD_NA': 1,
                'HD_N': 6
            },
            'labels': [
                (
                    'A:CHT(1):C5(1)',
                    'A:TRP(43):CD2(30)',
                    {'distance': 4.403228947034208}
                ),
                (
                    'A:CHT(1):C5(1)',
                    'A:TRP(43):CE2(32)',
                    {'distance': 3.923228947034208}
                ),
                (
                    'A:CHT(1):C5(1)',
                    'A:TRP(43):CE3(33)',
                    {'distance': 4.123228947034208}
                )
            ],
            'mol': <binana._structure.mol.Mol instance at 0x7feb203ce3f8>
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        cutoff (float, optional): The distance cutoff. Defaults to
            CLOSE_CONTACTS_DIST2_CUTOFF.

    Returns:
        dict: Contains the atom tallies ("counts"), a binana._structure.mol.Mol
        object with the participating atoms ("mol"), and the labels to use in
        the log file ("labels").
    """

    cutoff = _set_default(cutoff, CLOSE_CONTACTS_DIST2_CUTOFF)

    return _close.get_close(ligand, receptor, cutoff)


def get_closest(ligand, receptor, cutoff=None):
    """Identifies and counts the number of closest (very close) protein/ligand
    contacts. Output is formatted like this::

        {
            'counts': {
                'HD_OA': 8,
                'A_OA': 3
            },
            'labels': [
                (
                    'A:CHT(1):C9(7)',
                    'A:TRP(205):CB(467)',
                    {'distance': 4.403228947034208}
                ),
                (
                    'A:CHT(1):O2(8)',
                    'A:TRP(205):CG(468)',
                    {'distance': 3.923228947034208}
                )
            'mol': <binana._structure.mol.Mol instance at 0x7feb20290908>
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        cutoff (float, optional): The distance cutoff. Defaults to
            CLOSE_CONTACTS_DIST1_CUTOFF.

    Returns:
        dict: Contains the atom tallies ("counts"), a binana._structure.mol.Mol
        object with the participating atoms ("mol"), and the labels to use in
        the log file ("labels").
    """

    cutoff = _set_default(cutoff, CLOSE_CONTACTS_DIST1_CUTOFF)
    return _closest.get_closest(ligand, receptor, cutoff)


def get_metal_coordinations(ligand, receptor, cutoff=None):
    """Identifies and counts the number of metal-coordination protein/ligand
    contacts. Output is formatted like this::

        {
            'counts': {
                'N_ZN': 3,
                'O_ZN': 2
            },
            'labels': [
                (
                    'A:ZN(201):ZN(3059)',
                    'A:HIS(97):ND1(1426)',
                    {'distance': 1.974986835399159}
                ),
                (
                    'A:ZN(201):ZN(3059)',
                    'A:HIS(100):NE2(1470)',
                    {'distance': 2.0332422383965976}
                )
            ],
            'mol': <binana._structure.mol.Mol instance at 0x7feb20290908>
        }

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        cutoff (float, optional): The distance cutoff. Defaults to
            METAL_COORDINATION_DIST_CUTOFF.

    Returns:
        dict: Contains the atom tallies ("counts"), a binana._structure.mol.Mol
        object with the participating atoms ("mol"), and the labels to use in
        the log file ("labels").
    """

    cutoff = _set_default(cutoff, METAL_COORDINATION_DIST_CUTOFF)

    return _metal_coordination.get_metal_coordination(ligand, receptor, cutoff)


def get_all_interactions(
    ligand,
    receptor,
    closest_dist_cutoff=None,
    close_dist_cutoff=None,
    electrostatic_dist_cutoff=None,
    active_site_flexibility_dist_cutoff=None,
    hydrophobic_dist_cutoff=None,
    hydrogen_bond_dist_cutoff=None,
    hydrogen_halogen_bond_angle_cutoff=None,
    halogen_bond_dist_cutoff=None,
    pi_pi_general_dist_cutoff=None,
    pi_stacking_angle_tol=None,
    t_stacking_angle_tol=None,
    t_stacking_closest_dist_cutoff=None,
    cation_pi_dist_cutoff=None,
    salt_bridge_dist_cutoff=None,
    metal_coordination_dist_cutoff=None,
    pi_padding=None,
):
    """A single function to identify and characterize all BINANA-supported
    protein/ligand interactions. Output is formatted like this::

        {
            "closest": ...,
            "close": ...,
            "electrostatic_energies": ...,
            "active_site_flexibility": ...,
            "hydrophobics": ...,
            "hydrogen_bonds": ...,
            "halogen_bonds": ...,
            "ligand_atom_types": ...,
            "pi_pi": ...,
            "cat_pi": ...,
            "salt_bridges": ...,
            "metal_coordinations": ...,
        }

    where each `...` is a dictionary containing the corresponding interaction
    information (see the output of the other functions in this module).

    Args:
        ligand (binana._structure.mol.Mol): The ligand molecule to analyze.
        receptor (binana._structure.mol.Mol): The receptor molecule to analyze.
        closest_dist_cutoff (float, optional): The closest-atom distance
            cutoff. Defaults to CLOSE_CONTACTS_DIST1_CUTOFF.
        close_dist_cutoff (float, optional): The close-atom distance cutoff.
            Defaults to CLOSE_CONTACTS_DIST2_CUTOFF.
        electrostatic_dist_cutoff (float, optional): The electrostatic
            distance cutoff. Defaults to ELECTROSTATIC_DIST_CUTOFF.
        active_site_flexibility_dist_cutoff (float, optional): The
            active-site-flexibility distance cutoff. Defaults to
            ACTIVE_SITE_FLEXIBILITY_DIST_CUTOFF.
        hydrophobic_dist_cutoff (float, optional): The hydrophobic distance
            cutoff. Defaults to HYDROPHOBIC_DIST_CUTOFF.
        hydrogen_bond_dist_cutoff (float, optional): The hydrogen-bond distance
            cutoff. Defaults to HYDROGEN_BOND_DIST_CUTOFF.
        hydrogen_halogen_bond_angle_cutoff (float, optional): The hydrogen- and
            halogen-bond angle cutoff. Defaults to
            HYDROGEN_HALOGEN_BOND_ANGLE_CUTOFF.
        halogen_bond_dist_cutoff (float, optional): The halogen-bond distance
            cutoff. Defaults to HALOGEN_BOND_DIST_CUTOFF.
        pi_pi_general_dist_cutoff (float, optional): The distance cutoff used
            for all pi-pi interactions (stacking and T-shaped). Defaults to
            PI_PI_INTERACTING_DIST_CUTOFF.
        pi_stacking_angle_tol (float, optional): The angle tolerance for the
            pi-pi stacking interactions. Defaults to
            PI_STACKING_ANGLE_TOLERANCE.
        t_stacking_angle_tol (float, optional): The angle tolerance for the
            T-shaped interactions. Defaults to T_STACKING_ANGLE_TOLERANCE.
        t_stacking_closest_dist_cutoff (float, optional): The
            T-stacking_closest distance cutoff. Defaults to
            T_STACKING_CLOSEST_DIST_CUTOFF.
        cation_pi_dist_cutoff (float, optional): The cation-pi distance
            cutoff. Defaults to CATION_PI_DIST_CUTOFF.
        salt_bridge_dist_cutoff (float, optional): The salt-bridge distance
            cutoff. Defaults to SALT_BRIDGE_DIST_CUTOFF.
        metal_coordination_dist_cutoff (float, optional): The
            metal-coordination distance cutoff. Defaults to
            METAL_COORDINATION_DIST_CUTOFF.
        pi_padding (float, optional): The amount by which the radius of each pi
            ring should be artificially expanded, to be sure to catch the
            interactions. Defaults to PI_PADDING_DIST.

    Returns:
        dict: Contains the atom tallies ("counts"), binana._structure.mol.Mol
        objects with the participating atoms ("mol"), and labels to use in the
        log file ("labels"), for every BINANA interaction type.
    """

    closest_dist_cutoff = _set_default(closest_dist_cutoff, CLOSE_CONTACTS_DIST1_CUTOFF)
    close_dist_cutoff = _set_default(close_dist_cutoff, CLOSE_CONTACTS_DIST2_CUTOFF)
    electrostatic_dist_cutoff = _set_default(
        electrostatic_dist_cutoff, ELECTROSTATIC_DIST_CUTOFF
    )
    active_site_flexibility_dist_cutoff = _set_default(
        active_site_flexibility_dist_cutoff, ACTIVE_SITE_FLEXIBILITY_DIST_CUTOFF
    )
    hydrophobic_dist_cutoff = _set_default(
        hydrophobic_dist_cutoff, HYDROPHOBIC_DIST_CUTOFF
    )
    hydrogen_bond_dist_cutoff = _set_default(
        hydrogen_bond_dist_cutoff, HYDROGEN_BOND_DIST_CUTOFF
    )
    hydrogen_halogen_bond_angle_cutoff = _set_default(
        hydrogen_halogen_bond_angle_cutoff, HYDROGEN_HALOGEN_BOND_ANGLE_CUTOFF
    )
    halogen_bond_dist_cutoff = _set_default(
        halogen_bond_dist_cutoff, HALOGEN_BOND_DIST_CUTOFF
    )
    pi_pi_general_dist_cutoff = _set_default(
        pi_pi_general_dist_cutoff, PI_PI_INTERACTING_DIST_CUTOFF
    )
    pi_stacking_angle_tol = _set_default(
        pi_stacking_angle_tol, PI_STACKING_ANGLE_TOLERANCE
    )
    t_stacking_angle_tol = _set_default(
        t_stacking_angle_tol, T_STACKING_ANGLE_TOLERANCE
    )
    t_stacking_closest_dist_cutoff = _set_default(
        t_stacking_closest_dist_cutoff, T_STACKING_CLOSEST_DIST_CUTOFF
    )
    cation_pi_dist_cutoff = _set_default(cation_pi_dist_cutoff, CATION_PI_DIST_CUTOFF)
    salt_bridge_dist_cutoff = _set_default(
        salt_bridge_dist_cutoff, SALT_BRIDGE_DIST_CUTOFF
    )
    pi_padding = _set_default(pi_padding, PI_PADDING_DIST)

    # Get distance measurements between protein and ligand atom types, as
    # well as some other measurements
    closest = get_closest(ligand, receptor, closest_dist_cutoff)
    close = get_close(ligand, receptor, close_dist_cutoff)
    electrostatic_energies = get_electrostatic_energies(
        ligand, receptor, electrostatic_dist_cutoff
    )
    active_site_flexibility = get_active_site_flexibility(
        ligand, receptor, active_site_flexibility_dist_cutoff
    )
    hydrophobics = get_hydrophobics(ligand, receptor, hydrophobic_dist_cutoff)

    hydrogen_bonds = get_hydrogen_bonds(
        ligand,
        receptor,
        hydrogen_bond_dist_cutoff,
        hydrogen_halogen_bond_angle_cutoff
    )

    halogen_bonds = get_halogen_bonds(
        ligand,
        receptor,
        halogen_bond_dist_cutoff,
        hydrogen_halogen_bond_angle_cutoff
    )

    metal_coordinations = get_metal_coordinations(
        ligand, receptor, metal_coordination_dist_cutoff
    )

    ligand_atom_types = get_ligand_atom_types(ligand)

    # Count pi-pi stacking and pi-T stacking interactions
    pi_pi = get_pi_pi(
        ligand,
        receptor,
        pi_pi_general_dist_cutoff,
        pi_stacking_angle_tol,
        t_stacking_angle_tol,
        t_stacking_closest_dist_cutoff,
        pi_padding,
    )

    # Now identify cation-pi interactions
    cat_pi = get_cation_pi(ligand, receptor, cation_pi_dist_cutoff, pi_padding)

    # now count the number of salt bridges
    salt_bridges = get_salt_bridges(ligand, receptor, salt_bridge_dist_cutoff)

    # Also rotatable bonds
    num_lig_rot_bonds = ligand.rotatable_bonds_count

    return {
        "closest": closest,
        "close": close,
        "electrostatic_energies": electrostatic_energies,
        "active_site_flexibility": active_site_flexibility,
        "hydrophobics": hydrophobics,
        "hydrogen_bonds": hydrogen_bonds,
        "halogen_bonds": halogen_bonds,
        "ligand_atom_types": ligand_atom_types,
        "pi_pi": pi_pi,
        "cat_pi": cat_pi,
        "salt_bridges": salt_bridges,
        "metal_coordinations": metal_coordinations,
        "ligand_rotatable_bonds": num_lig_rot_bonds,
    }
