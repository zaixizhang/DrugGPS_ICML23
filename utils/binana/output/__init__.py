# This file is part of BINANA, released under the Apache 2.0 License. See
# LICENSE.md or go to https://opensource.org/licenses/Apache-2.0 for full
# details. Copyright 2021 Jacob D. Durrant.

"""A few functions to output BINANA analysis."""

from binana.output import _directory
from binana.output import dictionary
from binana.output import csv
from binana.output import _log
from binana.output import pdb_file
import binana

# __pragma__ ('skip')
# Python, just alias open
_openFile = open
import json as _json

# __pragma__ ('noskip')

"""?
# Transcrypt
import binana._utils.shim as _json
from binana._utils.shim import OpenFile as _openFile
?"""


def _write_main(
    parameters,
    ligand,
    receptor,
    closest,
    close,
    hydrophobics,
    hydrogen_bonds,
    halogen_bonds,
    salt_bridges,
    metal_coordinations,
    pi_pi,
    cat_pi,
    electrostatic_energies,
    active_site_flexibility,
    ligand_atom_types,
):
    """The main function for writing BINANA output to the disk (or the
    in-memory "fake" file system if using the JavaScript library). Output
    depends on the values in the ``parameters`` object (see
    :py:func:`~binana.run`).

    To write output files to a directory (and to create the VMD state file and
    supporting files required for VMD visualization)::

        -output_dir ./output_for_vmd/

    To write to a single PDB file::

        -output_file test.pdb

    To save data to a JSON file::

        -output_json test.json

    Args:
        parameters (binana._cli_params.get_params.CommandLineParameters): An
            object containing the user-specified parameters. See
            :py:func:`~binana.run`.
        ligand (binana._structure.mol.Mol): The ligand object.
        receptor (binana._structure.mol.Mol): The receptor object.
        closest (dict): A dictionary containing information about the closest
            protein/ligand interactions.
        close (dict): A dictionary containing information about the close
            protein/ligand interactions.
        hydrophobics (dict): A dictionary containing information about the
            hydrophobic protein/ligand interactions.
        hydrogen_bonds (dict): A dictionary containing information about the
            hydrogen bonds between the protein and ligand.
        halogen_bonds (dict): A dictionary containing information about the
            halogen bonds between the protein and ligand.
        salt_bridges (dict): A dictionary containing information about the
            salt-bridges protein/ligand interactions.
        metal_coordinations (dict): A dictionary containing information about
            the metal-coordination protein/ligand interactions.
        pi_pi (dict): A dictionary containing information about the pi-pi
            (stacking and T-shaped) protein/ligand interactions.
        cat_pi (dict): A dictionary containing information about the pi-cation
            protein/ligand interactions.
        electrostatic_energies (dict): A dictionary containing information
            about the electrostatic energies between protein and ligand atoms.
        active_site_flexibility (dict): A dictionary containing information
            about the flexibility of ligand-adjacent protein atoms.
        ligand_atom_types (dict): A dictionary containing information about
            the ligand atom types.
    """

    # call json_file have it return the dictionary and dump to a json file.
    # You'll use this regardless of directory or single file output.

    json_output = binana.output.dictionary.collect(
        closest,
        close,
        hydrophobics,
        hydrogen_bonds,
        halogen_bonds,
        salt_bridges,
        metal_coordinations,
        pi_pi,
        cat_pi,
        electrostatic_energies,
        active_site_flexibility,
        ligand_atom_types,
        ligand_rotatable_bonds=ligand.rotatable_bonds_count,
    )

    # Get the log text. You'll also use this regardless of single file or
    # directory output.
    log_output = binana.output._log.collect(
        parameters,
        ligand,
        closest,
        close,
        hydrophobics,
        hydrogen_bonds,
        halogen_bonds,
        salt_bridges,
        metal_coordinations,
        pi_pi,
        cat_pi,
        electrostatic_energies,
        active_site_flexibility,
        ligand_atom_types,
        json_output,
    )

    if parameters.params["output_csv"] != "":
        csv_txt = csv.collect(json_output)
        f = _openFile(parameters.params["output_csv"], "w")
        f.write(csv_txt)
        f.close()

    if parameters.params["output_json"] != "":
        f = _openFile(parameters.params["output_json"], "w")
        f.write(
            _json.dumps(json_output, indent=2, sort_keys=True, separators=(",", ": "))
        )
        f.close()

    if parameters.params["output_file"] != "":
        # Be sure to always keep this before make_directory_output below,
        # because it is in write() that residue names get changed per the
        # interaction type.
        binana.output.pdb_file.write(
            ligand,
            receptor,
            closest,
            close,
            hydrophobics,
            hydrogen_bonds,
            halogen_bonds,
            salt_bridges,
            metal_coordinations,
            pi_pi,
            cat_pi,
            active_site_flexibility,
            log_output,
            False,
            parameters.params["output_file"],
        )

    if parameters.params["output_dir"] != "":
        _directory.make_directory_output(
            parameters,
            closest,
            close,
            active_site_flexibility,
            hydrophobics,
            hydrogen_bonds,
            halogen_bonds,
            pi_pi,
            cat_pi,
            salt_bridges,
            metal_coordinations,
            ligand,
            receptor
        )
