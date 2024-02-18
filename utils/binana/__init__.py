# This file is part of BINANA, released under the Apache 2.0 License. See
# LICENSE.md or go to https://opensource.org/licenses/Apache-2.0 for full
# details. Copyright 2021 Jacob D. Durrant.

import binana  # Leave this for javascript conversion.
from binana import fs  # Leave
from binana import load_ligand_receptor  # Leave
from binana import interactions  # Leave
from binana import output  # Leave
from binana import _start
from binana._cli_params import _get_params

# __pragma__ ('skip')
# Python
from binana._test import _run_test
import sys as _sys
# __pragma__ ('noskip')

"""?
# Transcrypt
_sys = binana.sys
?"""

def run(args=None):
    """Gets all the interactions between a ligand and receptor, per the
    parameters specified in ``args``. If ``args`` is not ``None``, it should
    look like this::

        [
            "-receptor", "receptor.pdb",
            "-ligand", "ligand.pdb",
            "-close_contacts_dist1_cutoff", "2.5",
            "-close_contacts_dist2_cutoff", "4",
            "-electrostatic_dist_cutoff", "4",
            "-active_site_flexibility_dist_cutoff", "4",
            "-hydrophobic_dist_cutoff", "4",
            "-hydrogen_bond_dist_cutoff", "4",
            "-hydrogen_halogen_bond_angle_cutoff", "40",
            "-halogen_bond_dist_cutoff", "5.5",
            "-pi_padding_dist", "0.75",
            "-pi_pi_interacting_dist_cutoff", "7.5",
            "-pi_stacking_angle_tolerance", "30",
            "-T_stacking_angle_tolerance", "30",
            "-T_stacking_closest_dist_cutoff", "5",
            "-cation_pi_dist_cutoff", "6",
            "-salt_bridge_dist_cutoff", "5.5",
            "-metal_coordination_dist_cutoff", "3.5"
        ]

    If any of the parameters above are omitted, default values will be used.
    This function is most useful when using BINANA as a Python library (i.e.,
    not JavaScript).

    Args:
        args (list, optional): A list of strings corresponding to parameter
            name/value pairs. The parameter names must start with a hyphen.
            If None, uses sys.argv (command line arguments). Defaults to None.
    """

    """?
    console.warn("You probably don't want to call this using JavaScript (Python-only function).");
    ?"""

    # __pragma__ ('skip')
    binana._start._intro()
    # __pragma__ ('noskip')

    if args is None:
        # If no args provided to function, assume command-line use.
        args = _sys.argv[:]
    else:
        # Args provided. Make sure values are all strings (to standardize).
        for i, a in enumerate(args):
            args[i] = str(a)

    cmd_params = _get_params.CommandLineParameters(args)

    if cmd_params.params["test"]:
        # Run the tests, because `-test true` from command line. Not available
        # in JS module.

        # __pragma__ ('skip')
        _run_test(cmd_params)
        # __pragma__ ('noskip')
        return
    elif cmd_params.okay_to_proceed() == False:
        print(
            "Error: You need to specify the ligand and receptor PDBQT files to analyze using\nthe -receptor and -ligand tags from the command line.\n"
        )
        _sys.exit(0)
        return  # Needed for transcrypt

    if cmd_params.error != "":
        print("Warning: The following command-line parameters were not recognized:")
        print(("   " + cmd_params.error + "\n"))

    _start._get_all_interactions(cmd_params)
