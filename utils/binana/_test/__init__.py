# This file is part of BINANA, released under the Apache 2.0 License. See
# LICENSE.md or go to https://opensource.org/licenses/Apache-2.0 for full
# details. Copyright 2021 Jacob D. Durrant.

# Not putting shim here because should only be run from commandline/python (not
# javascript).

import os
from binana import _start
import binana
import glob
import re


def _remove_lines_with_pass(txt):
    for val in [
        "ligand",
        "output_dir",
        "output_file",
        "output_json",
        "output_csv",
        "receptor",
        "test",
    ]:
        txt = re.sub(r"^REMARK +?" + val + r".+?\n", "", txt, flags=re.M | re.S)
        txt = re.sub(r"^ +?" + val + r".+?\n", "", txt, flags=re.M | re.S)

        # "Rounds" numbers (to account for minor system differences)
        txt = re.sub(r"([0-9]\.[0-9]{4})[0-9]{1,15}", r"\1", txt)
    return txt


def _run_test(cmd_params):
    cur_dir = os.path.dirname(__file__) + os.sep
    for test_dir in glob.glob(cur_dir + "test_data" + os.sep + "/*"):
        test_dir = test_dir + os.sep
        lig = glob.glob(test_dir + "input" + os.sep + "ligand.*")[0]
        rec = glob.glob(test_dir + "input" + os.sep + "receptor.*")[0]

        out_dir = cur_dir + "output" + os.sep
        out_expected_dir = test_dir + "expected_output" + os.sep

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Modify the parameters in preparation or the test.
        cmd_params.params["receptor"] = rec
        cmd_params.params["ligand"] = lig
        cmd_params.params["test"] = False
        cmd_params.params["output_dir"] = out_dir

        args = []
        for arg in cmd_params.params:
            if arg == "test":
                continue
            args.append("-" + arg)
            args.append(cmd_params.params[arg])

        binana.run(args)
        
        print("=" * 80)
        print("TEST: " + os.path.basename(test_dir[:-1]).strip() + "\n")

        with open(test_dir + "info.txt") as f:
            print(f.read().strip())
            print("")

        for out_file in glob.glob(out_dir + "*"):
            expect_file = out_expected_dir + os.path.basename(out_file)

            out_txt = open(out_file).read()
            expect_txt = open(expect_file).read()

            out_txt = _remove_lines_with_pass(out_txt)
            expect_txt = _remove_lines_with_pass(expect_txt)

            if out_txt == expect_txt:
                print("PASS: " + os.path.basename(out_file))
            else:
                print("FAIL: " + os.path.basename(out_file))
                print("    Contents different:")
                print("        " + out_file)
                print("        " + expect_file)

        # Delete output files (clean up)
        # for ext in [".pdb", "state.vmd", "output.json", "log.txt"]:
        #     for fl in glob.glob(out_dir + "*" + ext):
        #         os.unlink(fl)

        print("")
        try:
            raw_input("Enter for next test > ")
        except:
            input("Enter for next test > ")
