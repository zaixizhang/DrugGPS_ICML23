# This file is part of BINANA, released under the Apache 2.0 License. See
# LICENSE.md or go to https://opensource.org/licenses/Apache-2.0 for full
# details. Copyright 2021 Jacob D. Durrant.

"""When used as a JavaScript library, BINANA cannot access the local file system
to save and load files. These functions save and read files to a fake in-memory
file system for use in the browser."""

import binana  # Leave this for javascript conversion.
from binana import _utils
from binana._utils import shim
from binana._utils.shim import OpenFile


def save_file(filename, text):
    """Save a file to the fake (in-memory) file system. This is for use with
    transcrypt.

    Args:
        filename (string): The filename.
        text (string): The contents of the file to save.
    """

    f = OpenFile(filename, "w")
    f.write(text)
    f.close()


def ls():
    """List all files in the fake (in-memory) file system. For transcrypt."""

    print(shim.fake_fs.keys())


def load_file(filename):
    """Load a file from the fake (in-memory) file system. For transcrypt.

    Args:
        filename (string): The filename to load.

    Returns:
        string: The contents of the file.
    """

    f = OpenFile(filename, "r")
    txt = f.read()
    f.close()

    return txt
