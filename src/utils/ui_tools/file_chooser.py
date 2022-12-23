"""
Filedialog for config selection.
"""
import tkinter as tk
from tkinter import filedialog
from pathlib import Path


def get_filepath_from_filedialog(title: str = "Open file", initial_directory: Path = None,
                                 type_description: dict = None) -> Path:
    """
    Opens a filedialog window and returns the path of the file selected by the user
    :param title: Title of the window
    :param initial_directory: Start directory of the selection dialog
    :param type_description: Description of the filetype to be selected
    :return: Path of the selected file
    """
    root = tk.Tk()
    root.withdraw()

    # Default initialization of mutable parameters - code safety.
    if type_description is None:
        type_description = {"yaml": "YAML Files"}

    filepath_string = filedialog.askopenfilename(
        title=title,
        initialdir=initial_directory,
        filetypes=[(value, key) for key, value in type_description.items()]
    )

    filepath = None
    if filepath_string:
        filepath = Path(filepath_string)

    return filepath
