"""
Messagebox for user input.
"""
import tkinter as tk
from tkinter import messagebox


def get_yes_no_from_user(title: str = "", message: str = "", **kwargs) -> bool:
    """
    Shows the user a messagebox with yes/no buttons and returns the answer
    :param title: Title of the box
    :param message: Message of the box
    :param kwargs: Additional kwargs
    :return: User input (yes or no)
    """
    root = tk.Tk()
    root.withdraw()

    answer: bool = messagebox.askyesno(title=title, message=message, **kwargs)

    return answer
