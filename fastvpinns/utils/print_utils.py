# File for Printing utilities
# of all the cells within the given mesh
# Author: Thivin Anandh D
# Date:  02/Nov/2023

from rich.console import Console
from rich.table import Table


def print_table(title, columns, col_1_values, col_2_values):
    """This function prints a table with two columns to the console.
    :param title: Title of the table
    :type title: str
    :param columns: List of column names
    :type columns: list
    :param col_1_values: List of values for column 1
    :type col_1_values: list
    :param col_2_values: List of values for column 2
    :type col_2_values: list
    :return: None
    :rtype: None
    """

    # Create a console object
    console = Console()

    # Create a table with a title
    table = Table(show_header=True, header_style="bold magenta", title=title)

    # Add columns to the table
    for column in columns:
        table.add_column(column)

    # Add rows to the table
    for val_1, val_2 in zip(col_1_values, col_2_values):
        # check if val_2 is a float
        if isinstance(val_2, float):
            # add the row to the table
            table.add_row(val_1, f"{val_2:.4f}")
        else:
            table.add_row(val_1, str(val_2))

    # Print the table to the console
    console.print(table)
