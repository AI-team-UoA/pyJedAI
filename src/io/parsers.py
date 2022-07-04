from tkinter.ttk import Separator
import pandas as pd
import numpy as np

import os 
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.constants import (
    COMMA_SEPARATOR
)

class Parser:

    def __init__(
            self,
            separator=None,
            names_in_first_row=None,
            attribute_names=None,
            text_clean_method=None
        ) -> None:
        self.separator = separator
        self.names_in_first_row = names_in_first_row
        self.attribute_names = attribute_names
        self.text_clean_method = text_clean_method

    def process(
            self,
            file_path
    ) -> pd.DataFrame:

        if _is_csv(file_path) and self.separator is None:
            self.dataset = pd.read_csv(os.path.abspath(file_path), sep=self.separator)

        return self.dataset

    def process_csv(self):
        # TODO
        pass

    def process_xml(self):
        # TODO
        pass


def _is_csv(file_path):
    return True if '.csv' in file_path else False

def _is_xml(file_path):
    return True if '.xml' in file_path else False