import logging
import string
import pandas as pd

from logging import error as error
from logging import exception as exception
from logging import info as info
from logging import warning as warning

import nltk
import numpy as np
# nltk.download('punkt')
import tqdm
import re
from tqdm import tqdm

def cora_text_cleaning_method(col):
    return col.str.lower()
                # .str.split()