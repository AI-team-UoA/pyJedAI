import logging
import os
import sys

import networkx
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from operator import methodcaller
from typing import Callable, Dict, List