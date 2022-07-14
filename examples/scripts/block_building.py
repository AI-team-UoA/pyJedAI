import os
import sys
import pandas as pd
import networkx
from networkx import (
    draw,
    DiGraph,
    Graph,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
from utils.tokenizer import cora_text_cleaning_method
from utils.utils import print_clusters
from blocks.utils import print_blocks, print_candidate_pairs
from evaluation.scores import Evaluation
from datamodel import Data

from blocks.building import (
    StandardBlocking,
    QGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking,
    ExtendedQGramsBlocking
)


# ---------------------------------------- #
# --------------- Dirty ER --------------- #
# ---------------------------------------- #

d1 = pd.read_csv("../data/cora/cora.csv", sep='|')
gt = pd.read_csv("../data/cora/cora_gt.csv", sep='|', header=None)
attr = ['Entity Id','author', 'title']

data = Data(
    dataset_1=d1,
    id_column_name_1='Entity Id',
    ground_truth=gt,
    attributes_1=attr
)

print("\n\nDirty ER in CORA dataset:\n")
data.process(cora_text_cleaning_method)
data.print_specs()
print("\n- StandardBlocking")
blocks = StandardBlocking().build_blocks(data)
Evaluation(data).report(blocks)

print("\n- QGramsBlocking")
blocks = QGramsBlocking(
    qgrams=2
).build_blocks(data)
Evaluation(data).report(blocks)

print("\n- ExtendedQGramsBlocking")
blocks = ExtendedQGramsBlocking(
    qgrams=2,
    threshold=0.9
).build_blocks(data)
Evaluation(data).report(blocks)


# ---------------------------------------------- #
# --------------- Clean-Clean ER --------------- #
# ---------------------------------------------- #

d1 = pd.read_csv("../data/D2/abt.csv", sep='|', engine='python').astype(str)
d2 = pd.read_csv("../data/D2/buy.csv", sep='|', engine='python').astype(str)
gt = pd.read_csv("../data/D2/gt.csv", sep='|', engine='python')

data = Data(
    dataset_1=d1,
    attributes_1=['id','name','description'],
    id_column_name_1='id',
    dataset_2=d2,
    attributes_2=['id','name','description'],
    id_column_name_2='id',
    ground_truth=gt,
)

print("\n\nClean-Clean ER in ABT-BUY dataset:\n")
data.process(cora_text_cleaning_method)
data.print_specs()
print("\n- StandardBlocking")
blocks = StandardBlocking().build_blocks(data)
Evaluation(data).report(blocks)

print("\n- QGramsBlocking")
blocks = QGramsBlocking(
    qgrams=2
).build_blocks(data)
Evaluation(data).report(blocks)

print("\n- ExtendedQGramsBlocking")
blocks = ExtendedQGramsBlocking(
    qgrams=2,
    threshold=0.9
).build_blocks(data)
Evaluation(data).report(blocks)