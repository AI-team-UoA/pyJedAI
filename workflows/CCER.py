'''
 Main workflow for Clean-Clean ER
'''
# --- Libs import --- #
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.tokenizer import Tokenizer, cora_text_cleaning_method
from src.blocks.building import StandardBlocking, QGramsBlocking
from src.blocks.cleaning import BlockFiltering
from src.blocks.comparison_cleaning import WeightedEdgePruning
from src.core.entities import Data
from src.blocks.utils import print_blocks, print_candidate_pairs
from src.matching.similarity import EntityMatching

# --- Read the dataset --- #

IS_DIRTY_ER = False

data = Data(
    dataset_1=pd.read_csv(
        "../data/cora/cora.csv",
        usecols=['title'],
        sep='|'
    ),
    dataset_2=pd.read_csv(
        "../data/cora/cora.csv",
        usecols=['author'],
        nrows=100,
        sep='|'
    ),
    ground_truth=pd.read_csv("../data/cora/cora_gt.csv", sep='|')
)

# --- Block Building techniques --- #

SB = StandardBlocking(text_cleaning_method=cora_text_cleaning_method)
blocks = SB.build_blocks(data)

# --- Block Filtering --- #

BF = BlockFiltering(ratio=0.9)
blocks = BF.process(blocks, data)

# print_blocks(blocks, IS_DIRTY_ER)


# --- META-Blocking -- #

# WE = WeightedEdgePruning()
# candidate_pairs_blocks = WE.process(blocks, data)

# print_candidate_pairs(candidate_pairs_blocks)

# --- Entity Matching --- #
EM = EntityMatching('jaccard')
pairs_graph = EM.predict(blocks, data)
