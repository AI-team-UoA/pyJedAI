import os
import sys

import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# --------------------------------- #
# Datamodel
# --------------------------------- #

from pyjedai.datamodel import Data

dirty_data = Data(
    dataset_1=pd.read_csv("data/der/cora/cora.csv", sep='|'),
    id_column_name_1='Entity Id',
    ground_truth=pd.read_csv("data/der/cora/cora_gt.csv", sep='|', header=None),
)
dirty_data.print_specs()

def test_datamodel_dirty_er():
    assert dirty_data is not None

clean_clean_data = Data(
    dataset_1=pd.read_csv("data/test/ccer/abt_100.csv", sep='|', engine='python').astype(str),
    id_column_name_1='id',
    dataset_2=pd.read_csv("data/test/ccer/buy_100.csv", sep='|', engine='python').astype(str),
    id_column_name_2='id',
    ground_truth=pd.read_csv("data/test/ccer/gt_100.csv", sep='|', engine='python')
)
clean_clean_data.print_specs()

def test_datamodel_clean_clean_er():
    assert clean_clean_data is not None

# --------------------------------- #
# Block building
# --------------------------------- #

from pyjedai.block_building import StandardBlocking
dblocks = StandardBlocking().build_blocks(dirty_data)
ccblocks = StandardBlocking().build_blocks(clean_clean_data)

# --------------------------------- #
# Block cleaning
# --------------------------------- #
from pyjedai.block_cleaning import BlockFiltering
dblocks = BlockFiltering().process(dblocks, dirty_data)
ccblocks =BlockFiltering().process(ccblocks, clean_clean_data)

from pyjedai.block_cleaning import BlockPurging
dblocks = BlockPurging().process(dblocks, dirty_data)
ccblocks =BlockPurging().process(ccblocks, clean_clean_data)

# --------------------------------- #
# Comparison cleaning
# --------------------------------- #

def test_WeightedEdgePruning():
    from pyjedai.comparison_cleaning import WeightedEdgePruning
    assert WeightedEdgePruning().process(dblocks, dirty_data) is not None
    assert WeightedEdgePruning().process(ccblocks, clean_clean_data) is not None

def test_WeightedNodePruning():
    from pyjedai.comparison_cleaning import WeightedNodePruning
    assert WeightedNodePruning().process(dblocks, dirty_data) is not None
    assert WeightedNodePruning().process(ccblocks, clean_clean_data) is not None

def test_CardinalityEdgePruning():
    from pyjedai.comparison_cleaning import CardinalityEdgePruning
    assert CardinalityEdgePruning().process(dblocks, dirty_data) is not None
    assert CardinalityEdgePruning().process(ccblocks, clean_clean_data) is not None

def test_CardinalityNodePruning():
    from pyjedai.comparison_cleaning import CardinalityNodePruning
    assert CardinalityNodePruning().process(dblocks, dirty_data) is not None
    assert CardinalityNodePruning().process(ccblocks, clean_clean_data) is not None

def test_BLAST():
    from pyjedai.comparison_cleaning import BLAST
    assert BLAST().process(dblocks, dirty_data) is not None
    assert BLAST().process(ccblocks, clean_clean_data) is not None

def test_ReciprocalCardinalityNodePruning():
    from pyjedai.comparison_cleaning import ReciprocalCardinalityNodePruning
    assert ReciprocalCardinalityNodePruning().process(dblocks, dirty_data) is not None
    assert ReciprocalCardinalityNodePruning().process(ccblocks, clean_clean_data) is not None

def test_ReciprocalWeightedNodePruning():
    from pyjedai.comparison_cleaning import ReciprocalWeightedNodePruning
    assert ReciprocalWeightedNodePruning().process(dblocks, dirty_data) is not None
    assert ReciprocalWeightedNodePruning().process(ccblocks, clean_clean_data) is not None

def test_ComparisonPropagation():
    from pyjedai.comparison_cleaning import ComparisonPropagation
    assert ComparisonPropagation().process(dblocks, dirty_data) is not None
    assert ComparisonPropagation().process(ccblocks, clean_clean_data) is not None
