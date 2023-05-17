import os
import sys

import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

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

from pyjedai.block_building import StandardBlocking
dblocks = StandardBlocking().build_blocks(dirty_data)
ccblocks = StandardBlocking().build_blocks(clean_clean_data)

# --------------------------------- #

def test_block_filtering():
    from pyjedai.block_cleaning import BlockFiltering
    assert BlockFiltering().process(dblocks, dirty_data) is not None
    assert BlockFiltering().process(ccblocks, clean_clean_data) is not None

def test_block_purging():
    from pyjedai.block_cleaning import BlockPurging
    assert BlockPurging().process(dblocks, dirty_data) is not None
    assert BlockPurging().process(ccblocks, clean_clean_data) is not None
