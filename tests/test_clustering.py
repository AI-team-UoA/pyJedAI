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


from pyjedai.block_building import StandardBlocking
dblocks = StandardBlocking().build_blocks(dirty_data)
ccblocks = StandardBlocking().build_blocks(clean_clean_data)

from pyjedai.block_cleaning import BlockFiltering
dblocks = BlockFiltering().process(dblocks, dirty_data)
ccblocks = BlockFiltering().process(ccblocks, clean_clean_data)

from pyjedai.block_cleaning import BlockPurging
dblocks = BlockPurging().process(dblocks, dirty_data)
ccblocks = BlockPurging().process(ccblocks, clean_clean_data)

from pyjedai.comparison_cleaning import WeightedEdgePruning
dblocks = WeightedEdgePruning().process(dblocks, dirty_data)
ccblocks = WeightedEdgePruning().process(ccblocks, clean_clean_data)

from pyjedai.matching import EntityMatching
dgraph = EntityMatching().predict(dblocks, dirty_data)
ccgraph = EntityMatching().predict(ccblocks, clean_clean_data)

# --------------------------------- #
# Clustering
# --------------------------------- #

def test_ConnectedComponentsClustering():
    from pyjedai.clustering import ConnectedComponentsClustering
    assert ConnectedComponentsClustering().process(dgraph, dirty_data) is not None
    assert ConnectedComponentsClustering().process(ccgraph, clean_clean_data) is not None

def test_UniqueMappingClustering():
    from pyjedai.clustering import UniqueMappingClustering
    assert UniqueMappingClustering().process(ccgraph, clean_clean_data) is not None

test_UniqueMappingClustering()