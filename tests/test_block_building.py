import os
import sys

import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# 
# Datamodel
#
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

def test_standard_blocking():
    from pyjedai.block_building import StandardBlocking
    assert StandardBlocking().build_blocks(dirty_data, tqdm_disable=True) is not None
    assert StandardBlocking().build_blocks(clean_clean_data, tqdm_disable=True) is not None

def test_qgrams_blocking():
    from pyjedai.block_building import QGramsBlocking
    assert QGramsBlocking().build_blocks(dirty_data, tqdm_disable=True) is not None
    assert QGramsBlocking().build_blocks(clean_clean_data, tqdm_disable=True) is not None

def test_extended_qgrams_blocking():
    from pyjedai.block_building import ExtendedQGramsBlocking
    assert ExtendedQGramsBlocking().build_blocks(dirty_data, tqdm_disable=True) is not None
    assert ExtendedQGramsBlocking().build_blocks(clean_clean_data, tqdm_disable=True) is not None
    
def test_suffix_arrays_blocking():
    from pyjedai.block_building import SuffixArraysBlocking
    assert SuffixArraysBlocking().build_blocks(dirty_data, tqdm_disable=True) is not None
    assert SuffixArraysBlocking().build_blocks(clean_clean_data, tqdm_disable=True) is not None
    
def test_extended_suffix_arrays_blocking():
    from pyjedai.block_building import ExtendedSuffixArraysBlocking
    assert ExtendedSuffixArraysBlocking().build_blocks(dirty_data, tqdm_disable=True) is not None
    assert ExtendedSuffixArraysBlocking().build_blocks(clean_clean_data, tqdm_disable=True) is not None
