import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from pyjedai.datamodel import Data
from pyjedai.block_building import (
    StandardBlocking,
    QGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking,
    ExtendedQGramsBlocking
)

dirty_data = Data(
    dataset_1=pd.read_csv("data/cora/cora.csv", sep='|'),
    id_column_name_1='Entity Id',
    ground_truth=pd.read_csv("data/cora/cora_gt.csv", sep='|', header=None),
    attributes_1=['Entity Id', 'author', 'title']
)
dirty_data.process()
dirty_data.print_specs()

def test_datamodel_dirty_er():
    assert dirty_data is not None

clean_clean_data = Data(
    dataset_1=pd.read_csv("data/D2/abt.csv", sep='|', engine='python').astype(str),
    attributes_1=['id', 'name', 'description'],
    id_column_name_1='id',
    dataset_2=pd.read_csv("data/D2/buy.csv", sep='|', engine='python').astype(str),
    attributes_2=['id', 'name', 'description'],
    id_column_name_2='id',
    ground_truth=pd.read_csv("data/D2/gt.csv", sep='|', engine='python')
)
clean_clean_data.process()
clean_clean_data.print_specs()

def test_datamodel_clean_clean_er():
    assert clean_clean_data is not None

def test_standard_blocking():
    assert StandardBlocking().build_blocks(dirty_data, tqdm_disable=True) is not None
    assert StandardBlocking().build_blocks(clean_clean_data, tqdm_disable=True) is not None

def test_qgrams_blocking():
    assert QGramsBlocking().build_blocks(dirty_data, tqdm_disable=True) is not None
    assert QGramsBlocking().build_blocks(clean_clean_data, tqdm_disable=True) is not None
    
def test_extended_qgrams_blocking():
    assert ExtendedQGramsBlocking().build_blocks(dirty_data, tqdm_disable=True) is not None
    assert ExtendedQGramsBlocking().build_blocks(clean_clean_data, tqdm_disable=True) is not None
    
def test_suffix_arrays_blocking():
    assert SuffixArraysBlocking().build_blocks(dirty_data, tqdm_disable=True) is not None
    assert SuffixArraysBlocking().build_blocks(clean_clean_data, tqdm_disable=True) is not None
    
def test_extended_suffix_arrays_blocking():
    assert ExtendedSuffixArraysBlocking().build_blocks(dirty_data, tqdm_disable=True) is not None
    assert ExtendedSuffixArraysBlocking().build_blocks(clean_clean_data, tqdm_disable=True) is not None
