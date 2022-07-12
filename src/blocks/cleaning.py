import logging
import os
import sys

import pandas as pd
import nltk
import numpy as np

from tqdm import tqdm
from sortedcontainers import SortedList, SortedSet

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datamodel import Block, Data
from utils.utils import insert_to_dict
from utils.constants import LIST, SET
from blocks.utils import drop_single_entity_blocks, create_entity_index, print_blocks

class AbstractBlockCleaning:
    def __init__(self) -> None:
        pass

class BlockFiltering(AbstractBlockCleaning):
    '''
    Block Filtering
    ---
    Retains every entity in a subset of its smallest blocks

    Filtering consists of 3 steps:
    - Blocks sort in ascending cardinality
    - Creation of Entity Index: inversed block dictionary
    - Retain every entity in ratio % of its smallest blocks
    - Blocks reconstruction
    '''

    _method_name = "Block Filtering"
    _method_info = ": it retains every entity in a subset of its smallest blocks."

    def __init__(self, ratio: float = 0.8) -> None:
        super().__init__()
        self.ratio = ratio
        self.blocks: dict

    def __str__(self) -> str:
        print(self._method_name + self._method_info)
        print("Ratio: ", self.ratio)
        return super().__str__()

    def process(self, blocks: dict = None, data: Data = None) -> dict:
        '''
        Main function of Block Filtering
        ---
        Input: dict of keys -> Block
        Returns: dict of keys -> Block
        '''
        self.data = data
        pbar = tqdm(total=3, desc="Block Filtering")
        # print("dataset_limit: ", dataset_limit)
        sorted_blocks = self._sort_blocks_cardinality(blocks)
        pbar.update(1)
        entity_index = create_entity_index(sorted_blocks, self.data.is_dirty_er)
        pbar.update(1)

        filtered_blocks = {}
        for entity_id, block_keys in entity_index.items():
            # Create new blocks from the entity index
            # print(int(self.ratio*len(block_keys)))
            for key in block_keys[:int(self.ratio*len(block_keys))]:
                filtered_blocks.setdefault(key, Block())

                # Entities ids start to 0 ... n-1 for 1st dataset
                # and n ... m for 2nd dataset
                if entity_id < self.data.dataset_limit:
                    # print(key)
                    filtered_blocks[key].entities_D1.add(entity_id)
                else:
                    filtered_blocks[key].entities_D2.add(entity_id)
        pbar.update(1)
        # print_blocks(filtered_blocks, self._is_dirty_er)
        self.blocks = drop_single_entity_blocks(filtered_blocks, self.data.is_dirty_er)

        return self.blocks

    def _sort_blocks_cardinality(self, blocks: dict) -> dict:
        return dict(sorted(blocks.items(), key=lambda x: x[1].get_cardinality(self.data.is_dirty_er)))

