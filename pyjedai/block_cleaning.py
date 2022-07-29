import numpy as np
import time

from tqdm.notebook import tqdm

# pyJedAI
from .datamodel import Data, Block
from .utils import create_entity_index, drop_big_blocks_by_size, drop_single_entity_blocks

class BlockFiltering:
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
        self.ratio = ratio
        self.blocks: dict
        
    def __str__(self) -> str:
        print(self._method_name + self._method_info)
        print("Ratio: ", self.ratio)
        return super().__str__()

    def process(self, 
                blocks: dict = None, 
                data: Data = None, 
                tqdm_disable: bool = False
    ) -> dict:
        '''
        Main function of Block Filtering
        ---
        Input: dict of keys -> Block
        Returns: dict of keys -> Block
        '''
        start_time = time.time()
        self.tqdm_disable = tqdm_disable
        self.data = data
        self._progress_bar = tqdm(
            total=3, 
            desc=self._method_name, 
            dynamic_ncols =True, 
            disable=self.tqdm_disable
        )
        
        sorted_blocks = sort_blocks_cardinality(blocks, self.data.is_dirty_er)
        self._progress_bar.update(1)
        
        entity_index = create_entity_index(sorted_blocks, self.data.is_dirty_er)
        self._progress_bar.update(1)
            
        filtered_blocks = {}
        for entity_id, block_keys in entity_index.items():
            # Create new blocks from the entity index
            for key in block_keys[:int(round(self.ratio*len(block_keys)))]:
                filtered_blocks.setdefault(key, Block())

                # Entities ids start to 0 ... n-1 for 1st dataset
                # and n ... m for 2nd dataset
                if entity_id < self.data.dataset_limit:
                    filtered_blocks[key].entities_D1.add(entity_id)
                else:
                    filtered_blocks[key].entities_D2.add(entity_id)
        self._progress_bar.update(1)
        
        self.blocks = drop_single_entity_blocks(filtered_blocks, self.data.is_dirty_er)
        
        self._progress_bar.close() 
        self.execution_time = time.time() - start_time
        
        return self.blocks

class BlockPurging:
    '''
    BlockPurging
    ---
    Discards the blocks exceeding a certain number of comparisons.
    '''
    _method_name = "Block Purging"
    _method_info = ": it discards the blocks exceeding a certain number of comparisons."

    def __init__(self, smoothing_factor: float = 1.025) -> any:
        self.smoothing_factor: float = smoothing_factor
        self.max_comparisons_per_block: float
    
    def process(
        self,
        blocks: dict,
        data: Data,
        tqdm_disable: bool = False
    ) -> dict:
        '''
        TODO: add description
        '''
        self.tqdm_disable = tqdm_disable
        start_time = time.time()
        self._progress_bar = tqdm(total=2*len(blocks), desc=self._method_name, disable=self.tqdm_disable)
        self.data = data
        if not blocks:
            print("Empty dict of blocks was given as input!") #TODO error
            return blocks
        
        new_blocks = blocks.copy()
        self._set_threshold(new_blocks)
        
        all_keys = list(new_blocks.keys())
        for key in all_keys:
            if new_blocks[key].get_cardinality(self.data.is_dirty_er) > self.max_comparisons_per_block:
                new_blocks.pop(key)
            self._progress_bar.update(1)
        
        self.execution_time = time.time() - start_time
        self._progress_bar.close()
        return new_blocks
    
    
    def _set_threshold(self, blocks: dict) -> None:
        sorted_blocks = sort_blocks_cardinality(blocks, self.data.is_dirty_er)
        
        distinct_comparisons_level = set(b.get_cardinality(self.data.is_dirty_er) for k, b in sorted_blocks.items())
        
        block_assignments = np.empty([len(distinct_comparisons_level)])
        comparisons_level = np.empty([len(distinct_comparisons_level)])
        total_comparisons_per_level = np.empty([len(distinct_comparisons_level)])
        
        index = -1
        for block_key, block in sorted_blocks.items():
            if index == -1:
                index += 1
                comparisons_level[index] = block.get_cardinality(self.data.is_dirty_er)
                block_assignments[index] = 0
                total_comparisons_per_level[index] = 0
            elif block.get_cardinality(self.data.is_dirty_er) != comparisons_level[index]:
                index += 1
                comparisons_level[index] = block.get_cardinality(self.data.is_dirty_er)
                block_assignments[index] = block_assignments[index-1]
                total_comparisons_per_level[index] = total_comparisons_per_level[index-1]
            
            block_assignments[index] += block.get_size()
            total_comparisons_per_level[index] += block.get_cardinality(self.data.is_dirty_er)
            self._progress_bar.update(1)
            
        
        current_bc = 0; current_cc = 0; current_size = 0
        previous_bc = 0; previous_cc = 0; previous_size = 0
        for i in range(len(block_assignments)-1, 0, -1):
            previous_size = current_size
            previous_bc = current_bc
            previous_cc = current_cc
            current_size = comparisons_level[i]
            current_bc = block_assignments[i]
            current_cc = total_comparisons_per_level[i]
            if current_bc * previous_cc < self.smoothing_factor * current_cc * previous_bc:
                break
                
        self.max_comparisons_per_block = previous_size
                        
    def _satisfies_threshold(self, block: Block) -> bool:
        return block.get_cardinality(self.data.is_dirty_er) <= self.max_comparisons_per_block

def sort_blocks_cardinality(blocks: dict, is_dirty_er: bool) -> dict:
    return dict(sorted(blocks.items(), key=lambda x: x[1].get_cardinality(is_dirty_er)))
