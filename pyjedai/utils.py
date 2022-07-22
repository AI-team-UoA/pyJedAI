'''
Utility functions
'''
import numpy as np

# Constants
EMPTY = -1

def create_entity_index(blocks: dict, is_dirty_er: bool):
    '''
     Creates a dict of entity ids -> block ids
    '''
    entity_index = {}
    for key, block in blocks.items():
        for entity_id in block.entities_D1:
            entity_index.setdefault(entity_id, [])
            entity_index[entity_id].append(key)

        if not is_dirty_er:
            for entity_id in block.entities_D2:
                entity_index.setdefault(entity_id, [])
                entity_index[entity_id].append(key)

    return entity_index

def drop_big_blocks_by_size(blocks: dict, max_block_size: int) -> dict:
    all_keys = list(blocks.keys())
    for key in all_keys:
        if blocks[key].get_size() > max_block_size:
            blocks.pop(key)
    return blocks

def drop_single_entity_blocks(blocks: dict, is_dirty_er: bool) -> dict:
    '''
     Removes one-size blocks for DER and empty for CCER
    '''
    all_keys = list(blocks.keys())
    
    if is_dirty_er:
        for key in all_keys:
            if len(blocks[key].entities_D1) == 1:
                blocks.pop(key)
    else:
        for key in all_keys:
            if len(blocks[key].entities_D1) == 0 or len(blocks[key].entities_D2) == 0:
                blocks.pop(key)

    return blocks 
    
def print_blocks(blocks, is_dirty_er):
    print("Number of blocks: ", len(blocks))
    for key, block in blocks.items():
        block.verbose(key, is_dirty_er)

def print_candidate_pairs(blocks):
    print("Number of blocks: ", len(blocks))
    for entity_id, candidates in blocks.items():
        print("\nEntity id ", "\033[1;32m"+str(entity_id)+"\033[0m", " is candidate with: ")
        print("- Number of candidates: " + "[\033[1;34m" + \
            str(len(candidates)) + " entities\033[0m]")
        print(candidates)

def print_clusters(clusters: list) -> None:
    print("Number of clusters: ", len(clusters))
    for (cluster_id, entity_ids) in zip(range(0, len(clusters)), clusters):
        print("\nCluster ", "\033[1;32m"+str(cluster_id)+"\033[0m", " contains: " + "[\033[1;34m" + \
            str(len(entity_ids)) + " entities\033[0m]")
        print(entity_ids)
        
def text_cleaning_method(col):
    return col.str.lower()

def chi_square(in_array: np.array) -> float:
    row_sum = np.sum(in_array, axis=1)
    column_sum = np.sum(in_array, axis=0)
    total = np.sum(in_array)    
    sum_sq = 0.0; expected = 0.0;
    for r in range(0, in_array.shape[0]):
        for c in range(0, in_array.shape[1]):
            expected = (row_sum[r]*column_sum[c])/total
            sum_sq += ((in_array[r][c]-expected)**2)/expected 
    return sum_sq
        