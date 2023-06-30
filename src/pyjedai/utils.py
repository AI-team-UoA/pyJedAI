from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from nltk import ngrams
from nltk.tokenize import word_tokenize
from pyjedai.datamodel import Block, Data
from typing import List, Tuple
import random
from queue import PriorityQueue
import math
import sys
from time import time
from networkx import Graph
from ordered_set import OrderedSet
from math import floor
# ----------------------- #
# Constants
# ----------------------- #
EMPTY = -1

# ----------------------- #
# Utility Methods
# ----------------------- #
def create_entity_index(blocks: dict, is_dirty_er: bool) -> dict:
    """Creates a dict of entity ids to block keys
        Example:
            e_id -> ['block_key_1', ..]
            ...  -> [ ... ]
    """
    entity_index = {}
    for key, block in blocks.items():
        for entity_id in block.entities_D1:
            entity_index.setdefault(entity_id, OrderedSet())
            entity_index[entity_id].add(key)

        if not is_dirty_er:
            for entity_id in block.entities_D2:
                entity_index.setdefault(entity_id, OrderedSet())
                entity_index[entity_id].add(key)

    return entity_index

def are_matching(entity_index, id1, id2) -> bool:
    '''
    id1 and id2 consist a matching pair if:
    - Blocks: intersection > 0 (comparison of sets)
    - Clusters: cluster-id-j == cluster-id-i (comparison of integers)
    '''

    if len(entity_index) < 1:
        raise ValueError("No entities found in the provided index")
    if isinstance(list(entity_index.values())[0], set): # Blocks case
        return len(entity_index[id1].intersection(entity_index[id2])) > 0
    return entity_index[id1] == entity_index[id2] # Clusters case

def get_blocks_cardinality(blocks: dict, is_dirty_er: bool) -> int:
    """Returns the cardinality of the blocks.

    Args:
        blocks (dict): Blocks.

    Returns:
        int: Cardinality.
    """
    return sum([block.get_cardinality(is_dirty_er) for block in blocks.values()])

def drop_big_blocks_by_size(blocks: dict, max_block_size: int, is_dirty_er: bool) -> dict:
    """Drops blocks if:
        - Contain only one entity
        - Have blocks with size greater than max_block_size

    Args:
        blocks (dict): Blocks.
        max_block_size (int): Max block size. If is greater that this, block will be rejected.
        is_dirty_er (bool): Type of ER.

    Returns:
        dict: New blocks.
    """
    return dict(filter(
        lambda e: not block_with_one_entity(e[1], is_dirty_er)
                    and e[1].get_size() <= max_block_size,
                blocks.items()
        )
    )

def drop_single_entity_blocks(blocks: dict, is_dirty_er: bool) -> dict:
    """Removes one-size blocks for DER and empty for CCER
    """
    return dict(filter(lambda e: not block_with_one_entity(e[1], is_dirty_er), blocks.items()))

def block_with_one_entity(block: Block, is_dirty_er: bool) -> bool:
    """Checks for one entity blocks.

    Args:
        block (Block): Block of entities.
        is_dirty_er (bool): Type of ER.

    Returns:
        bool: True if it contains only one entity. False otherwise.
    """
    return True if ((is_dirty_er and len(block.entities_D1) == 1) or \
        (not is_dirty_er and (len(block.entities_D1) == 0 or len(block.entities_D2) == 0))) \
            else False

def print_blocks(blocks: dict, is_dirty_er: bool) -> None:
    """Prints all the contents of the block index.

    Args:
        blocks (_type_):  Block of entities.
        is_dirty_er (bool): Type of ER.
    """
    print("Number of blocks: ", len(blocks))
    for key, block in blocks.items():
        block.verbose(key, is_dirty_er)

def print_candidate_pairs(blocks: dict) -> None:
    """Prints candidate pairs index in natural language.

    Args:
        blocks (dict): Candidate pairs structure.
    """
    print("Number of blocks: ", len(blocks))
    for entity_id, candidates in blocks.items():
        print("\nEntity id ", "\033[1;32m"+str(entity_id)+"\033[0m", " is candidate with: ")
        print("- Number of candidates: " + "[\033[1;34m" + \
            str(len(candidates)) + " entities\033[0m]")
        print(candidates)

def print_clusters(clusters: list) -> None:
    """Prints clusters contents.

    Args:
        clusters (list): clusters.
    """
    print("Number of clusters: ", len(clusters))
    for (cluster_id, entity_ids) in zip(range(0, len(clusters)), clusters):
        print("\nCluster ", "\033[1;32m" + \
              str(cluster_id)+"\033[0m", " contains: " + "[\033[1;34m" + \
            str(len(entity_ids)) + " entities\033[0m]")
        print(entity_ids)

def text_cleaning_method(col):
    """Lower clean.
    """
    return col.str.lower()

def chi_square(in_array: np.array) -> float:
    """Chi Square Method

    Args:
        in_array (np.array): Input array

    Returns:
        float: Statistic computation of Chi Square.
    """
    row_sum, column_sum, total = \
        np.sum(in_array, axis=1), np.sum(in_array, axis=0), np.sum(in_array)
    sum_sq = expected = 0.0
    for r in range(0, in_array.shape[0]):
        for c in range(0, in_array.shape[1]):
            expected = (row_sum[r]*column_sum[c])/total
            sum_sq += ((in_array[r][c]-expected)**2)/expected
    return sum_sq

def java_math_round(value):
    return int(value + 0.5)

def batch_pairs(iterable, batch_size: int = 1):
    """
    Generator function that breaks an iterable into batches of a set size.
    :param iterable: The iterable to be batched.
    :param batch_size: The size of each batch.
    """
    return (iterable[pos:pos + batch_size] for pos in range(0, len(iterable), batch_size))

def get_sorted_blocks_shuffled_entities(dirty_er: bool, blocks: dict) -> List[int]:
    """Sorts blocks in alphabetical order based on their token, shuffles the entities of each block, concatenates the result in a list

    Args:
        blocks (Dict[Block]): Dictionary of type token -> Block Instance

    Returns:
        List[Int]: List of shuffled entities of alphabetically, token sorted blocks
    """
    sorted_entities = []
    for _, block in sorted(blocks.items()):
        _shuffled_neighbors = list(block.entities_D1 | block.entities_D2 if not dirty_er else block.entities_D1)
        random.shuffle(_shuffled_neighbors)
        sorted_entities += _shuffled_neighbors

    return sorted_entities

class Tokenizer(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def tokenize(self, text: str) -> list:
        pass

class WordQgrammsTokenizer(Tokenizer):
    
    def __init__(self, q: int = 3) -> None:
        super().__init__()
        self.q = q
    
    def tokenize(self, text: str) -> list:
        return [' '.join(gram) for gram in list(ngrams(word_tokenize(text), self.q))]


class Tokenizer(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def tokenize(self, text: str) -> list:
        pass

class WordQgrammsTokenizer(Tokenizer):
    
    def __init__(self, q: int = 3) -> None:
        super().__init__()
        self.q = q
    
    def tokenize(self, text: str) -> list:
        return [' '.join(gram) for gram in list(ngrams(word_tokenize(text), self.q))]

class SubsetIndexer(ABC):
    """Stores the indices of retained entities of the initial datasets,
       calculates and stores the mapping of element indices from new to old dataset (id in subset -> id in original)
    """
    def __init__(self, blocks: dict, data: Data, subset : bool):
        self.d1_retained_ids: list[int] = None
        self.d2_retained_ids : list[int] = None
        self.subset : bool = subset
        self.store_retained_ids(blocks, data)

    def from_source_dataset(self, entity_id : int, data: Data) -> bool:
        return entity_id < data.dataset_limit

    def store_retained_ids(self, blocks: dict, data: Data) -> None:
        """Stores lists contains the ids of entities that we retained from both datasets
           in ascending order
        Args:
            blocks (dict): Mapping from entity id to a set of its neighbors ids
            data (Data): Dataset Module
        """

        if(not self.subset):
            self.d1_retained_ids = list(range(data.num_of_entities_1))

            if(not data.is_dirty_er):
                self.d2_retained_ids = list(range(data.num_of_entities_1, data.num_of_entities_1 + data.num_of_entities_2))
        else:
            _d1_retained_ids_set: set[int] = set()
            _d2_retained_ids_set: set[int] = set()

            if(data.is_dirty_er):
                _d1_retained_ids_set = set(blocks.keys())
                for neighbors in blocks.values():
                    _d1_retained_ids_set = _d1_retained_ids_set.union(neighbors)
                self.d1_retained_ids = sorted(list(_d1_retained_ids_set))
                self.d2_retained_ids = []
            else:

                for entity in blocks.keys():
                    if(self.from_source_dataset(entity, data)): _d1_retained_ids_set.add(entity) 
                    else: _d2_retained_ids_set.add(entity)

                    neighbors = blocks[entity]
                    for neighbor in neighbors:
                        if(self.from_source_dataset(neighbor, data)): _d1_retained_ids_set.add(entity)
                        else: _d2_retained_ids_set.add(entity)

                self.d1_retained_ids = sorted(list(_d1_retained_ids_set))
                self.d2_retained_ids = sorted(list(_d2_retained_ids_set))
                
class PositionIndex(ABC):
    """For each entity identifier stores a list of index it appears in, within the list of shuffled entities of sorted blocks

    Args:
        ABC (ABC): ABC Module 
    """
    
    def __init__(self, num_of_entities: int, sorted_entities: List[int]) -> None:
        self._num_of_entities = num_of_entities
        self._counters = self._num_of_entities * [0]
        self._entity_positions = [[] for _ in range(self._num_of_entities)]
        
        for entity in sorted_entities:
            self._counters[entity]+=1
            
        for i in range(self._num_of_entities):
            self._entity_positions[i] = [0] * self._counters[i]
            self._counters[i] = 0
            
        for index, entity in enumerate(sorted_entities):
            self._entity_positions[entity][self._counters[entity]] = index
            self._counters[entity] += 1
            
    def get_positions(self, entity: int):
        return self._entity_positions[entity]

class WhooshNeighborhood(ABC):
    """Stores information about the neighborhood of a given entity ID:
    - ID : The identifier of the entity as it is defined within the original dataframe
    - Total Weight : The total weight of entity's neighbors
    - Number of Neighbors : The total number of Neighbors
    - Neighbors : Entity's neighbors sorted in descending order of weight
    - Stage : Insert / Pop stage (entities stored in ascending / descending weight order)

    Args:
        ABC (ABC): ABC Module 
    """
    
    def __init__(self, id : int, budget : float) -> None:
        self._id : int = id
        self._budget : float = budget
        self._neighbors : PriorityQueue = PriorityQueue(self._budget) if not is_infinite(self._budget) else PriorityQueue()
        self._insert_stage : bool = True
        self._minimum_weight : float = sys.float_info.min
        self._neighbors_num : int = 0
        self._total_weight : float = 0.0
        self._average_weight : float = None
        
    def _insert(self, neighbor_id: int, weight : float) -> None:
        if(not self._insert_stage): self._change_state()
        
        if weight >= self._minimum_weight:
            self._neighbors.put((weight, neighbor_id))
        if self._neighbors.qsize() > self._budget:
            self._minimum_weight = self._neighbors.get()[0]
            
        self._update_neighbors_counter_by(1)
        self._update_total_weight_by(weight)
            
    def _pop(self) -> None:
        if(self._insert_stage): self._change_state()
        
        if(self._empty()):
            raise ValueError("No neighbors to pop!")
        
        _weight, _neighbor_id = self._neighbors.get()
        return -_weight, _neighbor_id
    
    def _empty(self) -> bool:
        return self._neighbors.empty()
    
    def _change_state(self) -> None:
        "Neighborhood can either be accepting or emitting neighbors" + \
        "Accepting Stage - Neighbors stored in ascending weight order" + \
        "Emitting Stage - Neighbors stored in descending weight order" 
        _neighbors_resorted : PriorityQueue = PriorityQueue(int(self._budget)) if not is_infinite(self._budget) else PriorityQueue()
        while(not self._neighbors.empty()):
            _weight, _neighbor_id = self._neighbors.get()
            _neighbors_resorted.put((-_weight, _neighbor_id))
            
        self._neighbors = _neighbors_resorted
        self._insert_stage = not self._insert_stage
        
    def _update_total_weight_by(self, weight) -> None:
        self._total_weight = self._total_weight + weight
        
    def _update_neighbors_counter_by(self, count) -> None:
        self._neighbors_num = self._neighbors_num + count
        
    def _get_neighbors_num(self) -> int:
        return self._neighbors_num
    
    def _get_total_weight(self) -> float:
        return self._total_weight
    
    def _get_average_weight(self) -> float:
        if(self._average_weight is None):
            self._average_weight = 0.0 if not self._get_neighbors_num() else (float(self._get_total_weight()) / float(self._get_neighbors_num()))
            return self._average_weight
        else:
            return self._average_weight
    
    def __eq__(self, other):
        if isinstance(other, WhooshNeighborhood):
            return self._get_average_weight() == other._get_average_weight()
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, WhooshNeighborhood):
            return self._get_average_weight() < other._get_average_weight()
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, WhooshNeighborhood):
            return self._get_average_weight() > other._get_average_weight()
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, WhooshNeighborhood):
            return self._get_average_weight() <= other._get_average_weight()
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, WhooshNeighborhood):
            return self._get_average_weight() >= other._get_average_weight()
        return NotImplemented 
    
class WhooshDataset(ABC):
    """Stores a dictionary [Entity -> Entity's Neighborhood Data (Whoosh Neighborhood)]
       Supplies auxiliarry functions for information retrieval from the sorted dataset

    Args:
        ABC (ABC): ABC Module
    """
     
    def __init__(self, entity_ids : List[int], budget : float) -> None:
        self._budget : float = budget
        self._total_entities : int = len(entity_ids)
        self._entity_budget : float = budget if is_infinite(self._budget) else max(1, 2 * self._budget / self._total_entities)
        self._neighborhoods : dict = {}
        for entity_id in entity_ids:  
            self._neighborhoods[entity_id] = WhooshNeighborhood(id=entity_id, budget=self._entity_budget)
        # used in defining proper emission strategy
        self._sorted_entities : List[int] = None
        self._current_neighborhood_index : int = 0
        self._current_entity : int = None
        self._current_neighborhood : WhooshNeighborhood = None
            
    def _insert_entity_neighbor(self, entity : int, neighbor : int, weight : float) -> None:
        self._neighborhoods[entity]._insert(neighbor, weight)
        
    def _pop_entity_neighbor(self, entity : int) -> Tuple[float, int]:
        return self._neighborhoods[entity]._pop()
    
    def _get_entity_neighborhood(self, entity : int) -> WhooshNeighborhood:
        return self._neighborhoods[entity]
    
    def _entity_has_neighbors(self, entity : int) -> bool:
        return not self._neighborhoods[entity]._empty()
    
    def _sort_neighborhoods_by_avg_weight(self) -> None:
        """Store a list of entity ids sorted in descending order of the average weight of their corresponding neighborhood"""
        self._sorted_entities : List = sorted(self._neighborhoods, key=lambda entity: self._neighborhoods[entity]._get_average_weight(), reverse=True)
    
    def _get_current_neighborhood(self) -> WhooshNeighborhood:
        return self._neighborhoods[self._current_entity]
        
    def _enter_next_neighborhood(self) -> None:
        """Sets the next in descending average weight order neighborhood
        """
        _curr_nei_idx : int = self._current_neighborhood_index
        self._current_neighborhood_index = _curr_nei_idx + 1 if _curr_nei_idx + 1 < self._total_entities else 0
        self._current_entity = self._sorted_entities[self._current_neighborhood_index]
        self._current_neighborhood = self._neighborhoods[self._current_entity]
        
    def _successful_emission(self, pair : Tuple[int, int]) -> bool:
        
        _entity, _neighbor = pair
        _entity_id = self._data._ids_mapping_1[_entity]
        _neighbor_id = self._data._ids_mapping_1[_neighbor] if self._data.is_dirty_er else self._data._ids_mapping_2[_neighbor]
                
        if(self._emitted_comparisons < self._budget):
            self._emitted_pairs.append((_entity_id, _neighbor_id))
            self._emitted_comparisons += 1
            return True
        else:
            return False
        
    def _emit_pairs(self, method : str, data : Data) -> List[Tuple[int, int]]:
        """Emits candidate pairs according to specified method

        Args:
            method (str): Emission Method
            data (Data): Dataset Module

        Returns:
            List[Tuple[int, int]]: List of candidate pairs
        """
        
        self._method : str = method
        self._data : Data = data
        
        self._emitted_pairs = []
        self._emitted_comparisons = 0    
        
        if(self._method == 'HB'):
            for sorted_entity in self._sorted_entities:
                if(self._entity_has_neighbors(sorted_entity)):
                    _, neighbor = self._pop_entity_neighbor(sorted_entity)
                    if(not self._successful_emission(pair=(sorted_entity, neighbor))):
                        return self._emitted_pairs
                   
        if(self._method == 'HB' or self._method == 'DFS'):
            _checked_entity = np.zeros(self._total_entities, dtype=bool)
            _sorted_entity_to_index = dict(zip(self._sorted_entities, range(0, self._total_entities)))
            
            for index, sorted_entity in enumerate(self._sorted_entities):
                _checked_entity[index] = True
                while(self._entity_has_neighbors(sorted_entity)):
                    _, neighbor = self._pop_entity_neighbor(sorted_entity)
                    if(neighbor not in _sorted_entity_to_index or _checked_entity[_sorted_entity_to_index[neighbor]]):
                        if(not self._successful_emission(pair=(sorted_entity, neighbor))):
                            return self._emitted_pairs
        else:
            _emissions_left = True
            _checked_entities = set()
            while(_emissions_left):
                _emissions_left = False
                for sorted_entity in self._sorted_entities:
                    if(self._entity_has_neighbors(sorted_entity)):
                        _, neighbor = self._pop_entity_neighbor(sorted_entity)
                        if(canonical_swap(sorted_entity, neighbor) not in _checked_entities):
                            if(not self._successful_emission(pair=(sorted_entity, neighbor))):
                                return self._emitted_pairs
                            _checked_entities.add(canonical_swap(sorted_entity, neighbor))
                            _emissions_left = True
        return self._emitted_pairs
    
class PredictionData(ABC):
    """Auxiliarry module used to store basic information about the to-emit, predicted pairs
       It is used to retrieve that data efficiently during the evaluation phase, and subsequent storage of emission data (e.x. total emissions)

    Args:
        ABC (ABC): ABC Module
    """
    def __init__(self, name : str, predictions, tps_checked = dict) -> None:
        self.set_name(name)
        self.set_tps_checked(tps_checked)
        self.set_predictions(self._format_predictions(predictions))
        # Pairs have not been emitted yet - Data Module has not been populated with performance data
        self.set_total_emissions(None)
        self.set_normalized_auc(None)
        self.set_cumulative_recall(None)
    
    def _format_predictions(self, predictions) -> List[Tuple[int, int]]:
        """Transforms given predictions into a list of duplets (candidate pairs)
           Currently Graph and Default input is supported

        Args:
            predictions (Graph / List[Tuple[int, int]]): Progressive Matcher predictions

        Returns:
            List[Tuple[int, int]]: Formatted Predictions
        """
        return [edge[:2] for edge in predictions.edges] if isinstance(predictions, Graph) else predictions
        
    def get_name(self) -> str:
        return self._name
    
    def get_predictions(self) -> List[Tuple[int, int]]:
        return self._predictions
    
    def get_tps_checked(self) -> dict:
        return self._tps_checked
    
    def get_total_emissions(self) -> int:
        if(self._total_emissions is None): raise ValueError("Pairs not emitted yet - Total Emissions are undefined")
        return self._total_emissions
    
    def get_normalized_auc(self) -> float:
        if(self._normalized_auc is None): raise ValueError("Pairs not emitted yet - Normalized AUC is undefined")
        return self._normalized_auc
    
    def get_cumulative_recall(self) -> float:
        if(self._cumulative_recall is None): raise ValueError("Pairs not emitted yet - Cumulative Recall is undefined")
        return self._cumulative_recall
    
    def set_name(self, name : str):
        self._name : str = name
    
    def set_predictions(self, predictions : List[Tuple[int, int]]) -> None:
        self._predictions : List[Tuple[int, int]] = predictions
    
    def set_tps_checked(self, tps_checked : dict) -> None:
        self._tps_checked : dict = tps_checked
    
    def set_total_emissions(self, total_emissions : int) -> None:
        self._total_emissions : int = total_emissions
        
    def set_normalized_auc(self, normalized_auc : float) -> None:
        self._normalized_auc : float = normalized_auc
        
    def set_cumulative_recall(self, cumulative_recall : float) -> None:
        self._cumulative_recall : float = cumulative_recall        

            
def canonical_swap(id1: int, id2: int) -> Tuple[int, int]:
    """Returns the identifiers in canonical order

    Args:
        id1 (int): ID1
        id2 (int): ID2

    Returns:
        Tuple[int, int]: IDs tuple in canonical order (ID1 < ID2)
    """
    if id2 > id1:
        return id1, id2
    else:
        return id2, id1

def sorted_enumerate(seq, reverse=True):
    return [i for (v, i) in sorted(((v, i) for (i, v) in enumerate(seq)), reverse=reverse)]


def is_infinite(value : float):
    return math.isinf(value) and value > 0




            
            
        
        
        
    

        