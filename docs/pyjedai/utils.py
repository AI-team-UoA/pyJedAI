import numpy as np
import re
import random
import math
import uuid
import os
import json
import pandas as pd
import inspect

from abc import ABC, abstractmethod
from typing import List, Tuple

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk import ngrams
from nltk.tokenize import word_tokenize

from queue import PriorityQueue
from networkx import Graph
from ordered_set import OrderedSet

from pyjedai.datamodel import Block, Data

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

def cosine(x, y):
    """Cosine similarity between two vectors
    """
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]

def get_ngrams(text, n ):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]

def get_qgram_from_tokenizer_name(tokenizer: str) -> int:
    """Returns the q-gram value from the tokenizer name.

    Args:
        tokenizer (str): Tokenizer name.

    Returns:
        int: q-gram value.
    """
    return [int(s) for s in re.findall('\d+', tokenizer)][0]

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

class WordQgramTokenizer(Tokenizer):
    
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

class EntityScheduler(ABC):
    """Stores information about the neighborhood of a given entity ID:
    - ID : The identifier of the entity as it is defined within the original dataframe
    - Total Weight : The total weight of entity's neighbors
    - Number of Neighbors : The total number of Neighbors
    - Neighbors : Entity's neighbors sorted in descending order of weight
    - Stage : Insert / Pop stage (entities stored in ascending / descending weight order)

    Args:
        ABC (ABC): ABC Module 
    """
    
    def __init__(self, id : int) -> None:
        self._id : int = id
        self._neighbors : PriorityQueue = PriorityQueue()
        self._neighbors_num : int = 0
        self._total_weight : float = 0.0
        self._average_weight : float = None
        
    def _insert(self, neighbor_id: int, weight : float) -> None:
        self._neighbors.put((-weight, neighbor_id))
        self._update_neighbors_counter_by(1)
        self._update_total_weight_by(weight)
            
    def _pop(self) -> Tuple[float, int]:
        if(self._empty()):
            raise ValueError("No neighbors to pop!")
        
        _weight, _neighbor_id = self._neighbors.get()
        self._update_neighbors_counter_by(-1)
        self._update_total_weight_by(_weight)
        
        return -_weight, _neighbor_id
    
    def _empty(self) -> bool:
        return self._neighbors.empty()
        
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
        if isinstance(other, EntityScheduler):
            return self._get_average_weight() == other._get_average_weight()
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, EntityScheduler):
            return self._get_average_weight() < other._get_average_weight()
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, EntityScheduler):
            return self._get_average_weight() > other._get_average_weight()
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, EntityScheduler):
            return self._get_average_weight() <= other._get_average_weight()
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, EntityScheduler):
            return self._get_average_weight() >= other._get_average_weight()
        return NotImplemented 
    
class DatasetScheduler(ABC):
    """Stores a dictionary [Entity -> Entity's Neighborhood Data (Whoosh Neighborhood)]
       Supplies auxiliarry functions for information retrieval from the sorted dataset

    Args:
        ABC (ABC): ABC Module
    """
     
    def __init__(self, budget : float = float('inf'), entity_ids : List[int] = [], global_top : bool = False) -> None:
        self._budget : float = budget
        self._total_entities : int = len(entity_ids)
        self._neighborhoods : dict = {}
        # global emission case
        self._global_top : bool = global_top
        self._all_candidates = PriorityQueue() if self._global_top else None
        for entity_id in entity_ids:  
            self._neighborhoods[entity_id] = EntityScheduler(id=entity_id)
        # used in defining proper emission strategy
        self._sorted_entities : List[int] = None
        self._current_neighborhood_index : int = 0
        self._current_entity : int = None
        self._current_neighborhood : EntityScheduler = None
            
    def _insert_entity_neighbor(self, entity : int, neighbor : int, weight : float) -> None:
        if(not self._global_top):
            if(entity not in self._neighborhoods):
                _new_neighborhood : EntityScheduler = EntityScheduler(entity)
                _new_neighborhood._insert(neighbor, weight)
                self._neighborhoods[entity] = _new_neighborhood
            else:
                self._neighborhoods[entity]._insert(neighbor, weight)
        else:
            self._all_candidates.put((-weight, entity, neighbor))
        
    def _pop_entity_neighbor(self, entity : int) -> Tuple[float, int]:
        return self._neighborhoods[entity]._pop()
    
    def _get_entity_neighborhood(self, entity : int) -> EntityScheduler:
        return self._neighborhoods[entity]
    
    def _entity_has_neighbors(self, entity : int) -> bool:
        return not self._neighborhoods[entity]._empty()
    
    def _sort_neighborhoods_by_avg_weight(self) -> None:
        """Store a list of entity ids sorted in descending order of the average weight of their corresponding neighborhood"""
        self._sorted_entities : List = sorted(self._neighborhoods, key=lambda entity: self._neighborhoods[entity]._get_average_weight(), reverse=True)
    
    def _get_current_neighborhood(self) -> EntityScheduler:
        return self._neighborhoods[self._current_entity]
        
    def _enter_next_neighborhood(self) -> None:
        """Sets the next in descending average weight order neighborhood
        """
        _curr_nei_idx : int = self._current_neighborhood_index
        self._current_neighborhood_index = _curr_nei_idx + 1 if _curr_nei_idx + 1 < self._total_entities else 0
        self._current_entity = self._sorted_entities[self._current_neighborhood_index]
        self._current_neighborhood = self._neighborhoods[self._current_entity]
        
    def _successful_emission(self, pair : Tuple[float, int, int]) -> bool:
        _score, _entity, _neighbor = pair
                
        if(self._emitted_comparisons < self._budget):
            self._emitted_pairs.append((_score, _entity, _neighbor))
            self._emitted_comparisons += 1
            return True
        else:
            return False
        
    def _print_info(self):
        _n_ids : int
        if(self._sorted_entities is None):
            print("Neighborhood Status - Not Sorted by average weight")
            _n_ids = self._neighborhoods.keys()
        else:
            print("Neighborhood Status - Sorted by average weight")
            _n_ids = self._sorted_entities
        for _n_id in _n_ids:
            _current_neighborhood = self._neighborhoods[_n_id]
            print("#############################")
            print(f"Neighborhood[{_n_id}]")
            print(f"Total Neighbords[{_current_neighborhood._get_neighbors_num()}]")
            print(f"Average Weight[{_current_neighborhood._get_average_weight()}]")

    def _checked_pair(self, entity : int, candidate : int) -> bool:
        """Checks if the given pair has been checked previously in the scheduling process.
           In the case the given pair has been constructed in the reverse indexing context,
           proper translation to inorder indexing identification is done for correct checking.
           Finally, if the pair has not been checked in the past, it is added to the checked pool.
        Args:
            entity (int): Entity ID
            candidate (int): Candidate ID

        Returns:
            bool: Given pair has already been checked in the scheduling process
        """
        _d1_inorder_entity, _d2_inorder_entity = self._get_inorder_representation(entity, candidate)
        
        if((_d1_inorder_entity, _d2_inorder_entity) not in self._checked_entities):
            self._checked_entities.add((_d1_inorder_entity, _d2_inorder_entity))
            return False
        else:
            return True
        
    def _get_inorder_representation(self, entity : int, candidate : int) -> Tuple[int, int]:
        """Takes as input the ID of the entity of the first and second dataset in its schedule indexing context (in that order!). 
           Returns the ids of given entities in the inorder context,
           in the following order (id of the entity of the first dataset in the inorder context, -//- second -//-)
        Args:
            entity (int): Entity ID
            candidate (int): Candidate ID

        Returns:
            Tuple[int, int]: (id of entity of first dataframe, id of entity of second dataframe) in inorder context
        """
        if(entity < self._data.num_of_entities): return entity, candidate

        # reverse context case
        # - number of entities (to transfer the IDs from Scheduler -> Workflow ID representation)
        # + / - dataset limit in order to express (D1 in reverse context == D2 in inorder context, and the reverse)
        entity = entity - self._data.num_of_entities + self._data.num_of_entities_2
        candidate = candidate - self._data.num_of_entities - self._data.num_of_entities_1
        
        return candidate, entity 
            

    def _emit_pairs(self, method : str, data : Data) -> List[Tuple[float, int, int]]:
        """Emits candidate pairs according to specified method

        Args:
            method (str): Emission Method
            data (Data): Dataset Module of the 

        Returns:
            List[Tuple[int, int]]: List of candidate pairs
        """
        
        self._method : str = method
        self._emitted_pairs = []
        self._emitted_comparisons = 0    
        self._checked_entities = set()
        self._data : Data = data

        if(self._method == 'TOP'):
            while(not self._all_candidates.empty()):
                score, sorted_entity, neighbor = self._all_candidates.get()
                if(not self._checked_pair(sorted_entity, neighbor)):
                    if(not self._successful_emission(pair=(-score, sorted_entity, neighbor))):
                        return self._emitted_pairs
                
            return self._emitted_pairs
            
        
        if(self._method == 'HB'):
            for sorted_entity in self._sorted_entities:
                if(self._entity_has_neighbors(sorted_entity)):
                    score, neighbor = self._pop_entity_neighbor(sorted_entity)
                    if(not self._checked_pair(sorted_entity, neighbor)):
                        if(not self._successful_emission(pair=(score, sorted_entity, neighbor))):
                            return self._emitted_pairs
                   
        if(self._method == 'HB' or self._method == 'DFS'):            
            for sorted_entity in self._sorted_entities:
                while(self._entity_has_neighbors(sorted_entity)):
                    score, neighbor = self._pop_entity_neighbor(sorted_entity)
                    if(not self._checked_pair(sorted_entity, neighbor)):
                        if(not self._successful_emission(pair=(score, sorted_entity, neighbor))):
                            return self._emitted_pairs
        else:
            _emissions_left = True
            while(_emissions_left):
                _emissions_left = False
                for sorted_entity in self._sorted_entities:
                    if(self._entity_has_neighbors(sorted_entity)):
                        score, neighbor = self._pop_entity_neighbor(sorted_entity)
                        if(not self._checked_pair(sorted_entity, neighbor)):
                            if(not self._successful_emission(pair=(score, sorted_entity, neighbor))):
                                return self._emitted_pairs
                        _emissions_left = True
        return self._emitted_pairs
    
class PredictionData(ABC):
    """Auxiliarry module used to store basic information about the to-emit, predicted pairs
       It is used to retrieve that data efficiently during the evaluation phase, and subsequent storage of emission data (e.x. total emissions)

    Args:
        ABC (ABC): ABC Module
    """
    def __init__(self, matcher, matcher_info : dict) -> None:
        self.set_matcher_info(matcher_info)
        self.set_duplicate_emitted(matcher.duplicate_emitted)
        self.set_candidate_pairs(self._format_predictions(matcher.pairs))
    
    def _format_predictions(self, predictions) -> List[Tuple[int, int]]:
        """Transforms given predictions into a list of duplets (candidate pairs)
           Currently Graph and Default input is supported

        Args:
            predictions (Graph / List[Tuple[int, int]]): Progressive Matcher predictions

        Returns:
            List[Tuple[int, int]]: Formatted Predictions
        """
        return [edge[:3] for edge in predictions.edges] if isinstance(predictions, Graph) else predictions
        
    def get_name(self) -> str:
        _matcher_info : dict = self.get_matcher_info()
        if('name' not in _matcher_info): raise ValueError("Matcher doesn't have a name - Make sure its execution data has been calculated")
        return _matcher_info['name']
    
    def get_candidate_pairs(self) -> List[Tuple[float, int, int]]:
        if(self._candidate_pairs is None): raise ValueError("Pairs not scheduled yet - Cannot retrieve candidate pairs")
        return self._candidate_pairs
    
    def get_duplicate_emitted(self) -> dict:
        if(self._duplicate_emitted is None): raise ValueError("No information about the status of true positives' emission")
        return self._duplicate_emitted
    
    def get_total_emissions(self) -> int:
        _matcher_info : dict = self.get_matcher_info()
        if('total_emissions' not in _matcher_info): raise ValueError("Pairs not emitted yet - Total Emissions are undefined")
        return _matcher_info['total_emissions']
    
    def get_normalized_auc(self) -> float:
        _matcher_info : dict = self.get_matcher_info()
        if('auc' not in _matcher_info): raise ValueError("Pairs not emitted yet - Normalized AUC is undefined")
        return _matcher_info['auc']
    
    def get_cumulative_recall(self) -> float:
        _matcher_info : dict = self.get_matcher_info()
        if('recall' not in _matcher_info): raise ValueError("Pairs not emitted yet - Cumulative Recall is undefined")
        return _matcher_info['recall']
    
    def get_matcher_info(self) -> dict:
        if(self._matcher_info is None): raise ValueError("Pairs not emitted yet - Matcher Info is undefined")
        return self._matcher_info
    
    def set_matcher_info(self, matcher_info : dict) -> None:
        self._matcher_info : dict = matcher_info
    
    def set_name(self, name : str):
        _matcher_info : dict = self.get_matcher_info()
        _matcher_info['name'] = name  
    
    def set_candidate_pairs(self, candidate_pairs : List[Tuple[float, int, int]]) -> None:
        self._candidate_pairs : List[Tuple[float, int, int]] = candidate_pairs
    
    def set_duplicate_emitted(self, duplicate_emitted : dict) -> None:
        self._duplicate_emitted : dict = duplicate_emitted
    
    def set_total_emissions(self, total_emissions : int) -> None:
        _matcher_info : dict = self.get_matcher_info()
        _matcher_info['total_emissions'] = total_emissions
        
    def set_normalized_auc(self, normalized_auc : float) -> None:
        _matcher_info : dict = self.get_matcher_info()
        _matcher_info['auc'] = normalized_auc
        
    def set_cumulative_recall(self, cumulative_recall : float) -> None:
        _matcher_info : dict = self.get_matcher_info()
        _matcher_info['recall'] = cumulative_recall        
       
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

def reverse_data_indexing(data : Data) -> Data:
    """Returns a new data model based upon the given data model with reversed indexing of the datasets
    Args:
        data (Data): input dat a model

    Returns:
        Data : New Data Module with reversed indexing
    """
    return Data(dataset_1 = data.dataset_2,
                id_column_name_1 = data.id_column_name_2,
                attributes_1 = data.attributes_2,
                dataset_name_1 = data.dataset_name_2,
                dataset_2 = data.dataset_1,
                attributes_2 = data.attributes_1,
                id_column_name_2 = data.id_column_name_1,
                dataset_name_2 = data.dataset_name_1,
                ground_truth = data.ground_truth)

def get_class_function_arguments(class_reference, function_name : str) -> List[str]:
    """Returns a list of argument names for requested function of the given class
    Args:
        class_reference: Reference to a class
        function_name (str): Name of the requested function

    Returns:
        List[str] : List of requested function's arguments' names
    """
    if not inspect.isclass(class_reference):
        raise ValueError(f"{class_reference.__name__} class reference is not valid.")

    if not hasattr(class_reference, function_name):
        raise ValueError(f"The class {class_reference.__name__} does not have a function named {function_name}.")

    function_obj = getattr(class_reference, function_name)
    if not inspect.isfunction(function_obj):
        raise ValueError(f"The provided name {function_name} does not correspond to a function in class '{class_reference.__name__}'.")

    function_signature = inspect.signature(function_obj)
    argument_names = list(function_signature.parameters.keys())[1:]

    return argument_names

def new_dictionary_from_keys(dictionary : dict, keys : list) -> dict:
    """Returns a subset of the given dictionary including only the given keys.
       Unrecognized keys are not included.
    Args:
        dictionary (dict): Target dictionary
        keys (list): Keys to keep

    Returns:
        dict : Subset of the given dictionary including only the requested keys
    """
    new_dictionary : dict = {key: dictionary[key] for key in keys if key in dictionary}
    return new_dictionary


def has_duplicate_pairs(pairs : List[Tuple[float, int, int]]):
    seen_pairs = set()
    for pair in pairs:
        entity : int = pair[1]
        candidate : int = pair[2]
        if (entity, candidate) in seen_pairs:
            return True
        seen_pairs.add((entity, candidate))
    return False

def reverse_blocks_entity_indexing(blocks : dict, data : Data) -> dict:
    """Returns a new instance of blocks containing the entity IDs of the given blocks translated into the reverse indexing system
    Args:
        blocks (dict): blocks as defined in the previous indexing
        data (Data): Previous data module used to define the reversed ids based on previous dataset limit and dataset sizes

    Returns:
        dict : New block instance with identifiers defined in the context of the reverse indexing
    """
    if(blocks is None): return None
    all_blocks = list(blocks.values())
    if 'Block' in str(type(all_blocks[0])):
        return reverse_raw_blocks_entity_indexing(blocks, data)
    elif isinstance(all_blocks[0], set):
        return reverse_prunned_blocks_entity_indexing(blocks, data)
 
def reverse_prunned_blocks_entity_indexing(blocks : dict, data : Data) -> dict:
    _reversed_blocks : dict = dict()
    _reversed_block : set
     
    for entity in blocks:
        _updated_entity : int = get_reverse_indexing_id(entity, data)
        _reversed_block = set()
        block : set = blocks[entity]
        for candidate in block:
            _reversed_block.add(get_reverse_indexing_id(candidate, data))
        _reversed_blocks[_updated_entity] = _reversed_block
        
    return _reversed_blocks
        
def reverse_raw_blocks_entity_indexing(blocks : dict, data : Data) -> dict:
    _reversed_blocks : dict = dict()
    _reversed_block : Block 
    
    for token in blocks:
        _current_block : Block = blocks[token]
        _updated_D1_entities = OrderedSet()
        _updated_D2_entities = OrderedSet()
        
        for d1_entity in _current_block.entities_D1:
            _updated_D2_entities.add(get_reverse_indexing_id(d1_entity, data))
            
        for d2_entity in _current_block.entities_D2:
            _updated_D1_entities.add(get_reverse_indexing_id(d2_entity, data))
         
        _reversed_block = Block()   
        _reversed_block.entities_D1 = _updated_D1_entities
        _reversed_block.entities_D2 = _updated_D2_entities
        
        _reversed_blocks[token] = _reversed_block
    
    return _reversed_blocks
        
    
def get_reverse_indexing_id(id : int, data : Data) -> int:
    return (id + data.num_of_entities_2) if (id < data.num_of_entities_1) else (id - data.num_of_entities_1)


# Progressive Workflow Grid Search Utility Functions

def values_given(configuration: dict, parameter: str) -> bool:
    """Values for requested parameters have been supplied by the user in the configuration file

    Args:
        configuration (dict): Configuration File
        parameter (str): Requested parameter name

    Returns:
        bool: Values for requested parameter supplied
    """
    return (parameter in configuration) and (isinstance(configuration[parameter], list)) and (len(configuration[parameter]) > 0)

def get_multiples(num : int, n : int) -> list:
    """Returns a list of multiples of the requested number up to n * number

    Args:
        num (int): Number
        n (int): Multiplier

    Returns:
        list: Multiplies of num up to n * num 
    """
    multiples = []
    for i in range(1, n+1):
        multiples.append(num * i)
    return multiples

def necessary_dfs_supplied(configuration : dict) -> bool:
    """Configuration file contains values for source, target and ground truth dataframes

    Args:
        configuration (dict): Configuration file

    Raises:
        ValueError: Zero values supplied for one or more paths

    Returns:
        bool: _description_
    """
    for path in ['source_dataset_path', 'target_dataset_path', 'ground_truth_path']:
        if(not values_given(configuration, path)):
            raise ValueError(f"{path}: No values given")
        
    return len(configuration['source_dataset_path']) == len(configuration['target_dataset_path']) == len(configuration['ground_truth_path'])

def generate_unique_identifier() -> str:
    """Returns unique identifier which is used to cross reference workflows stored in json file and their performance graphs

    Returns:
        str: Unique identifier
    """
    return str(uuid.uuid4())  


def to_path(path : str):
    return os.path.expanduser(path)
            
def clear_json_file(path : str):
    if os.path.exists(path):
        if os.path.getsize(path) > 0:
            open(path, 'w').close()
            
            
def purge_id_column(columns : list):
    non_id_columns : list = []
    for column in columns:
        if(column != 'id'):
            non_id_columns.append(column)
    
    return non_id_columns

def common_elements(elements1 : list, elements2 : list) -> list:
    """Returns the union of the elements of both lists in the order they appear in the first list

    Args:
        elements1 (list): Source list of elements
        elements2 (list): Target list of elements

    Returns:
        list : Returns the union of the elements of both lists in the order they appear in the first list
    """
    _common_elements : list = []
    
    for element in elements1:
        if element in elements2:
            _common_elements.append(element)
    return _common_elements

def matching_arguments(workflow : dict, arguments : dict) -> bool:
    """Checks if given workflow's arguments that are shared with the target arguments have values that appear in the those arguments

    Args:
        workflow (dict): Dictionary of argument -> value for the given workflow
        arguments (dict): Dictionary of argument -> lists of values that are valid for the workflow in order for it to be matching

    Returns:
        bool : Checks if given workflow's arguments that are shared with the target arguments have values that appear in the those arguments
    """
    for argument, value in workflow.items():
        if argument in arguments and value not in arguments[argument]:
            return False  
    return True

def update_top_results(results : dict, new_workflow : dict, metric : str, keep_top_budget : bool) -> dict:
    """Based on its performance, sets the new workflow as the top one in 
       its budget/global category (don't / only keep the budget with top performance)  

    Args:
        results (dict): Budget -> Best workflow for giben budget
        new_workflow (dict): Arguments -> values for given workflow
        metric (str) : Metric upon which workflows are being compared
        keep_top_budget (bool): Keep only the workflow corresponding to the budget with the best performance

    Returns:
        dict : Updated Results Dictionary
    """
    
    _budget : int = new_workflow['budget']
    _current_top_workflow = (None if not results else results[next(iter(results))]) if keep_top_budget \
                            else (None if _budget not in results else results[_budget])
    
    if(_current_top_workflow is None or _current_top_workflow[metric] < new_workflow[metric]):
        if(keep_top_budget):
            return {_budget : new_workflow}
        else:
            results[_budget] = new_workflow
            return results
    return results

def retrieve_top_workflows(workflows : dict = None,
                           workflows_path : str = None,
                           store_path : str = None,
                           metric : str = 'auc',
                           top_budget : bool = False,
                           **arguments):
    """Takes a workflow dictionary or retrieves it from given path.
       Gathers the best workflows was specified comparison metric and argument values.
       Stores the best workflows in the given storage path.  

    Args:
        workflows (dict): Dictionary containing the workflows (Defaults to None)
        workflows_path (dict): Path from which the program will attempt to retrieve the workflows (Defaults to None)
        store_path (str) : Path in which the best workflows will be stored in json format (Defaults to None)
        metric (bool): Metric used to compare workflows (Default to 'auc')
        top_budget (bool): Store only the workflow for the budget with the best performance (Defaults to False)
        arguments (dict): Arguments and the corresponding values that workflows have to possess in order to be considered

    Returns:
        dict : Updated Results Dictionary
    """
    
    retrievable_metrics = ['time', 'auc', 'recall']
    
    if(workflows is not None):
        _workflows = workflows
    elif(workflows_path is not None):
        with open(workflows_path) as file:
            _workflows = json.load(file)
    else:
        raise ValueError("Please provide workflows dictionary / json file path.")
    
    if metric not in ['time', 'auc', 'recall']:
        raise AttributeError(
            'Metric ({}) does not exist. Please select one of the available. ({})'.format(
                metric, retrievable_metrics
                )
            )
     
    _results : dict = {} 
    # datasets, matchers and language models
    # for which we want to find the top workflows  
    datasets : List[str] = None if 'dataset' not in arguments else arguments['dataset'] 
    matchers : List[str] = None if 'matcher' not in arguments else arguments['matcher'] 
    lms : List[str] = None if 'language_model' not in arguments else arguments['language_model']
    
    _dataset_names : List[str] = _workflows.keys() if datasets is None else common_elements(datasets, workflows.keys())
    _current_workflows : List[dict] = []

    for _dataset_name in _dataset_names:
        _dataset_info : dict = _workflows[_dataset_name]
        _matcher_names = _dataset_info.keys() if matchers is None else common_elements(matchers, _dataset_info.keys())
        for _matcher_name in _matcher_names:
            _matcher_info : dict = _dataset_info[_matcher_name]
            if _matcher_name == 'EmbeddingsNNBPM':
                _lm_names = _matcher_info.keys() if lms is None else common_elements(lms, _matcher_info.keys())
                for _lm_workflows in _matcher_info[_lm_names]:
                    _current_workflows += _lm_workflows
            else:
                _current_workflows += _matcher_info
            for _current_workflow in _current_workflows:
                if(matching_arguments(workflow=_current_workflow, arguments=arguments)):
                    _results = update_top_results(results=_results, 
                                                  new_workflow=_current_workflow,
                                                  metric=metric,
                                                  keep_top_budget=top_budget)
    
    print(_results)                
    if (store_path is not None):
        with open(store_path, 'w', encoding="utf-8") as file:
            json.dump(_results, file, indent=4)
  

def add_entry(workflow : dict, dataframe_dictionary : dict) -> None:
    """Retrieves features and their values from the given workflow dictionary,
       and stores them in the to-be-constructed dataframe dictionary

    Args:
        workflow (dict): Dictionary containing workflow's arguments and their values
        dataframe_dictionary (dict): Dictionary that stores workflows arguments and their values - 
                                     to be transformed into columns
    """
    for feature, value in workflow.items():
        if(feature != 'tp_idx'):
            if feature not in dataframe_dictionary:
                dataframe_dictionary[feature] = []
            dataframe_dictionary[feature].append(value)
          
def workflows_to_dataframe(workflows : dict = None,
                           workflows_path : str = None,
                           store_path : str = None) -> pd.DataFrame:
    """Takes a workflow dictionary or retrieves it from given path.
       Stores all of its entries in a dataframe.
       Stores the dataframe in specified path if provided.

    Args:
        workflows (dict): Dictionary containing the workflows (Defaults to None)
        workflows_path (dict): Path from which the program will attempt to retrieve the workflows (Defaults to None)
        store_path (str) : Path in which the dataframe will be stored in json format (Defaults to None)

    Returns:
        pd.Dataframe : Dataframe containing the workflow entries in the given workflows dictionary
    """
    if(workflows is not None):
        _workflows = workflows
    elif(workflows_path is not None):
        with open(workflows_path) as file:
            _workflows = json.load(file)
    else:
        raise ValueError("Please provide workflows dictionary / json file path.")
    
    
    dataframe_dictionary : dict = {}
    workflows_dataframe : pd.DataFrame   
    
    for dataset in _workflows:
        dataset_info : dict = _workflows[dataset]
        for matcher in dataset_info:
            matcher_info : dict = dataset_info[matcher]
            current_workflows : list = []
            if(matcher == 'EmbeddingsNNBPM'):
                for lm in matcher_info:
                    current_workflows += matcher_info[lm]
            else:
                current_workflows += matcher_info
                
            for current_workflow in current_workflows:
                add_entry(current_workflow, dataframe_dictionary)
     
    workflows_dataframe = pd.DataFrame(dataframe_dictionary)           
    if(store_path is not None):
        workflows_dataframe.to_csv(store_path, index=False)
        
    return workflows_dataframe

# Frequency based Vectorization/Similarity evaluation Module   
class FrequencyEvaluator(ABC):
    def __init__(self, vectorizer : str, tokenizer : str, qgram : int) -> None:
        super().__init__()
        self.vectorizer_name : str = vectorizer
        self.tokenizer : str = tokenizer
        self.qgram : int = qgram
        self.analyzer = 'char' if 'char' in self.tokenizer else 'word'
        
        if self.vectorizer_name == 'tfidf' or self.vectorizer_name == 'boolean':
            self.vectorizer = TfidfVectorizer(analyzer='') if self.qgram is None else \
                            TfidfVectorizer(analyzer=self.analyzer, ngram_range=(self.qgram, self.qgram))
        elif self.vectorizer_name == 'tf':
            self.vectorizer = CountVectorizer(analyzer=self.analyzer) if self.qgram is None else \
                            CountVectorizer(analyzer=self.analyzer, ngram_range=(self.qgram, self.qgram))
        else:
            raise ValueError(f"{self.vectorizer_name}: Invalid Frequency Evaluator Model Name")
        
        self.dataset_identifier : str = None 
        self.indexing : str = None
        self.distance_matrix : np.ndarray = None
        self.distance_matrix_loaded : bool = False
        self.distance_matrix_indexing : str = None
      
    def save_distance_matrix(self) -> None:
        """Store the distance matrix of frequency evaluator in the hidden .dm directory within the execution path.
           The name of the file contains the vectorizer, tokenizer, dataset and metric, so it can be retrieved and
           used as precalculated distances matrix.
        """
        distance_matrix_file_name = '_'.join([self.indexing, self.dataset_identifier, self.vectorizer_name, self.tokenizer.split('_')[0], self.metric, "q" + str(self.qgram) + ".npy"])
        
        hidden_directory_path = os.path.join(os.getcwd(), ".dm")
        os.makedirs(hidden_directory_path, exist_ok=True)
        distance_matrix_file_path = os.path.join(hidden_directory_path, distance_matrix_file_name)
        try:
            print(f"Saving Distance Matrix -> {distance_matrix_file_path}")
            np.save(distance_matrix_file_path, self.distance_matrix) 
            pass
        except FileNotFoundError:
            print(f"Unable to save distance matrix -> {distance_matrix_file_path}")   
            
            
    def load_distance_matrix_from_path(self, path : str) -> np.ndarray:
        """Load the precalculated distance matrix for current execution's arguments combination.
        Args:
            path (str): Path to the distance matrix file
        Returns:
            np.ndarray: Precalculated distance matrix for current execution parameters combination
        """
        try:
            print(f"Loading Distance Matrix from: {path}")
            return np.load(path) 
        except FileNotFoundError:
            print(f"Unable to load distance matrix -> {path}")       
            
    def retrieve_distance_matrix_file_path(self) -> Tuple[str, str]:
        """Attemps to retrieve a precalculated DM from disk for current experiment
        Returns:
            str: Precalculated DM file path (None if doesn't exist)
        """
        
        _requested_indexing : str = self.indexing
        _opposite_indexing : str = "inorder" if (self.indexing == "reverse") else "reverse"
        _requested_indexing_file_name = '_'.join([_requested_indexing, self.dataset_identifier, self.vectorizer_name, self.tokenizer.split('_')[0], self.metric, "q" + str(self.qgram) + ".npy"])
        _opposite_indexing_file_name = '_'.join([_opposite_indexing, self.dataset_identifier, self.vectorizer_name, self.tokenizer.split('_')[0], self.metric, "q" + str(self.qgram) + ".npy"])
        
        hidden_directory_path = os.path.join(os.getcwd(), ".dm")
        os.makedirs(hidden_directory_path, exist_ok=True)
        
        
        _available_indexing : str = None
        _available_file_path : str = None
        _requested_indexing_file_path = os.path.join(hidden_directory_path, _requested_indexing_file_name)
        _opposite_indexing_file_path = os.path.join(hidden_directory_path, _opposite_indexing_file_name)
        
        
        if(os.path.exists(_requested_indexing_file_path) and os.path.isfile(_requested_indexing_file_path)):
            _available_indexing = _requested_indexing
            _available_file_path = _requested_indexing_file_path
        elif(os.path.exists(_opposite_indexing_file_path) and os.path.isfile(_opposite_indexing_file_path)):
            _available_indexing = _opposite_indexing
            _available_file_path = _opposite_indexing_file_path

        return (_available_indexing, _available_file_path)
    
    
    def distance_to_similarity_matrix(self, distance_matrix : np.ndarray) -> np.ndarray:
        """Transforms the input distance matrix into similarity matrix
        Args:
            distance_matrix (np.ndarray): Input pairwise distance matrix
        Returns:
            np.ndarray: Pairwise similarity matrix
        """
        
        if(self.metric == 'sqeuclidean'):
            return 1.0 / (1.0 + (distance_matrix ** 2))
        elif('cosine' in self.metric):
            return 1.0 - distance_matrix
        else:
            return distance_matrix
      
      
    def _get_sparse_matrix_method(self, metric : str) -> str:
        if(metric == 'sqeuclidean'):
            return 'euclidean'
        else:
            return metric
        
    def fit(self, 
            metric : str, 
            dataset_identifier : str,
            indexing : str,
            d1_entities : list = None, 
            d2_entities : list = None, 
            save_dm : bool = False) -> None:
        """Initializes the entities' corpus, and constructs the similarity matrix 
        Args:
            metric (str): Distance metric for entity strings
            dataset_identifier (str): Name of the dataset we are conducting our experiment on
            indexing (str): Indexing that the candidate entities follow
            d1_entities (list): List of D1 entities' string representations
            d2_entities (list): List of D2 entities' string representations
            save_dm (bool): Save the distance matrix in hidden directory on disk
        """
        if(d1_entities is None or d2_entities is None):
            raise NotImplementedError(f"{self.vectorizer_name} Frequency Evaluator Model - Dirty ER is not implemented yet")
        else:
            self.metric : str = metric
            self._entities_d1 : list = d1_entities
            self._entities_d2 : list = d2_entities
            self._entities_d1_num : int = len(self._entities_d1)
            self._entities_d2_num : int = len(self._entities_d2)
            self.save_dm : bool = save_dm
            self.dataset_identifier : str = dataset_identifier
            self.indexing : str = indexing
            
            _dm_indexing, _dm_path = self.retrieve_distance_matrix_file_path()
            if(_dm_path is not None):
                self.distance_matrix : np.ndarray = self.load_distance_matrix_from_path(path=_dm_path)
                self.distance_matrix_loaded : bool = True
                self.distance_matrix_indexing : str = _dm_indexing
            else:
                self.corpus = self._entities_d1 + self._entities_d2
                self._tf_limit = len(self._entities_d1)
                self.corpus_as_matrix = self.vectorizer.fit_transform(self.corpus)
                if self.vectorizer_name == 'boolean':
                    self.corpus_as_matrix = self.corpus_as_matrix.astype(bool).astype(int)

                self.distance_matrix : np.ndarray = self.distance_to_similarity_matrix(
                                                    distance_matrix=pairwise_distances(
                                                    self.corpus_as_matrix, 
                                                    metric=self._get_sparse_matrix_method(metric=self.metric)))

                self.distance_matrix_loaded : bool = False
                self.distance_matrix_indexing : str = self.indexing
                
                if(self.save_dm): 
                    self.save_distance_matrix()
                    
     
    def predict(self, id1 : int, id2 : int) -> float:
        """Returns the predicted similarity score for the given entities
        Args:
            id1 (int): id of an entity of the 1nd dataset within experiment context (not necessarily preloaded matrix)
            id2 (int): id of an entity of the 2nd dataset within experiment context (not necessarily preloaded matrix)
        Returns:
            float: Similarity score of entities with specified IDs
        """
        # candidates = np.vstack((self.corpus_as_matrix[id1], self.corpus_as_matrix[id2]))
        # distances = pairwise_distances(candidates, metric=self.metric)        
        # return 1.0 - distances[0][1]
        if(self.indexing == self.distance_matrix_indexing):
            return self.distance_matrix[id1][id2]
        # _id1 = (id1 + self._entities_d2_num) if (self.indexing == "inorder") else (id1 + self._entities_d1_num)
        # _id2 = (id2 - self._entities_d1_num) if (self.indexing == "inorder") else (id2 - self._entities_d2_num)
        _id1 = (id1 + self._entities_d2_num)
        _id2 = (id2 - self._entities_d1_num)

        return self.distance_matrix[_id1][_id2]

def read_data_from_json(json_path, base_dir, verbose=True):
    """
    Reads dataset details from a JSON file and returns a Data object.

    Parameters:
    - json_path (str): Path to the JSON configuration file.
    - verbose (bool): Whether to print information about the loaded datasets.

    Returns:
    - Data: A pyjedai Data object initialized with the dataset details.
    """
    # Load JSON configuration
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Extract common settings
    separator = config.get("separator", ",")
    engine = config.get("engine", "python")
    na_filter = config.get("na_filter", False)
    dataset_dir = config.get("dir", "")

    # Construct file paths
    d1_path = f"{base_dir}{dataset_dir}/{config['d1']}.{config.get('format', 'csv')}"
    d2_path = f"{base_dir}{dataset_dir}/{config['d2']}.{config.get('format', 'csv')}" if "d2" in config else None
    gt_path = f"{base_dir}{dataset_dir}/{config['gt']}.{config.get('format', 'csv')}" if "gt" in config else None

    # Load datasets
    d1 = pd.read_csv(d1_path, sep=separator, engine=engine, na_filter=na_filter)
    d2 = pd.read_csv(d2_path, sep=separator, engine=engine, na_filter=na_filter) if d2_path else None
    gt = pd.read_csv(gt_path, sep=separator, engine=engine) if gt_path else None

    # Initialize Data object
    data = Data(
        dataset_1=d1,
        id_column_name_1=config["d1_id"],
        dataset_name_1=config.get("d1", None),
        dataset_2=d2,
        id_column_name_2=config.get("d2_id", None),
        dataset_name_2=config.get("d2", None),
        ground_truth=gt,
        skip_ground_truth_processing=config.get("skip_ground_truth_processing", False)
    )

    if verbose:
        data.print_specs()
    
    return data
