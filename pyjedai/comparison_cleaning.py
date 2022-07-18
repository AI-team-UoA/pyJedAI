'''
Comparison cleaning methods
'''
import numpy as np
import os, sys
import tqdm
import math
import time
from tqdm.notebook import tqdm
from math import log10
from queue import PriorityQueue

# pyJedAI
from .datamodel import Data
from .utils import EMPTY, DISCRETIZATION_FACTOR, create_entity_index

class AbstractComparisonCleaning:
    '''
    TODO: add comment
    '''
    _progress_bar = None

    def __init__(self) -> None:
        self.data: Data
        self._limit: int
        self._num_of_blocks: int
        self._valid_entities: set() = set()
        self._entity_index: dict
        self._weighting_scheme: str
        self._blocks: dict() # initial blocks
        self.blocks = dict() # blocks after CC

    def process(
            self,
            blocks: dict,
            data: Data
    ) -> dict:
        '''
        TODO: add description
        '''
        start_time = time.time()
        
        self.data = data
        self._entity_index = create_entity_index(blocks, self.data.is_dirty_er)
        self._num_of_blocks = len(blocks)
        self._blocks: dict = blocks
        self._limit = self.data.num_of_entities if self.data.is_dirty_er else self.data.dataset_limit
        
        self._progress_bar = tqdm(total=self._limit, desc=self._method_name)
        
        blocks = self._apply_main_processing()
        
        self.execution_time = time.time() - start_time
        self._progress_bar.close()
        
        return blocks

class AbstractMetablocking(AbstractComparisonCleaning):
    '''
    Goal: Restructure a redundancy-positive block collection into a new
    one that contains substantially lower number of redundant
    and superfluous comparisons, while maintaining the original number of matching ones
    TODO: add comment
    '''

    def __init__(self) -> None:
        super().__init__()

        self._flags: np.array
        self._counters: np.array
        self._flags: np.array
        self._comparisons_per_entity: np.array
        self._node_centric: bool
        self._threshold: float
        self._distinct_comparisons: int
        self._comparisons_per_entity: np.array
        self._neighbors: set() = set()
        self._retained_neighbors: set() = set()
        self._block_assignments: int = 0
        self.weighting_scheme: str

    def _apply_main_processing(self) -> dict:
        self._counters = np.empty([self.data.num_of_entities], dtype=float)
        self._flags = np.empty([self.data.num_of_entities], dtype=int)
        
        if self.weighting_scheme == 'EJS':
            self._set_statistics()
            
        self._set_threshold()

        return self._prune_edges()

    def _get_weight(self, entity_id: int, neighbor_id: int) -> float:
        ws = self.weighting_scheme
        if ws == 'ARCS' or ws == 'CBS':
            return self._counters[neighbor_id]
        elif ws == 'ECBS':
            return float(
                self._counters[neighbor_id] *
                log10(float(self._num_of_blocks / len(self._entity_index[entity_id]))) *
                log10(float(self._num_of_blocks / len(self._entity_index[neighbor_id])))
            )
        elif ws == 'JS':
            return self._counters[neighbor_id] / (len(self._entity_index[entity_id]) + \
                    len(self._entity_index[neighbor_id]) - self._counters[neighbor_id])
        elif ws == 'EJS':
            probability = self._counters[neighbor_id] / (len(self._entity_index[entity_id]) + \
                            len(self._entity_index[neighbor_id]) - self._counters[neighbor_id])
            return float(probability * \
                    log10(self._distinct_comparisons / self._comparisons_per_entity[entity_id]) * \
                    log10(self._distinct_comparisons / self._comparisons_per_entity[neighbor_id]))
        elif ws == 'PEARSON_X2':
            # TODO: ChiSquared
            pass
        else:
            # TODO: Error handling
            print('This weighting scheme does not exist')

    def _normalize_neighbor_entities(self, block_key: str, entity_id: int) -> None:
        self._neighbors.clear()
        if self.data.is_dirty_er:
            if not self._node_centric:
                self._neighbors.update([neighbor_id for neighbor_id in self._blocks[block_key].entities_D1 if neighbor_id < entity_id])
            else:
                self._neighbors.update([neighbor_id for neighbor_id in self._blocks[block_key].entities_D1 if neighbor_id != entity_id])
        else:
            if entity_id < self.data.dataset_limit:
                self._neighbors.update(self._blocks[block_key].entities_D2)
            else:
                self._neighbors.update(self._blocks[block_key].entities_D1)

    def _discretize_comparison_weight(self, weight: float) -> int:
        return int(weight * DISCRETIZATION_FACTOR)

    def _set_statistics(self) -> None:
        self._distinct_comparisons = 0
        self._comparisons_per_entity = np.empty([self.data.num_of_entities], dtype=float)
        
        distinct_neighbors = set()
        for entity_id in range(0, self.data.num_of_entities, 1):
            if entity_id in self._entity_index:
                associated_blocks = self._entity_index[entity_id]
                distinct_neighbors.clear()
                for block_id in associated_blocks:
                    distinct_neighbors = set.union(
                        distinct_neighbors,
                        self._get_neighbor_entities(block_id, entity_id)
                    )
                self._comparisons_per_entity[entity_id] = len(distinct_neighbors)

                if self.data.is_dirty_er:
                    self._comparisons_per_entity[entity_id] -= 1

                self._distinct_comparisons += self._comparisons_per_entity[entity_id]
        self._distinct_comparisons /= 2

    def _get_neighbor_entities(self, block_id: int, entity_id: int) -> set:
        if not self.data.is_dirty_er and entity_id < self.data.dataset_limit:
            return self._blocks[block_id].entities_D2
        return self._blocks[block_id].entities_D1
        
class ComparisonPropagation(AbstractComparisonCleaning):
    '''
    TODO: ComparisonPropagation
    ''' 
    
    _method_name = "Comparison Propagation"
    _method_info = ": it eliminates all redundant comparisons from a set of overlapping blocks."
    
    def __init__(self) -> None:
        super().__init__()
    
    def _apply_main_processing(self) -> dict:
        self.blocks = dict()
        for i in range(0, self._limit):
            if i in self._entity_index:
                self._valid_entities.clear()
                
                associated_blocks = self._entity_index[i]
                for block_index in associated_blocks:
                    if self.data.is_dirty_er:
                        self._valid_entities.update([neighbor_id for neighbor_id in self._blocks[block_key].entities_D1 if i < neighbor_id])
                    else:
                        self._valid_entities.update(self._blocks[block_index].entities_D2)
                self.blocks[i] = self._valid_entities.copy()
            self._progress_bar.update(1)
            
        return self.blocks       
        
class WeightedEdgePruning(AbstractMetablocking):
    '''
    TODO: add comment
    '''
    _method_name = "Weighted Edge Pruning"
    _method_info = ": a Meta-blocking method that retains all comparisons \
                that have a weight higher than the average edge weight in the blocking graph."

    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__()
        self.weighting_scheme = weighting_scheme
        self._node_centric = False
        self._num_of_edges: float

    def _prune_edges(self) -> dict:
        for i in range(0, self._limit):
            self._process_entity(i)
            self._verify_valid_entities(i)
            self._progress_bar.update(1)

        return self.blocks

    def _process_entity(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return

        self._valid_entities.clear()
        self._flags[:] = EMPTY
        associated_blocks = self._entity_index[entity_id]
        
        for block_id in associated_blocks:
            self._normalize_neighbor_entities(block_id, entity_id)
            for neighbor_id in self._neighbors:
                if self._flags[neighbor_id] != entity_id:
                    self._counters[neighbor_id] = 0
                    self._flags[neighbor_id] = entity_id

                if self.weighting_scheme == 'ARCS':
                    self._counters[neighbor_id] += 1 / self._blocks[block_id].get_cardinality(self.data.is_dirty_er)
                else:
                    self._counters[neighbor_id] += 1
                self._valid_entities.add(neighbor_id)

    def _update_threshold(self, entity_id: int) -> None:
        self._num_of_edges += len(self._valid_entities)
        for neighbor_id in self._valid_entities:
            self._threshold += self._get_weight(entity_id, neighbor_id)

    def _set_threshold(self):
        self._num_of_edges = 0.0
        self._threshold = 0.0

        for i in range(0, self._limit):
            self._process_entity(i)
            self._update_threshold(i)

        self._threshold /= self._num_of_edges
        print(self._threshold)

    def _verify_valid_entities(self, entity_id: int) -> None:            
        self._retained_neighbors.clear()
        for neighbor_id in self._valid_entities:
            weight = self._get_weight(entity_id, neighbor_id)
            if self._threshold <= weight:
                self._retained_neighbors.add(neighbor_id)

        if len(self._retained_neighbors) > 0:
            self.blocks[entity_id] = self._retained_neighbors.copy()

class CardinalityEdgePruning(WeightedEdgePruning):
    '''
    TODO: CardinalityEdgePruning
    '''
    
    _method_name = "Cardinality Edge Pruning"
    _method_info = ": a Meta-blocking method that retains the comparisons \
                        that correspond to the top-K weighted edges in the blocking graph."
    
    def __init__(self, weighting_scheme: str = 'JS') -> None:
        super().__init__(weighting_scheme)
        
        self._minimum_weight: float = sys.float_info.min
        self._top_k_edges: PriorityQueue
        
    def _prune_edges(self) -> dict:
        self.blocks = dict()
        self._top_k_edges = PriorityQueue(int(2*self._threshold))
        for i in range(0, self._limit):
            self._process_entity(i)
            self._verify_valid_entities(i)
            self._progress_bar.update(1)
        
        while not self._top_k_edges.empty():
            comparison = self._top_k_edges.get()
            self.blocks.setdefault(comparison[1], set())
            self.blocks[comparison[1]].add(comparison[2])
        
        return self.blocks
     
    def _set_threshold(self) -> None:
        block_assignments = 0
        for block in self._blocks.values():
            block_assignments += block.get_size()
        self._threshold = block_assignments / 2
    
    def _verify_valid_entities(self, entity_id: int) -> None:
        for neighbor_id in self._valid_entities:
            weight = self._get_weight(entity_id, neighbor_id)
            if weight >= self._minimum_weight:
                self._top_k_edges.put(
                    (weight, entity_id, neighbor_id)
                )
                
                if self._threshold < self._top_k_edges.qsize():
                    self._minimum_weight = self._top_k_edges.get()[0]
                    
class CardinalityNodePruning(CardinalityEdgePruning):
    '''
    TODO: CardinalityNodePruning
    '''    
    _method_name = "Cardinality Node Pruning"
    _method_info = ": a Meta-blocking method that retains for every entity, \
                        the comparisons that correspond to its top-k weighted edges in the blocking graph."
    
    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__(weighting_scheme)
        self._nearest_entities: dict
        self._node_centric = True
        self._top_k_edges: PriorityQueue
        
    def _prune_edges(self) -> dict:
        self._nearest_entities = dict()
        for i in range(0, self.data.num_of_entities):
            self._process_entity(i)
            self._verify_valid_entities(i)
            self._progress_bar.update(1)
            
        return self._retain_valid_comparisons()
        
    def _retain_valid_comparisons(self) -> dict:
        self.blocks = dict()
        
        for i in range(0, self.data.num_of_entities):
            if i in self._nearest_entities:
                for neighbor_id in self._nearest_entities[i]: # Comparison is a triple: (id1, id2, weight)                
                    if self._is_valid_comparison(i, neighbor_id):
                        self.blocks.setdefault(i, set())
                        self.blocks[i].add(neighbor_id)
    
        return self.blocks

    def _is_valid_comparison(self, entity_id: int, neighbor_id: int) -> bool:
        if neighbor_id not in self._nearest_entities:
            return True
        if entity_id in self._nearest_entities[neighbor_id]:
            return entity_id < neighbor_id
        return True
        
    def _set_threshold(self) -> None:
        block_assignments = 0
        for block in self._blocks.values():
            block_assignments += block.get_size()
        self._threshold = max(1, block_assignments / self.data.num_of_entities)
    
    def _verify_valid_entities(self, entity_id: int) -> None: 
        self._top_k_edges = PriorityQueue(int(2*self._threshold))
        self._minimum_weight = sys.float_info.min
        for neighbor_id in self._valid_entities:
            weight = self._get_weight(entity_id, neighbor_id)
            if weight >= self._minimum_weight:
                self._top_k_edges.put(
                    (weight, entity_id, neighbor_id)
                )
                
                if self._threshold < self._top_k_edges.qsize():
                    self._minimum_weight = self._top_k_edges.get()[0]
                    
        if self._top_k_edges:         
            self._nearest_entities.setdefault(entity_id, set())
        while not self._top_k_edges.empty():
            comparison = self._top_k_edges.get()
            self._nearest_entities[entity_id].add(comparison[2])
        

class ReciprocalCardinalityNodePruning(CardinalityNodePruning):
    '''
    TODO: ReciprocalCardinalityNodePruning
    '''    
    
    _method_name = "Reciprocal Cardinality Node Pruning"
    _method_info = ": a Meta-blocking method that retains the comparisons \
                        that correspond to edges in the blocking graph that are among the top-k weighted  \
                            ones for both adjacent entities/nodes."
    
    def __init__(self, weighting_scheme: str = 'ARCS') -> None:
        super().__init__(weighting_scheme)

    def _is_valid_comparison(self, entity_id: int, neighbor_id: int) -> bool:
        if neighbor_id not in self._nearest_entities:
            return False
        if entity_id in self._nearest_entities[neighbor_id]:
            return  entity_id < neighbor_id
        return False
        
class WeightedNodePruning(WeightedEdgePruning):
    '''
    TODO: WeightedNodePruning
    '''    
    _method_name = "Weighted Node Pruning"
    _method_info = ": a Meta-blocking method that retains for every entity, the comparisons \
                    that correspond to edges in the blocking graph that are exceed \
                    the average edge weight in the respective node neighborhood."
            
    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__(weighting_scheme)
        self._average_weight: np.array
        self._node_centric = True
        self._limit = self.data.num_of_entities

    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        return weight if ((self._average_weight[entity_id] <= weight or \
                             self._average_weight[neighbor_id] <= weight) and 
                                entity_id < neighbor_id) else 0
        
    def _set_threshold(self):
        self._average_weight = np.empty([self.data.num_of_entities], dtype=float)
        for i in range(0, self.data.num_of_entities):
            self._process_entity(i)
            self._update_threshold(i)

    def _update_threshold(self, entity_id: int) -> None:
        self._average_weight[entity_id] = 0.0
        for neighbor_id in self._valid_entities:
            self._average_weight[entity_id] += super()._get_weight(entity_id, neighbor_id)        
        self._average_weight[entity_id] /= len(self._valid_entities)
        
    def _verify_valid_entities(self, entity_id: int) -> None:
        self._retained_neighbors.clear()
        for neighbor_id in self._valid_entities:
            if self._get_valid_weight(entity_id, neighbor_id):
                self._retained_neighbors.add(neighbor_id)
        if len(self._retained_neighbors) > 0:
            self.blocks[entity_id] = self._retained_neighbors.copy()

class BLAST(WeightedNodePruning):
    '''
    TODO: BLAST
    
    PEARSON_X2 scheme
    '''
    
    _method_name = "BLAST"
    _method_info = ": a Meta-blocking method that retains the comparisons \
                        that correspond to edges in the blocking graph that are exceed 1/4 of the sum \
                            of the maximum edge weights in the two adjacent node neighborhoods."
    
    def __init__(self, weighting_scheme: str = 'PEARSON_X2') -> None:
        super().__init__(weighting_scheme)
        
    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        edge_threshold = (self._average_weight[entity_id] + self._average_weight[neighbor_id]) / 4
        return edge_threshold <= weight and entity_id < neighbor_id
        
    def _update_threshold(self, entity_id: int) -> None:
        self._average_weight[entity_id] = 0.0
        for neighbor_id in self._valid_entities:
            self._average_weight[entity_id] = max(self._average_weight[entity_id], self._get_weight(entity_id, neighbor_id))
            
class ReciprocalWeightedNodePruning(WeightedNodePruning):
    '''
    TODO: ReciprocalWeightedNodePruning
    '''    
    
    _method_name = "Reciprocal Weighted Node Pruning"
    _method_info = ": a Meta-blocking method that retains the comparisons\
                        that correspond to edges in the blocking graph that are \
                            exceed the average edge weight in both adjacent node neighborhoods."
    
    def __init__(self, weighting_scheme: str = 'ARCS') -> None:
        super().__init__(weighting_scheme)
        
    
    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        return weight if ((self._average_weight[entity_id] <= weight and \
                             self._average_weight[neighbor_id] <= weight) and 
                                entity_id < neighbor_id) else 0
