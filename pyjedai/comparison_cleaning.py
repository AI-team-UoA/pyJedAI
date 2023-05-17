import sys
import warnings
from itertools import chain
from collections import defaultdict
from logging import warning
from math import log10
from queue import PriorityQueue
from time import time

import numpy as np
from tqdm.autonotebook import tqdm

from .evaluation import Evaluation

from .datamodel import Data, PYJEDAIFeature
from .utils import chi_square, create_entity_index

from abc import ABC, abstractmethod


class AbstractComparisonCleaning(PYJEDAIFeature):
    """Abstract class for Block cleaning
    """

    def __init__(self) -> None:
        super().__init__()
        self._limit: int
        self._num_of_blocks: int
        self._valid_entities: set() = set()
        self._entity_index: dict
        self._weighting_scheme: str
        self._blocks: dict # initial blocks
        self.blocks = dict() # blocks after CC
        self._node_centric: bool

    def process(
            self,
            blocks: dict,
            data: Data,
            tqdm_disable: bool = False
    ) -> dict:
        """Main method for comparison cleaning

        Args:
            blocks (dict): blocks creted from previous steps of pyjedai
            data (Data): dataset module
            tqdm_disable (bool, optional): Disables tqdm progress bars. Defaults to False.

        Returns:
            dict: cleaned blocks
        """
        start_time = time()
        self.tqdm_disable, self.data = tqdm_disable, data
        self._limit = self.data.num_of_entities \
                if self.data.is_dirty_er or self._node_centric else self.data.dataset_limit
        self._progress_bar = tqdm(
            total=self._limit,
            desc=self._method_name,
            disable=self.tqdm_disable
        )

        self._entity_index = create_entity_index(blocks, self.data.is_dirty_er)
        self._num_of_blocks = len(blocks)
        self._blocks: dict = blocks
        self._blocks = self._apply_main_processing()
        self.execution_time = time() - start_time
        self._progress_bar.close()

        return self._blocks

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes from D1:\n\t" + ', '.join(c for c in (self.attributes_1 if self.attributes_1 is not None \
                else self.data.dataset_1.columns)) +
            ("\nAttributes from D2:\n\t" + ', '.join(c for c in (self.attributes_2 if self.attributes_2 is not None \
                else self.data.dataset_2.columns)) if not self.data.is_dirty_er else "") +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

    @abstractmethod
    def _apply_main_processing(self) -> dict:
        pass

    @abstractmethod
    def _configuration(self) -> dict:
        pass
    
    def stats(self) -> None:
        pass

    def evaluate(self,
                 prediction,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:

        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " +
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        true_positives = 0
        total_matching_pairs = sum([len(block) for block in prediction.values()])
        for _, (id1, id2) in self.data.ground_truth.iterrows():
            id1 = self.data._ids_mapping_1[id1]
            id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er \
                                                else self.data._ids_mapping_2[id2]
            if (id1 in prediction and id2 in prediction[id1]) or   \
                (id2 in prediction and id1 in prediction[id2]):
                true_positives += 1

        eval_obj.calculate_scores(true_positives=true_positives, 
                                  total_matching_pairs=total_matching_pairs)
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)

class AbstractMetablocking(AbstractComparisonCleaning, ABC):
    """Restructure a redundancy-positive block collection into a new
        one that contains substantially lower number of redundant
        and superfluous comparisons, while maintaining the original number of matching ones
    """

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
        elif ws == 'X2':
            observed = [int(self._counters[neighbor_id]),
                        int(len(self._entity_index[entity_id])-self._counters[neighbor_id])]
            expected = [int(len(self._entity_index[neighbor_id])-observed[0]),
                        int(self._num_of_blocks - (observed[0] + observed[1] - self._counters[neighbor_id]))]
            return chi_square(np.array([observed, expected]))
        else:
            raise ValueError("This weighting scheme does not exist")

    def _normalize_neighbor_entities(self, block_key: str, entity_id: int) -> None:
        self._neighbors.clear()
        if self.data.is_dirty_er:
            if not self._node_centric:
                self._neighbors.update(
                    [neighbor_id for neighbor_id in self._blocks[block_key].entities_D1 \
                        if neighbor_id < entity_id]
                )
            else:
                self._neighbors.update(
                    [neighbor_id for neighbor_id in self._blocks[block_key].entities_D1 \
                        if neighbor_id != entity_id]
                )
        else:
            if entity_id < self.data.dataset_limit:
                self._neighbors.update(self._blocks[block_key].entities_D2)
            else:
                self._neighbors.update(self._blocks[block_key].entities_D1)

    def _set_statistics(self) -> None:
        self._distinct_comparisons = 0
        self._comparisons_per_entity = np.empty([self.data.num_of_entities], dtype=float)
        distinct_neighbors = set()
        for entity_id in range(0, self.data.num_of_entities, 1):
            if entity_id in self._entity_index:
                associated_blocks = self._entity_index[entity_id]
                distinct_neighbors.clear()
                # distinct_neighbors = set().union(*[
                #     self._get_neighbor_entities(block_id, entity_id) for block_id in associated_blocks
                # ])
                distinct_neighbors = set(chain.from_iterable(
                    self._get_neighbor_entities(block_id, entity_id) for block_id in associated_blocks
                ))
                # for block_id in associated_blocks:
                #     distinct_neighbors = set.union(
                #         distinct_neighbors,
                #         self._get_neighbor_entities(block_id, entity_id)
                #     )
                self._comparisons_per_entity[entity_id] = len(distinct_neighbors)

                if self.data.is_dirty_er:
                    self._comparisons_per_entity[entity_id] -= 1

                self._distinct_comparisons += self._comparisons_per_entity[entity_id]
        self._distinct_comparisons /= 2

    def _get_neighbor_entities(self, block_id: int, entity_id: int) -> set:
        return self._blocks[block_id].entities_D2 \
            if (not self.data.is_dirty_er and entity_id < self.data.dataset_limit) else \
                self._blocks[block_id].entities_D1

    @abstractmethod
    def _set_threshold(self):
        pass

    @abstractmethod
    def _prune_edges(self) -> dict:
        pass

class ComparisonPropagation(AbstractComparisonCleaning):
    """Comparison Propagation
    """

    _method_name = "Comparison Propagation"
    _method_short_name: str = "CP"
    _method_info = "Eliminates all redundant comparisons from a set of overlapping blocks."

    def __init__(self) -> None:
        super().__init__()
        self._node_centric = False

    def _apply_main_processing(self) -> dict:
        self.blocks = {}
        for i in range(0, self._limit):
            if i in self._entity_index:
                self._valid_entities.clear()
                associated_blocks = self._entity_index[i]
                for block_index in associated_blocks:
                    if self.data.is_dirty_er:
                        self._valid_entities.update(
                            [neighbor_id for neighbor_id in self._blocks[block_index].entities_D1 if i < neighbor_id]
                        )
                    else:
                        self._valid_entities.update(self._blocks[block_index].entities_D2)
                self.blocks[i] = self._valid_entities.copy()
            self._progress_bar.update(1)
        return self.blocks

    def _configuration(self) -> dict:
        return {
            "Node centric" : self._node_centric
        }

class WeightedEdgePruning(AbstractMetablocking):
    """A Meta-blocking method that retains all comparisons \
        that have a weight higher than the average edge weight in the blocking graph.
    """

    _method_name = "Weighted Edge Pruning"
    _method_short_name: str = "WEP"
    _method_info = "A Meta-blocking method that retains all comparisons " + \
                "that have a weight higher than the average edge weight in the blocking graph."

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
        self._flags[:] = -1
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
        if entity_id not in self._entity_index:
            return

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

    def _verify_valid_entities(self, entity_id: int) -> None:    
        if entity_id not in self._entity_index:
            return

        self._retained_neighbors.clear()
        for neighbor_id in self._valid_entities:
            weight = self._get_weight(entity_id, neighbor_id)
            if self._threshold <= weight:
                self._retained_neighbors.add(neighbor_id)

        if len(self._retained_neighbors) > 0:
            self.blocks[entity_id] = self._retained_neighbors.copy()

    def _configuration(self) -> dict:
        return {
            "Node centric" : self._node_centric,
            "Weighting scheme" : self.weighting_scheme
        }

class CardinalityEdgePruning(WeightedEdgePruning):
    """A Meta-blocking method that retains the comparisons \
            that correspond to the top-K weighted edges in the blocking graph.
    """

    _method_name = "Cardinality Edge Pruning"
    _method_short_name: str = "CEP"
    _method_info = "A Meta-blocking method that retains the comparisons " + \
                        "that correspond to the top-K weighted edges in the blocking graph."

    def __init__(self, weighting_scheme: str = 'JS') -> None:
        super().__init__(weighting_scheme)
        self._minimum_weight: float = sys.float_info.min
        self._top_k_edges: PriorityQueue

    def _prune_edges(self) -> dict:
        self.blocks = defaultdict(set)
        self._top_k_edges = PriorityQueue(int(2*self._threshold))
        for i in range(0, self._limit):
            self._process_entity(i)
            self._verify_valid_entities(i)
            self._progress_bar.update(1)
        while not self._top_k_edges.empty():
            comparison = self._top_k_edges.get()
            self.blocks[comparison[1]].add(comparison[2])

        return self.blocks

    def _set_threshold(self) -> None:
        block_assignments = 0
        for block in self._blocks.values():
            block_assignments += block.get_size()
        self._threshold = block_assignments / 2

    def _verify_valid_entities(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return

        for neighbor_id in self._valid_entities:
            weight = self._get_weight(entity_id, neighbor_id)
            if weight >= self._minimum_weight:
                self._top_k_edges.put(
                    (weight, entity_id, neighbor_id)
                )
                if self._threshold < self._top_k_edges.qsize():
                    self._minimum_weight = self._top_k_edges.get()[0]

class CardinalityNodePruning(CardinalityEdgePruning):
    """A Meta-blocking method that retains for every entity, \
        the comparisons that correspond to its top-k weighted edges in the blocking graph."
    """

    _method_name = "Cardinality Node Pruning"
    _method_short_name: str = "CNP"
    _method_info = "A Meta-blocking method that retains for every entity, " + \
                    "the comparisons that correspond to its top-k weighted edges in the blocking graph."

    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__(weighting_scheme)
        self._nearest_entities: dict
        self._node_centric = True
        self._top_k_edges: PriorityQueue

    def _prune_edges(self) -> dict:
        self._nearest_entities = dict()
        for i in range(0, self._limit):
            self._process_entity(i)
            self._verify_valid_entities(i)
            self._progress_bar.update(1)
        return self._retain_valid_comparisons()

    def _retain_valid_comparisons(self) -> dict:
        self.blocks = dict()
        for i in range(0, self.data.num_of_entities):
            if i in self._nearest_entities:
                for neighbor_id in self._nearest_entities[i]:
                    # Comparison is a triple: (id1, id2, weight)
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
        if entity_id not in self._entity_index:
            return
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
    """A Meta-blocking method that retains the comparisons \
        that correspond to edges in the blocking graph that are among the top-k weighted  \
        ones for both adjacent entities/nodes.
    """

    _method_name = "Reciprocal Cardinality Node Pruning"
    _method_short_name: str = "RCNP"
    _method_info = "A Meta-blocking method that retains the comparisons " + \
                    "that correspond to edges in the blocking graph that are among " + \
                    "the top-k weighted ones for both adjacent entities/nodes."

    def __init__(self, weighting_scheme: str = 'ARCS') -> None:
        super().__init__(weighting_scheme)

    def _is_valid_comparison(self, entity_id: int, neighbor_id: int) -> bool:
        if neighbor_id not in self._nearest_entities:
            return False
        if entity_id in self._nearest_entities[neighbor_id]:
            return  entity_id < neighbor_id
        return False

class WeightedNodePruning(WeightedEdgePruning):
    """A Meta-blocking method that retains for every entity, the comparisons \
        that correspond to edges in the blocking graph that are exceed \
        the average edge weight in the respective node neighborhood.
    """

    _method_name = "Weighted Node Pruning"
    _method_short_name: str = "WNP"
    _method_info = "A Meta-blocking method that retains for every entity, the comparisons \
                    that correspond to edges in the blocking graph that are exceed \
                    the average edge weight in the respective node neighborhood."

    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__(weighting_scheme)
        self._average_weight: np.array
        self._node_centric = True

    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        return weight if ((self._average_weight[entity_id] <= weight or \
                             self._average_weight[neighbor_id] <= weight) and 
                                entity_id < neighbor_id) else 0

    def _set_threshold(self):
        self._average_weight = np.empty([self.data.num_of_entities], dtype=float)
        for i in range(0, self._limit):
            self._process_entity(i)
            self._update_threshold(i)

    def _update_threshold(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return
        self._average_weight[entity_id] = 0.0
        for neighbor_id in self._valid_entities:
            self._average_weight[entity_id] += super()._get_weight(entity_id, neighbor_id)
        self._average_weight[entity_id] /= len(self._valid_entities)

    def _verify_valid_entities(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return
        self._retained_neighbors.clear()
        for neighbor_id in self._valid_entities:
            if self._get_valid_weight(entity_id, neighbor_id):
                self._retained_neighbors.add(neighbor_id)
        if len(self._retained_neighbors) > 0:
            self.blocks[entity_id] = self._retained_neighbors.copy()

class BLAST(WeightedNodePruning):
    """Meta-blocking method that retains the comparisons \
        that correspond to edges in the blocking graph that are exceed 1/4 of the sum \
        of the maximum edge weights in the two adjacent node neighborhoods.
    """

    _method_name = _method_short_name = "BLAST"
    _method_info = "Meta-blocking method that retains the comparisons " + \
                "that correspond to edges in the blocking graph that are exceed 1/4 of the sum " + \
                "of the maximum edge weights in the two adjacent node neighborhoods."

    def __init__(self, weighting_scheme: str = 'X2') -> None:
        super().__init__(weighting_scheme)

    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        edge_threshold = (self._average_weight[entity_id] + self._average_weight[neighbor_id]) / 4
        return edge_threshold <= weight and entity_id < neighbor_id

    def _update_threshold(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return
        self._average_weight[entity_id] = 0.0
        for neighbor_id in self._valid_entities:
            self._average_weight[entity_id] = \
                max(self._average_weight[entity_id], self._get_weight(entity_id, neighbor_id))

class ReciprocalWeightedNodePruning(WeightedNodePruning):
    """Meta-blocking method that retains the comparisons\
        that correspond to edges in the blocking graph that are \
        exceed the average edge weight in both adjacent node neighborhoods.
    """

    _method_name = "Reciprocal Weighted Node Pruning"
    _method_short_name: str = "RWNP"
    _method_info = "Meta-blocking method that retains the comparisons " + \
                    "that correspond to edges in the blocking graph that are " + \
                    "exceed the average edge weight in both adjacent node neighborhoods."

    def __init__(self, weighting_scheme: str = 'ARCS') -> None:
        super().__init__(weighting_scheme)

    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        return weight if ((self._average_weight[entity_id] <= weight and \
                             self._average_weight[neighbor_id] <= weight) and
                                entity_id < neighbor_id) else 0

class ProgressiveCardinalityEdgePruning(CardinalityEdgePruning):
    def __init__(self, weighting_scheme: str = 'JS', budget: int = 0) -> None:
        super().__init__(weighting_scheme)
        self._budget = budget

    def _set_threshold(self) -> None:
        self._threshold = self._budget

    def process(self, blocks: dict, data: Data, tqdm_disable: bool = False, cc: AbstractMetablocking = None) -> dict:

        if(cc is None):
            return super().process(blocks, data, tqdm_disable)
        else:
            self._threshold = self._budget
            self._top_k_edges = PriorityQueue(int(2*self._threshold))
            self._minimum_weight = sys.float_info.min            
            self.trimmed_blocks : dict = defaultdict(set)

            for entity_id, neighbors in blocks.items():
                for neighbor_id in neighbors:
                    weight = cc._get_weight(entity_id, neighbor_id)
                    if weight >= self._minimum_weight:
                        self._top_k_edges.put(
                        (weight, entity_id, neighbor_id)
                        )
                        if self._threshold < self._top_k_edges.qsize():
                            self._minimum_weight = self._top_k_edges.get()[0]

            while not self._top_k_edges.empty():
                comparison = self._top_k_edges.get()
                self.trimmed_blocks[comparison[1]].add(comparison[2])

            return self.trimmed_blocks

class ProgressiveCardinalityNodePruning(CardinalityNodePruning):
    def __init__(self, weighting_scheme: str = 'CBS', budget: int = 0) -> None:
        super().__init__(weighting_scheme)
        self._budget = budget

    def _set_threshold(self) -> None:
        self._threshold = max(1, 2 * self._budget / self.data.num_of_entities)

    def process(self, blocks: dict, data: Data, tqdm_disable: bool = False, cc: AbstractMetablocking = None) -> dict:

        if(cc is None):
            return super().process(blocks, data, tqdm_disable)
        else:
            self._threshold = max(1, 2 * self._budget / data.num_of_entities)           
            self.trimmed_blocks : dict = defaultdict(set)

            for entity_id, neighbors in blocks.items():
                self._minimum_weight = sys.float_info.min
                self._top_k_edges = PriorityQueue(int(2*self._threshold))
                for neighbor_id in neighbors:
                    weight = cc._get_weight(entity_id, neighbor_id)
                    if weight >= self._minimum_weight:
                        self._top_k_edges.put(
                        (weight, entity_id, neighbor_id)
                        )
                        if self._threshold < self._top_k_edges.qsize():
                            self._minimum_weight = self._top_k_edges.get()[0]

                while not self._top_k_edges.empty():
                    comparison = self._top_k_edges.get()
                    self.trimmed_blocks[entity_id].add(comparison[2])

        return self.trimmed_blocks        

def get_meta_blocking_approach(acronym: str, w_scheme: str, budget: int = 0) -> any:
    """Return method by acronym

    Args:
        acronym (str): Method acronym
        w_scheme (str): Weighting Scheme name

    Returns:
        any: Comparison Cleaning Method
    """
    if acronym == "BLAST":
        return BLAST(w_scheme)
    elif acronym == "CEP":
        return CardinalityEdgePruning(w_scheme)
    elif acronym == "CNP":
        return CardinalityNodePruning(w_scheme)
    elif acronym == "RCNP":
        return ReciprocalCardinalityNodePruning(w_scheme)
    elif acronym == "RWNP":
        return ReciprocalWeightedNodePruning(w_scheme)
    elif acronym == "WEP":
        return WeightedEdgePruning(w_scheme)
    elif acronym == "WNP":
        return WeightedNodePruning(w_scheme)
    elif acronym == "PCEP":
        return ProgressiveCardinalityEdgePruning(w_scheme, budget)
    elif acronym == "PCNP":
        return ProgressiveCardinalityNodePruning(w_scheme, budget)
    else:
        warnings.warn("Wrong meta-blocking approach selected. Returning Comparison Propagation.")
        return ComparisonPropagation()
