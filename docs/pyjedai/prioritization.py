"""Entity Matching Prioritization Module
"""
import numpy as np
from time import time
import matplotlib.pyplot as plt
from .matching import EntityMatching
from .comparison_cleaning import (
    ComparisonPropagation,
    ProgressiveCardinalityEdgePruning,
    ProgressiveCardinalityNodePruning,
    GlobalProgressiveSortedNeighborhood,
    LocalProgressiveSortedNeighborhood,
    ProgressiveEntityScheduling)
from .joins import PETopKJoin
from .vector_based_blocking import EmbeddingsNNBlockBuilding

from networkx import Graph
from py_stringmatching.similarity_measure.cosine import Cosine
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.similarity_measure.generalized_jaccard import \
    GeneralizedJaccard
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.jaro import Jaro
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.overlap_coefficient import \
    OverlapCoefficient
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.whitespace_tokenizer import \
    WhitespaceTokenizer
from sklearn.metrics.pairwise import pairwise_distances
from tqdm.autonotebook import tqdm

from .datamodel import Data, PYJEDAIFeature
from .evaluation import Evaluation
from .matching import EntityMatching
from .comparison_cleaning import AbstractMetablocking
from queue import PriorityQueue
from random import sample
from abc import abstractmethod
from typing import Tuple, List
from .utils import (
    SubsetIndexer,
    DatasetScheduler,
    EntityScheduler,
    is_infinite,
    PredictionData,
    reverse_data_indexing,
    reverse_blocks_entity_indexing,
    sorted_enumerate,
    canonical_swap,
    WordQgramTokenizer,
    cosine,
    get_qgram_from_tokenizer_name,
    FrequencyEvaluator)
import pandas as pd
import os
from collections import defaultdict
import sys
from faiss import METRIC_INNER_PRODUCT, METRIC_L2
import json
import re


# Directory where the whoosh index is stored
INDEXER_DIR='.indexer'

metrics_mapping = {
    'edit_distance': Levenshtein(),
    'cosine' : Cosine(),
    'jaro' : Jaro(),
    'jaccard' : Jaccard(),
    'generalized_jaccard' : GeneralizedJaccard(),
    'dice': Dice(),
    'overlap_coefficient' : OverlapCoefficient(),
}

vector_metrics_mapping = {
    'cosine': cosine
}

string_metrics = [
    'jaro', 'edit_distance'
]

set_metrics = [
    'cosine', 'dice', 'generalized_jaccard', 'jaccard', 'overlap_coefficient'
]

vector_metrics = [ 
    'cosine', 'dice', 'jaccard'
]

whoosh_index_metrics = [
    'TF-IDF', 'Frequency', 'PL2', 'BM25F'
] 

faiss_metrics = [
    'cosine', 'euclidean'
]

magellan_metrics = string_metrics + set_metrics
available_metrics = magellan_metrics + vector_metrics + whoosh_index_metrics + faiss_metrics

#
# Tokenizers
#
char_qgram_tokenizers = { 'char_'+ str(i) + 'gram':i for i in range(1, 7) }
word_qgram_tokenizers = { 'word_'+ str(i) + 'gram':i for i in range(1, 7) }
magellan_tokenizers = ['white_space_tokenizer']

tfidf_tokenizers = [ 'tfidf_' + cq for cq in char_qgram_tokenizers.keys() ] + \
                    [ 'tfidf_' + wq for wq in word_qgram_tokenizers.keys() ]

tf_tokenizers = [ 'tf_' + cq for cq in char_qgram_tokenizers.keys() ] + \
                    [ 'tf_' + wq for wq in word_qgram_tokenizers.keys() ]
                        
boolean_tokenizers = [ 'boolean_' + cq for cq in char_qgram_tokenizers.keys() ] + \
                        [ 'boolean_' + wq for wq in word_qgram_tokenizers.keys() ]

vector_tokenizers = tfidf_tokenizers + tf_tokenizers + boolean_tokenizers

available_tokenizers = [key for key in char_qgram_tokenizers] + [key for key in word_qgram_tokenizers] + magellan_tokenizers + vector_tokenizers

class ProgressiveMatching(EntityMatching):
    """Applies the matching process to a subset of available pairs progressively 
    """

    _method_name: str = "Progressive Matching"
    _method_info: str = "Applies the matching process to a subset of available pairs progressively "
    def __init__(
            self,
            similarity_function: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            vectorizer : str = None,
            qgram : int = 1,
            similarity_threshold: float = 0.0,
            tokenizer_return_unique_values = True, # unique values or not
            attributes: any = None,
        ) -> None:

        super().__init__(metric=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=vectorizer,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)
        self.similarity_function : str = similarity_function
        self.dataset_identifier : str = None
        
    def predict(self,
            data: Data,
            blocks: dict,
            dataset_identifier: str = "dataset",
            budget: int = 0,
            algorithm : str = 'HB',
            indexing : str = 'inorder',
            comparison_cleaner: AbstractMetablocking = None,
            tqdm_disable: bool = False,
            emit_all_tps_stop : bool = False) -> List[Tuple[float, int, int]]:
        """Main method of  progressive entity matching. Inputs a set of blocks and outputs a list \
           that contains duplets of ids corresponding to candidate pairs to emit.
            Args:
                blocks (dict): blocks of entities
                data (Data): dataset module
                tqdm_disable (bool, optional): Disables progress bar. Defaults to False.
                method (str) : DFS/BFS/Hybrid approach for specified algorithm
                emit_all_tps_stop (bool) : Stop emission once all true positives are found
            Returns:
                networkx.Graph: entity ids (nodes) and similarity scores between them (edges)
        """
        start_time = time()
        self.tqdm_disable = tqdm_disable
        self._budget : int = budget
        self._indexing : str = indexing
        self._comparison_cleaner: AbstractMetablocking = comparison_cleaner
        self._algorithm : str= algorithm
        self._emit_all_tps_stop : bool = emit_all_tps_stop
        self.duplicate_emitted : dict = None if not self._emit_all_tps_stop else {}
        self._prediction_data : PredictionData = None
        self.data : Data = data
        self.duplicate_of = data.duplicate_of
        self.scheduler : DatasetScheduler = None
        self.dataset_identifier : str = dataset_identifier

        if not blocks:
            raise ValueError("Empty blocks structure")
        
        if self.data.is_dirty_er and self._indexing == 'bilateral':
            raise ValueError("Cannot apply bilateral indexing to dirty Entity Resolution (single dataset)")
            
        _inorder_blocks = blocks  
        self._pairs_top_score : dict = defaultdict(lambda: -1)
        all_blocks = list(blocks.values())
        self._progress_bar = tqdm(total=len(blocks),
                        desc=self._method_name,
                        disable=self.tqdm_disable)
        
        if(indexing == 'bilateral'): self._indexing = 'inorder'
        if(self._indexing == 'inorder'):
            if 'Block' in str(type(all_blocks[0])):
                self._predict_raw_blocks(blocks)
            elif isinstance(all_blocks[0], set):
                if(self._comparison_cleaner == None):
                    raise AttributeError("No precalculated weights were given from the CC step") 
                self._predict_prunned_blocks(blocks)
            else:
                raise AttributeError("Wrong type of Blocks")
            self._schedule_candidates()
            
            
        if(indexing == 'bilateral'): self._indexing = 'reverse'
        if(self._indexing == 'reverse'):
            _reverse_blocks = reverse_blocks_entity_indexing(_inorder_blocks, self.data)
            self.data = reverse_data_indexing(self.data)
            if 'Block' in str(type(all_blocks[0])):
                self._predict_raw_blocks(_reverse_blocks)
            elif isinstance(all_blocks[0], set):
                if(self._comparison_cleaner == None):
                    raise AttributeError("No precalculated weights were given from the CC step") 
                self._predict_prunned_blocks(_reverse_blocks)
            else:
                raise AttributeError("Wrong type of Blocks")
            self._schedule_candidates()
        
        self._gather_top_pairs()
        self.execution_time = time() - start_time
        self._progress_bar.close()
        
        return self.pairs
    
    
    def _store_id_mappings(self) -> None:
        """Stores the mapping [Workflow ID -> Dataframe ID] for the current indexing phase
        """
        if(self._indexing == "inorder"):
            self._inorder_d1_id = self.data._gt_to_ids_reversed_1
            self._inorder_d2_id = self.data._gt_to_ids_reversed_2  
        if(self._indexing == "reverse"):
            self._reverse_d1_id = self.data._gt_to_ids_reversed_1
            self._reverse_d2_id = self.data._gt_to_ids_reversed_2 
      
    def _schedule_candidates(self) -> None:
        """Translates the workflow identifiers back into dataframe identifiers
           Populates the dataset scheduler with the candidate pairs of the current indexing stage
        """
        self.scheduler = DatasetScheduler(budget=float('inf') if self._emit_all_tps_stop else self._budget, global_top=(self._algorithm=="TOP")) if self.scheduler == None else self.scheduler
        self._store_id_mappings()
        
        for score, entity, candidate in self.pairs:            
            # entities of first and second dataframe in the context of the current indexing
            d1_entity, d2_entity = (entity, candidate) if(entity < candidate) else (candidate, entity)
            d1_map, d2_map = (self._inorder_d1_id, self._inorder_d2_id) if (self._indexing == 'inorder') else (self._reverse_d1_id, self._reverse_d2_id)
            
            # print(f"#############################################################")
            # print(f"Score: {score}")
            # print(f"---------------Workflow IDs [{self._indexing}]---------------")
            # print(f"Entity: {entity}")
            # print(f"Candidate: {candidate}")
            # print(f"---------------Workflow IDs [D1 context Ent First]---------------")
            # print(f"D1 Entity: {d1_entity}")
            # print(f"D2 Entity: {d2_entity}")
            
            # the dataframe ids of the entities from first and second dataset in the context of indexing
            d1_entity_df_id, d2_entity_df_id = (d1_map[d1_entity], d2_map[d2_entity])
            _inorder_d1_entity_df_id, _inorder_d2_entity_df_id = (d1_entity_df_id, d2_entity_df_id) if (self._indexing == 'inorder') else (d2_entity_df_id, d1_entity_df_id)
            if(self._emit_all_tps_stop and _inorder_d2_entity_df_id in self.duplicate_of[_inorder_d1_entity_df_id]):
                self.duplicate_emitted[(_inorder_d1_entity_df_id, _inorder_d2_entity_df_id)] = False          
            
            # in the case of reverse indexing stage, adjust the workflow identifiers of the entities so we can differ them from inorder entity ids
            d1_entity = d1_entity if(self._indexing == 'inorder') else d1_entity + self.data.num_of_entities
            d2_entity = d2_entity if(self._indexing == 'inorder') else d2_entity + self.data.num_of_entities
            
            # print(f"---------------Dataframe IDs [{self._indexing}]---------------")
            # print(f"D1 Entity DF ID: {d1_entity_df_id}")
            # print(f"D2 Entity DF ID: {d2_entity_df_id}")
            # print(f"---------------Inorder Dataframe IDs [{self._indexing}]---------------")
            # print(f"Inorder D1 Entity DF ID: {_inorder_d1_entity_df_id}")
            # print(f"Inorder D2 Entity DF ID: {_inorder_d2_entity_df_id}")
            # print(f"---------------Scheduler IDs [D1 context Ent First]---------------")
            # print(f"D1 Entity: {d1_entity}")
            # print(f"D2 Entity: {d2_entity}")
            # if(_inorder_d2_entity_df_id in self.duplicate_of[_inorder_d1_entity_df_id]):
            #     print("^ THIS IS A TRUE POSITIVE ^") 
            # we want entities to be inserted in D1 -> D2 order (current context e.x. reverse) which translates to D2 -> D1 order (reverse context e.x. inorder)
            self.scheduler._insert_entity_neighbor(d1_entity, d2_entity, score)
            
    def _inorder_phase_entity(self, id : int) -> bool:
        """Given identifier corresponds to an entity proposed in the inorder indexing phase

        Args:
            id (int): Identifier

        Returns:
            bool: Identifier proposed in the inorder phase
        """
        return id < self.data.num_of_entities
    
    def _retrieve_entity_df_id(self, id : int) -> int:
        """Returns the corresponding id in the dataframe of the given entity id in the context of its indexing phase 

        Args:
            id (int): Workflow Identifier

        Returns:
            int: Dataframe Identifier
        """
        _workflow_id : int
        _df_id_of : dict
        if(self._inorder_phase_entity(id)):
            _workflow_id = id
            _df_id_of = self._inorder_d1_id if (_workflow_id < len(self._inorder_d1_id)) else self._inorder_d2_id
        else:
            _workflow_id = id - self.data.num_of_entities
            _df_id_of = self._reverse_d1_id if (_workflow_id < len(self._reverse_d1_id)) else self._reverse_d2_id
            
        return _df_id_of[_workflow_id]    
    
    def _gather_top_pairs(self) -> None:
        """Emits the pairs from the scheduler based on the defined algorithm
        """
        self.scheduler._sort_neighborhoods_by_avg_weight()
        self.pairs = self.scheduler._emit_pairs(method=self._algorithm, data=self.data)
        
        _identified_pairs = []
        for score, entity, candidate in self.pairs:
            _inorder_entities : bool = self._inorder_phase_entity(entity)
            entity, candidate = (self._retrieve_entity_df_id(entity), self._retrieve_entity_df_id(candidate))
            entity, candidate = (entity, candidate) if _inorder_entities else (candidate, entity)
            _identified_pairs.append((score, entity, candidate))
            
        self.pairs = _identified_pairs

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
        total_matching_pairs = prediction.number_of_edges()
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
    
    def get_prediction_data(self) -> PredictionData:
        if(self._prediction_data is None):
            raise ValueError("Pairs not emitted yet - No Data to show")
        return self._prediction_data
    
    def get_total_emissions(self) -> int:
        return self.get_prediction_data().get_total_emissions()
    
    def get_cumulative_recall(self) -> float:
        return self.get_prediction_data().get_cumulative_recall()
    
    def get_normalized_auc(self) -> float:
        return self.get_prediction_data().get_normalized_auc()
    
    def set_prediction_data(self, prediction_data : PredictionData):
        self._prediction_data : PredictionData = prediction_data
class BlockIndependentPM(ProgressiveMatching):
    """Applies the matching process to a subset of available pairs progressively 
    """

    _method_name: str = "Progressive Matching"
    _method_info: str = "Applies the matching process to a subset of available pairs progressively "

    def __init__(
            self,
            similarity_function: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            vectorizer : str = None,
            qgram : int = 1,
            similarity_threshold: float = 0.0,
            tokenizer_return_unique_values = True, # unique values or not
            attributes: any = None,
        ) -> None:

        super().__init__(similarity_function=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=vectorizer,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)
        
    def predict(self,
            data: Data,
            blocks: dict,
            dataset_identifier: str = "dataset",
            budget: int = 0,
            algorithm : str = 'HB',
            indexing : str = 'inorder',
            comparison_cleaner: AbstractMetablocking = None,
            tqdm_disable: bool = False,
            emit_all_tps_stop : bool = False) -> List[Tuple[float, int, int]]:
        """Main method of  progressive entity matching. Inputs a set of blocks and outputs a list \
           that contains duplets of ids corresponding to candidate pairs to emit.
            Args:
                blocks (dict): blocks of entities
                data (Data): dataset module
                tqdm_disable (bool, optional): Disables progress bar. Defaults to False.
                method (str) : DFS/BFS/Hybrid approach for specified algorithm
                emit_all_tps_stop (bool) : Stop emission once all true positives are found
            Returns:
                networkx.Graph: entity ids (nodes) and similarity scores between them (edges)
        """
        start_time = time()
        self.tqdm_disable = tqdm_disable
        self._budget : int = budget
        self._indexing : str = indexing
        self._comparison_cleaner: AbstractMetablocking = comparison_cleaner
        self._algorithm : str= algorithm
        self._emit_all_tps_stop : bool = emit_all_tps_stop
        self.duplicate_emitted : dict = None 
        self._prediction_data : PredictionData = None
        self.data : Data = data
        self.duplicate_of = data.duplicate_of
        self.scheduler : DatasetScheduler = None
        self.dataset_identifier : str = dataset_identifier
        
        if self.data.is_dirty_er and self._indexing == 'bilateral':
            raise ValueError("Cannot apply bilateral indexing to dirty Entity Resolution (single dataset)")
            
        _inorder_blocks = blocks  
        self._pairs_top_score : dict = defaultdict(lambda: -1)
        all_blocks = list(blocks.values()) if blocks is not None else None
        self._progress_bar = tqdm(total=len(blocks) if blocks is not None else 0,
                        desc=self._method_name,
                        disable=self.tqdm_disable)
        
        if(indexing == 'bilateral'): self._indexing = 'inorder'
        if(self._indexing == 'inorder'):
            if all_blocks is None or 'Block' in str(type(all_blocks[0])):
                self._predict_raw_blocks(blocks)
            elif isinstance(all_blocks[0], set):
                if(self._comparison_cleaner == None):
                    raise AttributeError("No precalculated weights were given from the CC step") 
                self._predict_prunned_blocks(blocks)
            else:
                raise AttributeError("Wrong type of Blocks")
            self._schedule_candidates()
            
            
        if(indexing == 'bilateral'): self._indexing = 'reverse'
        if(self._indexing == 'reverse'):
            _reverse_blocks = reverse_blocks_entity_indexing(_inorder_blocks, self.data)
            self.data = reverse_data_indexing(self.data)
            if all_blocks is None or 'Block' in str(type(all_blocks[0])):
                self._predict_raw_blocks(_reverse_blocks)
            elif isinstance(all_blocks[0], set):
                if(self._comparison_cleaner == None):
                    raise AttributeError("No precalculated weights were given from the CC step") 
                self._predict_prunned_blocks(_reverse_blocks)
            else:
                raise AttributeError("Wrong type of Blocks")
            self._schedule_candidates()
        
        self._gather_top_pairs()
        self.execution_time = time() - start_time
        self._progress_bar.close()
        
        return self.pairs

class HashBasedProgressiveMatching(ProgressiveMatching):
    """Applies hash based candidate graph prunning, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Hash Based Progressive Matching"
    _method_info: str = "Applies hash based candidate graph prunning, sorts retained comparisons and applies Progressive Matching"

    def __init__(
        self,
        weighting_scheme: str = 'X2',
        similarity_function: str = 'dice',
        tokenizer: str = 'white_space_tokenizer',
        vectorizer : str = None,
        qgram : int = 1,
        similarity_threshold: float = 0.0,
        tokenizer_return_unique_values = True, # unique values or not
        attributes: any = None,
    ) -> None:

        super().__init__(similarity_function=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=vectorizer,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)
        self._weighting_scheme : str = weighting_scheme

class GlobalTopPM(HashBasedProgressiveMatching):
    """Applies Progressive CEP, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Global Top Progressive Matching"
    _method_info: str = "Applies Progressive CEP, sorts retained comparisons and applies Progressive Matching"

    def __init__(
        self,
        weighting_scheme: str = 'X2',
        similarity_function: str = 'dice',
        tokenizer: str = 'white_space_tokenizer',
        vectorizer : str = None,
        qgram : int = 1,
        similarity_threshold: float = 0.0,
        tokenizer_return_unique_values = True, # unique values or not
        attributes: any = None,
    ) -> None:

        super().__init__(weighting_scheme=weighting_scheme,
                        similarity_function=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=vectorizer,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)

    def _predict_raw_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
        self.pairs = Graph()
        pcep : ProgressiveCardinalityEdgePruning = ProgressiveCardinalityEdgePruning(self._weighting_scheme, self._budget)
        candidates : dict = pcep.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=None, emit_all_tps_stop=self._emit_all_tps_stop)
        self.blocks = candidates

        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, pcep.get_precalculated_weight(entity_id, candidate_id))
         
        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        self.pairs = [(edge[2]['weight'], edge[0], edge[1]) for edge in self.pairs.edges]
        return self.pairs


    def _predict_prunned_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
        self.pairs = Graph()
        pcep : ProgressiveCardinalityEdgePruning = ProgressiveCardinalityEdgePruning(self._weighting_scheme, self._budget)
        candidates : dict = pcep.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=self._comparison_cleaner, emit_all_tps_stop=self._emit_all_tps_stop)
        self.blocks = candidates

        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, self._comparison_cleaner.get_precalculated_weight(entity_id, candidate_id))

        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        self.pairs = [(edge[2]['weight'], edge[0], edge[1]) for edge in self.pairs.edges]
        return self.pairs

class LocalTopPM(HashBasedProgressiveMatching):
    """Applies Progressive CNP, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Global Top Progressive Matching"
    _method_info: str = "Applies Progressive CNP, sorts retained comparisons and applies Progressive Matching"

    def __init__(
        self,
        weighting_scheme: str = 'X2',
        similarity_function: str = 'dice',
        number_of_nearest_neighbors: int = 10,
        tokenizer: str = 'white_space_tokenizer',
        vectorizer : str = None,
        qgram : int = 1,
        similarity_threshold: float = 0.0,
        tokenizer_return_unique_values = True, # unique values or not
        attributes: any = None,
    ) -> None:

        super().__init__(weighting_scheme=weighting_scheme,
                        similarity_function=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=vectorizer,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)
        self._number_of_nearest_neighbors : int = number_of_nearest_neighbors
        
    def _predict_raw_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
        self.pairs = Graph()
        pcnp : ProgressiveCardinalityNodePruning = ProgressiveCardinalityNodePruning(weighting_scheme=self._weighting_scheme, budget=self._budget)
        candidates : dict = pcnp.process(blocks=blocks, data=self.data, number_of_nearest_neighbors=self._number_of_nearest_neighbors, tqdm_disable=True, cc=None, emit_all_tps_stop=self._emit_all_tps_stop)
        self.blocks = candidates
        
        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, pcnp.get_precalculated_weight(entity_id, candidate_id))

        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        self.pairs = [(edge[2]['weight'], edge[0], edge[1]) for edge in self.pairs.edges]
        return self.pairs

    def _predict_prunned_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
        self.pairs = Graph()
        pcnp : ProgressiveCardinalityNodePruning = ProgressiveCardinalityNodePruning(self._weighting_scheme, self._budget)
        candidates : dict = pcnp.process(blocks=blocks, data=self.data, number_of_nearest_neighbors=self._number_of_nearest_neighbors, tqdm_disable=True, cc=self._comparison_cleaner, emit_all_tps_stop=self._emit_all_tps_stop)
        self.blocks = candidates

        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, self._comparison_cleaner.get_precalculated_weight(entity_id, candidate_id))

        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True) 
        self.pairs = [(edge[2]['weight'], edge[0], edge[1]) for edge in self.pairs.edges]
        return self.pairs


class EmbeddingsNNBPM(BlockIndependentPM):
    """Utilizes/Creates entity embeddings, constructs neighborhoods via NN Approach and applies Progressive Matching
    """

    _method_name: str = "Embeddings NN Blocking Based Progressive Matching"
    _method_info: str = "Utilizes/Creates entity embeddings, constructs neighborhoods via NN Approach and applies Progressive Matching"

    def __init__(
            self,
            language_model: str = 'bert',
            number_of_nearest_neighbors: int = 10,
            similarity_search: str = 'faiss',
            vector_size: int = 300,
            num_of_clusters: int = 5,
            similarity_function: str = 'cosine',
            tokenizer: str = 'white_space_tokenizer',
            vectorizer : str = None,
            qgram : int = 1,
            similarity_threshold: float = 0.0,
            tokenizer_return_unique_values = True, # unique values or not
            attributes: any = None
        ) -> None:

        super().__init__(similarity_function=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=vectorizer,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)
        
        self._language_model : str = language_model
        self._number_of_nearest_neighbors : int = number_of_nearest_neighbors
        self._similarity_search : str = similarity_search
        self._vector_size : int = vector_size
        self._num_of_clusters : int = num_of_clusters

    def _top_pair_emission(self) -> None:
        """Applies global sorting to all entity pairs produced by NN,
           and returns pairs based on distance in ascending order
        """
        self.pairs = []
        n, k = self.neighbors.shape

        for i in range(n):
            entity_id = self.ennbb._si.d1_retained_ids[i] if self.data.is_dirty_er else self.ennbb._si.d2_retained_ids[i]
            for j in range(k):
                if self.neighbors[i][j] != -1:
                    candidate_id = self.ennbb._si.d1_retained_ids[self.neighbors[i][j]]
                    self.pairs.append((entity_id, candidate_id, self.scores[i][j]))

        self.pairs = [(x[2], x[0], x[1]) for x in self.pairs]

    def _dfs_pair_emission(self) -> None:
        """Sorts NN neighborhoods in ascending average distance from their query entity,
           iterate over each neighborhoods' entities in ascending distance to query entity 
        """
        self.pairs = []

        average_neighborhood_distances = np.mean(self.scores, axis=1)
        sorted_neighborhoods = sorted_enumerate(average_neighborhood_distances)

        for sorted_neighborhood in sorted_neighborhoods:
            neighbor_scores = self.scores[sorted_neighborhood]
            neighbors = self.neighbors[sorted_neighborhood]
            entity_id = self.ennbb._si.d1_retained_ids[sorted_neighborhood] \
            if self.data.is_dirty_er \
            else self.ennbb._si.d2_retained_ids[sorted_neighborhood]
            
            for neighbor_index, neighbor in enumerate(neighbors):
                if(neighbor != -1):
                    neighbor_id = self.ennbb._si.d1_retained_ids[neighbor]
                    self.pairs.append((entity_id, neighbor_id, neighbor_scores[neighbor_index]))

        self.pairs = [(x[2], x[0], x[1]) for x in self.pairs]
        
    def _hb_pair_emission(self) -> None:
        """Sorts NN neighborhoods in ascending average distance from their query entity,
           emits the top entity for each neighborhood, then iterates over the sorte neighborhoods,
           and emits the pairs in descending weight order
        """
        self.pairs = []
        _first_emissions = []
        _remaining_emissions = []

        average_neighborhood_distances = np.mean(self.scores, axis=1)
        sorted_neighborhoods = sorted_enumerate(average_neighborhood_distances)

        for sorted_neighborhood in sorted_neighborhoods:
            neighbor_scores = self.scores[sorted_neighborhood]
            neighbors = self.neighbors[sorted_neighborhood]
            entity_id = self.ennbb._si.d1_retained_ids[sorted_neighborhood] \
            if self.data.is_dirty_er \
            else self.ennbb._si.d2_retained_ids[sorted_neighborhood]
            for neighbor_index, neighbor in enumerate(neighbors):
                if(neighbor != -1):
                    neighbor_id = self.ennbb._si.d1_retained_ids[neighbor]              
                    _current_emissions = _remaining_emissions if neighbor_index else _first_emissions
                    _current_emissions.append((entity_id, neighbor_id, neighbor_scores[neighbor_index]))

        self.pairs = [(x[2], x[0], x[1]) for x in _first_emissions] + [(x[2], x[0], x[1]) for x in _remaining_emissions]
        
    def _bfs_pair_emission(self) -> None:
        """Sorts NN neighborhoods in ascending average distance from their query entity,
           and iteratively emits the current top pair per neighborhood 
        """
        self.pairs = []
        average_neighborhood_distances = np.mean(self.scores, axis=1)
        sorted_neighborhoods = sorted_enumerate(average_neighborhood_distances)

        _emissions_per_pair = self.neighbors.shape[1]
        for current_emission_per_pair in range(_emissions_per_pair):
            for sorted_neighborhood in sorted_neighborhoods:
                neighbor = self.neighbors[sorted_neighborhood][current_emission_per_pair]
                if(neighbor != -1):
                    neighbor_id = self.ennbb._si.d1_retained_ids[neighbor]
                    entity_id = self.ennbb._si.d1_retained_ids[sorted_neighborhood] \
                    if self.data.is_dirty_er \
                    else self.ennbb._si.d2_retained_ids[sorted_neighborhood]
                    self.pairs.append((entity_id, neighbor_id, self.scores[sorted_neighborhood][current_emission_per_pair]))
                    
        self.pairs = [(x[2], x[0], x[1]) for x in self.pairs]

    def _produce_pairs(self):
        """Calls pairs emission based on the requested approach
        Raises:
            AttributeError: Given emission technique hasn't been defined
        """
        # currently first phase algorithms are in charge of gathering the subset of the original dataset
        # that will be used to initialize the scheduler, we simply retrieve all the pairs and their scores
        self._top_pair_emission()

    def save_datasets_embeddings(self, vectors_1: np.array, vectors_2: np.array) -> None:
        """Stores the non-precalculated (not loaded) embeddings in corresponding dataset paths
        """

        if(self._d1_emb_load_path is None):
            try:
                print(f"Saving D1 Embeddings -> {self._d1_emb_save_path}")
                np.save(self._d1_emb_save_path, vectors_1) 
                pass
            except FileNotFoundError:
                print(f"Unable to save Embeddings -> {self._d1_emb_save_path}") 
                
        if(self._d2_emb_load_path is None):
            try:
                print(f"Saving D2 Embeddings -> {self._d2_emb_save_path}")
                np.save(self._d2_emb_save_path, vectors_2) 
                pass
            except FileNotFoundError:
                print(f"Unable to save Embeddings -> {self._d2_emb_save_path}") 
                 
    def retrieve_embeddings_file_paths(self):
        return(self.retrieve_dataset_embeddings_file_path(first_dataset=True), self.retrieve_dataset_embeddings_file_path(first_dataset=False))
            
    def retrieve_dataset_embeddings_file_path(self, first_dataset : bool = True) -> str:
        """Attemps to retrieve the precalculated embeddings of first/second dataset from disk for current experiment
        Returns:
            str: Precalculated Embeddings file path (None if doesn't exist)
        """
    
        _requested_indexing, _opposite_indexing = ("reverse", "inorder") if (self._indexing == "reverse") \
                                                else ("inorder", "reverse")
        _requested_dataset, _opposite_dataset = ("1","2") if(first_dataset) \
                                            else ("2", "1")
        
        _requested_indexing_file_name = '_'.join([_requested_indexing, self.dataset_identifier, self._language_model, _requested_dataset + ".npy"])
        _opposite_indexing_file_name = '_'.join([_opposite_indexing, self.dataset_identifier, self._language_model, _opposite_dataset + ".npy"])
        
        hidden_directory_path = os.path.join(os.getcwd(), ".embs")
        os.makedirs(hidden_directory_path, exist_ok=True)
        
        
        _available_file_path : str = None
        _requested_indexing_file_path = os.path.join(hidden_directory_path, _requested_indexing_file_name)
        _opposite_indexing_file_path = os.path.join(hidden_directory_path, _opposite_indexing_file_name)
        
        if(os.path.exists(_requested_indexing_file_path) and os.path.isfile(_requested_indexing_file_path)):
            _available_file_path = _requested_indexing_file_path
        elif(os.path.exists(_opposite_indexing_file_path) and os.path.isfile(_opposite_indexing_file_path)):
            _available_file_path = _opposite_indexing_file_path
            
        if(first_dataset):
            self._d1_emb_load_path = _available_file_path
            self._d1_emb_save_path = _requested_indexing_file_path
        else:
            self._d2_emb_load_path = _available_file_path
            self._d2_emb_save_path = _requested_indexing_file_path
            
        return _available_file_path 

    def _predict_raw_blocks(self, blocks: dict = None) -> List[Tuple[int, int]]:
        self.ennbb : EmbeddingsNNBlockBuilding = EmbeddingsNNBlockBuilding(self._language_model, self._similarity_search)
        
        
        load_path_d1, load_path_d2 = self.retrieve_embeddings_file_paths()
        
        self.final_blocks = self.ennbb.build_blocks(data=self.data,
                                                    vector_size=self._vector_size,
                                                    num_of_clusters=self._num_of_clusters,
                                                    top_k=self._number_of_nearest_neighbors,
                                                    return_vectors=False,
                                                    tqdm_disable=False,
                                                    save_embeddings=False,
                                                    load_embeddings_if_exist=True,
                                                    load_path_d1=load_path_d1,
                                                    load_path_d2=load_path_d2,
                                                    with_entity_matching=False,
                                                    input_cleaned_blocks=blocks,
                                                    similarity_distance=self.similarity_function)
        
        self.save_datasets_embeddings(vectors_1=self.ennbb.vectors_1, vectors_2=self.ennbb.vectors_2)
        self.scores = self.ennbb.distances
        self.neighbors = self.ennbb.neighbors
        self.final_vectors = (self.ennbb.vectors_1, self.ennbb.vectors_2)
        self._produce_pairs()
        return self.pairs

    def _predict_prunned_blocks(self, blocks: dict = None) -> List[Tuple[int, int]]:
        return self._predict_raw_blocks(blocks)
    
class SimilarityBasedProgressiveMatching(ProgressiveMatching):
    """Applies similarity based candidate graph prunning, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Similarity Based Progressive Matching"
    _method_info: str = "Applies similarity based candidate graph prunning, sorts retained comparisons and applies Progressive Matching"

    def __init__(
            self,
            weighting_scheme: str = 'ACF',
            window_size: int = 10,
            similarity_function: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            vectorizer : str = None,
            qgram : int = 1,
            similarity_threshold: float = 0.0,
            tokenizer_return_unique_values = True, # unique values or not
            attributes: any = None
        ) -> None:
        super().__init__(similarity_function=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=vectorizer,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)
        self._weighting_scheme : str = weighting_scheme
        self._window_size : int = window_size
        
class GlobalPSNM(SimilarityBasedProgressiveMatching):
    """Applies Global Progressive Sorted Neighborhood Matching
    """

    _method_name: str = "Global Progressive Sorted Neighborhood Matching"
    _method_info: str = "For each entity sorted accordingly to its block's tokens, " + \
                        "evaluates its neighborhood pairs defined within shifting windows of incremental size" + \
                        " and retains the globally best candidate pairs"

    def __init__(
            self,
            weighting_scheme: str = 'ACF',
            window_size: int = 10,
            similarity_function: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            vectorizer : str = None,
            qgram : int = 1,
            similarity_threshold: float = 0.0,
            tokenizer_return_unique_values = True, # unique values or not
            attributes: any = None
        ) -> None:

        super().__init__(weighting_scheme=weighting_scheme,
                        window_size=window_size,
                        similarity_function=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=vectorizer,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)

    def _predict_raw_blocks(self, blocks: dict) -> List[Tuple[float, int, int]]:
        gpsn : GlobalProgressiveSortedNeighborhood = GlobalProgressiveSortedNeighborhood(self._weighting_scheme, self._budget)
        self.pairs : List[Tuple[float, int, int]] = gpsn.process(blocks=blocks, data=self.data, window_size=self._window_size, tqdm_disable=True, emit_all_tps_stop=self._emit_all_tps_stop)
        return self.pairs

    def _predict_prunned_blocks(self, blocks: dict) -> List[Tuple[float, int, int]]:
        raise NotImplementedError("Sorter Neighborhood Algorithms don't support prunned blocks")
    
class LocalPSNM(SimilarityBasedProgressiveMatching):
    """Applies Local Progressive Sorted Neighborhood Matching
    """

    _method_name: str = "Global Progressive Sorted Neighborhood Matching"
    _method_info: str = "For each entity sorted accordingly to its block's tokens, " + \
                        "evaluates its neighborhood pairs defined within shifting windows of incremental size" + \
                        " and retains the globally best candidate pairs"

    def __init__(
            self,
            weighting_scheme: str = 'ACF',
            window_size: int = 10,
            similarity_function: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            vectorizer : str = None,
            qgram : int = 1,
            similarity_threshold: float = 0.0,
            tokenizer_return_unique_values = True, # unique values or not
            attributes: any = None
        ) -> None:

        super().__init__(weighting_scheme=weighting_scheme,
                        window_size=window_size,
                        similarity_function=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=vectorizer,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)

    def _predict_raw_blocks(self, blocks: dict) -> List[Tuple[float, int, int]]:
        lpsn : LocalProgressiveSortedNeighborhood = LocalProgressiveSortedNeighborhood(self._weighting_scheme, self._budget)
        self.pairs : List[Tuple[float, int, int]] = lpsn.process(blocks=blocks, data=self.data, window_size=self._window_size, tqdm_disable=True, emit_all_tps_stop=self._emit_all_tps_stop)
        return self.pairs

    def _predict_prunned_blocks(self, blocks: dict) -> List[Tuple[float, int, int]]:
        raise NotImplementedError("Sorter Neighborhood Algorithms don't support prunned blocks " + \
                                "(pre comparison-cleaning entities per block distribution required")
class RandomPM(ProgressiveMatching):
    """Picks a number of random comparisons equal to the available budget
    """

    _method_name: str = "Random Progressive Matching"
    _method_info: str = "Picks a number of random comparisons equal to the available budget"

    def __init__(
            self,
            similarity_function: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.0,
            qgram: int = 2, # for jaccard
            tokenizer_return_unique_values = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(similarity_function, tokenizer, similarity_threshold, qgram, tokenizer_return_unique_values, attributes, delim_set, padding, prefix_pad, suffix_pad)

    def _predict_raw_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
        cp : ComparisonPropagation = ComparisonPropagation()
        cleaned_blocks = cp.process(blocks=blocks, data=self.data, tqdm_disable=True)
        self._predict_prunned_blocks(cleaned_blocks)

    def _predict_prunned_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
        _all_pairs = [(id1, id2) for id1 in blocks for id2 in blocks[id1]]
        _total_pairs = len(_all_pairs)
        random_pairs = sample(_all_pairs, self._budget) if self._budget <= _total_pairs and not self._emit_all_tps_stop else _all_pairs
        self.pairs.add_edges_from(random_pairs)
        
class PESM(HashBasedProgressiveMatching):
    """Applies Progressive Entity Scheduling Matching
    """

    _method_name: str = "Progressive Entity Scheduling Matching"
    _method_info: str = "Applies Progressive Entity Scheduling - Sorts entities in descending order of their average weight, " + \
                        "emits the top pair per entity. Finally, traverses the sorted " + \
                        "entities and emits their comparisons in descending weight order " + \
                        "within specified budget."
                        
    def __init__(
            self,
            weighting_scheme: str = 'CBS',
            similarity_function: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            vectorizer : str = None,
            qgram : int = 1,
            similarity_threshold: float = 0.0,
            tokenizer_return_unique_values = True, # unique values or not
            attributes: any = None,
        ) -> None:

        super().__init__(weighting_scheme=weighting_scheme,
                        similarity_function=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=vectorizer,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)

    def _predict_raw_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
        
        pes : ProgressiveEntityScheduling = ProgressiveEntityScheduling(self._weighting_scheme, self._budget)
        self.pairs = pes.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=None, method=self._algorithm, emit_all_tps_stop=self._emit_all_tps_stop)
        return self.pairs
    
    def _predict_prunned_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
        return self._predict_raw_blocks(blocks)
        # raise NotImplementedError("Sorter Neighborhood Algorithms doesn't support prunned blocks (lack of precalculated weights)")
    
# class WhooshPM(BlockIndependentPM):
#     """Applies progressive index based matching using whoosh library 
#     """

#     _method_name: str = "Whoosh Progressive Matching"
#     _method_info: str = "Applies Whoosh Progressive Matching - Indexes the entities of the second dataset, " + \
#                         "stores their specified attributes, " + \
#                         "defines a query for each entity of the first dataset, " + \
#                         "and retrieves its pair candidates from the indexer within specified budget"

#     def __init__(
#             self,
#             similarity_function: str = 'WH-TF-IDF',
#             number_of_nearest_neighbors: int = 10,
#             tokenizer: str = 'white_space_tokenizer',
#             similarity_threshold: float = 0.0,
#             qgram: int = 2, # for jaccard
#             tokenizer_return_unique_values = True, # unique values or not
#             attributes: any = None,
#             delim_set: list = None, # DelimiterTokenizer
#             padding: bool = True, # QgramTokenizer
#             prefix_pad: str = '#', # QgramTokenizer (if padding=True)
#             suffix_pad: str = '$' # QgramTokenizer (if padding=True)
#         ) -> None:
#         # budget set to float('inf') implies unlimited budget
#         super().__init__(similarity_function, tokenizer, similarity_threshold, qgram, tokenizer_return_unique_values, attributes, delim_set, padding, prefix_pad, suffix_pad)
#         self._number_of_nearest_neighbors : int = number_of_nearest_neighbors
        
#     def _set_whoosh_datasets(self) -> None:
#         """Saves the rows of both datasets corresponding to the indices of the entities that have been retained after comparison cleaning
#         """
        
#         self._whoosh_d1 = self.data.dataset_1[self.attributes + [self.data.id_column_name_1]] if self.attributes else self.data.dataset_1
#         self._whoosh_d1 = self._whoosh_d1[self._whoosh_d1[self.data.id_column_name_1].isin(self._whoosh_d1_retained_index)]
#         if(not self.data.is_dirty_er):  
#             self._whoosh_d2 = self.data.dataset_2[self.attributes + [self.data.id_column_name_2]] if self.attributes else self.data.dataset_2
#             self._whoosh_d2 = self._whoosh_d2[self._whoosh_d2[self.data.id_column_name_2].isin(self._whoosh_d2_retained_index)]
        

#     def _set_retained_entries(self) -> None:
#         """Saves the indices of entities of both datasets that have been retained after comparison cleaning
#         """
#         self._whoosh_d1_retained_index = pd.Index([self.data._gt_to_ids_reversed_1[id] 
#         for id in self._si.d1_retained_ids])
        
#         if(not self.data.is_dirty_er):
#             self._whoosh_d2_retained_index = pd.Index([self.data._gt_to_ids_reversed_2[id] 
#         for id in self._si.d2_retained_ids])
    
    
#     def _initialize_index_path(self):
#         """Creates index directory if non-existent, constructs the absolute path to the current whoosh index
#         """
#         global INDEXER_DIR
#         INDEXER_DIR = os.path.abspath(INDEXER_DIR)  
#         _d1_name = self.data.dataset_name_1 if self.data.dataset_name_1 is not None else 'd3'    
#         self._index_path = os.path.join(INDEXER_DIR, _d1_name if self.data.is_dirty_er else (_d1_name + (self.data.dataset_name_2 if self.data.dataset_name_2 is not None else 'd4')))
#         if not os.path.exists(self._index_path):
#             print('Created index directory at: ' + self._index_path)
#             os.makedirs(self._index_path, exist_ok=True)
        
    
#     def _create_index(self):
#         """Defines the schema [ID, CONTENT], creates the index in the defined path 
#            and populates it with all the entities of the target dataset (first - Dirty ER, second - Clean ER)
#         """
#         self._schema = Schema(ID=ID(stored=True), content=TEXT(stored=True))
#         self._index = create_in(self._index_path, self._schema)
#         writer = self._index.writer()
        
#         _target_dataset = self._whoosh_d1 if self.data.is_dirty_er else self._whoosh_d2
#         _id_column_name = self.data.id_column_name_1 if self.data.is_dirty_er else self.data.id_column_name_2
        
#         for _, entity in _target_dataset.iterrows():
#             entity_values = [str(entity[column]) for column in _target_dataset.columns if column != _id_column_name]
#             writer.add_document(ID=entity[_id_column_name], content=' '.join(entity_values))
#         writer.commit()
    
#     def _populate_whoosh_dataset(self) -> None:
#         """For each retained entity in the first dataset, construct a query with its text content,
#            parses it to the indexers, retrieves best candidates and stores them in entity's neighborhood.
#            Populates a list with all the retrieved pairs.
#         """
#         # None value for budget implies unlimited budget in whoosh 
#         _query_budget = self._number_of_nearest_neighbors
        
#         if(self.similarity_function not in whoosh_similarity_function):
#             print(f'{self.similarity_function} Similarity Function is Undefined')
#             self.similarity_function = 'Frequency'
#         print(f'Applying {self.similarity_function} Similarity Function')
#         _scorer = whoosh_similarity_function[self.similarity_function]
        
#         with self._index.searcher(weighting=_scorer) as searcher:
#             self._parser = qparser.QueryParser('content', schema=self._index.schema, group=qparser.OrGroup)
#             for _, entity in self._whoosh_d1.iterrows():
#                 entity_values = [str(entity[column]) for column in self._whoosh_d1.columns if column != self.data.id_column_name_1]
#                 entity_string = ' '.join(entity_values)
#                 entity_id = entity[self.data.id_column_name_1]
#                 entity_query = self._parser.parse(entity_string)
#                 query_results = searcher.search(entity_query, limit = _query_budget)
                
#                 for neighbor in query_results:
#                     _score = neighbor.score
#                     _neighbor_id = neighbor['ID']
#                     self.pairs.append((_score, self.data._ids_mapping_1[entity], self.data._ids_mapping_2[_neighbor_id]))
                       
#     def _predict_raw_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
#         self._start_time = time()
#         self._si = SubsetIndexer(blocks=blocks, data=self.data, subset=False)
#         self._set_retained_entries()
#         self._set_whoosh_datasets()
#         self._initialize_index_path()
#         self._create_index()
#         self.pairs : List[Tuple[float, int, int]] = []
#         self._budget = float('inf') if self._emit_all_tps_stop else self._budget
#         self._populate_whoosh_dataset()
#         self.execution_time = time() - self._start_time
#         return self.pairs
        
#     def _predict_prunned_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
#         self._predict_raw_blocks(blocks)  

class TopKJoinPM(ProgressiveMatching):
    """Applies index based matching for ES, emits candidate pairs using defined budget/emission technique
    """

    _method_name: str = "Top-K Join Progressive Matching"
    _method_info: str = "Applies index based matching for ES, emits candidate pairs using defined budget/emission technique"
    def __init__(
            self,
            similarity_function: str = 'dice',
            number_of_nearest_neighbors : int = 10,
            tokenizer: str = None,
            weighting_scheme : str = None,
            qgram : int = 1,
            similarity_threshold: float = 0.0,
            tokenizer_return_unique_values = True, # unique values or not
            attributes: any = None,
        ) -> None:

        super().__init__(similarity_function=similarity_function,
                        tokenizer=tokenizer,
                        vectorizer=weighting_scheme,
                        qgram=qgram,
                        similarity_threshold=similarity_threshold,
                        tokenizer_return_unique_values=tokenizer_return_unique_values,
                        attributes=attributes)
        
        self.similarity_function : str = similarity_function
        self.number_of_nearest_neighbors : int = number_of_nearest_neighbors
        self.weighting_scheme : str = weighting_scheme
        self.qgram : int = qgram 
        
    def _predict_raw_blocks(self, blocks: dict, load_neighborhoods : bool = True) -> List[Tuple[int, int]]:
        
        _store_neighborhoods : bool = load_neighborhoods
        _loaded_neighborhoods : dict[List[Tuple[float, int]]]
        
        if(load_neighborhoods):
            print("Neighborhood Retrieval Enabled...")
            _loaded_neighborhoods = self.retrieve_neighborhoods_from_disk()
        else:
            print("Neighborhood Retrieval Disabled...")
            _loaded_neighborhoods = None
        
        if(_loaded_neighborhoods is None):
            ptkj : PETopKJoin = PETopKJoin(K=self.number_of_nearest_neighbors,
                                            metric=self.similarity_function,
                                            tokenization=self.tokenizer,
                                            qgrams=self.qgram)

            _pet_vectorizer = self.initialize_vectorizer() if (self.weighting_scheme is not None) else None
            self.pairs = ptkj.fit(data=self.data,
                                reverse_order=True,
                                attributes_1=self.data.attributes_1,
                                attributes_2=self.data.attributes_2,
                                vectorizer=_pet_vectorizer,
                                store_neighborhoods=_store_neighborhoods)
            
            if(_store_neighborhoods): 
                self.pairs = self.neighborhoods_to_pairs(neighborhoods=ptkj.neighborhoods, strict_top_k=True)
                self.neighborhoods_to_json(neighborhoods=ptkj.neighborhoods)
            else:
                self.pairs = [(edge[2]['weight'], edge[0], edge[1]) for edge in self.pairs.edges(data=True)]
        else:
            self.pairs = self.neighborhoods_to_pairs(neighborhoods=_loaded_neighborhoods, strict_top_k=True) 
            
        return self.pairs
    
    def _predict_prunned_blocks(self, blocks: dict) -> List[Tuple[int, int]]:
        raise NotImplementedError("Progressive TopKJoin PM for prunned blocks - Not implemented yet!")
        
    
    def neighborhoods_to_pairs(self, neighborhoods : dict[List[Tuple[float, int]]], strict_top_k : bool = False) -> List[Tuple[float, int, int]]:
        previous_weight = None
        _pairs : List[Tuple[float, int, int]] = []
        for d1_id, d2_ids in neighborhoods.items():
            distinct_weights = 0
            _d1_id = int(d1_id)
            for current_weight, d2_id in d2_ids:
                if(strict_top_k or current_weight != previous_weight):
                    previous_weight = current_weight
                    distinct_weights += 1
                if distinct_weights <= self.number_of_nearest_neighbors:
                    _pairs.append((current_weight, d2_id, _d1_id))
                else:
                    break  
        return _pairs
    
    def neighborhoods_to_json(self, neighborhoods : dict[List[Tuple[float, int]]]) -> None:
        """Stores the neighborhood in the corresponding experiment's neighborhoods json file within the hidden .ngbs directory
        Args:
            neighborhoods (dict[List[Tuple[float, int]]]): Neighborhoods of indexed entities of current experiment, dictionary in the form
                                                           [indexed entity id] -> [sorted target dataset neighbors in descending similarity order]
        """
        
        _json_file_name = '_'.join(self._requested_file_components)
        
        neighborhoods_directory_path = os.path.join(os.getcwd(), ".ngbs")
        os.makedirs(neighborhoods_directory_path, exist_ok=True)
        
        _json_store_path = os.path.join(neighborhoods_directory_path, _json_file_name)
        print(f"Storing Neighborhood Json in -> {_json_store_path}")
        with open(_json_store_path, 'w') as json_file:
            json.dump(neighborhoods, json_file, indent=4)
    
    def matching_file_components(self, 
                                 source_components : List[str], 
                                 target_components : List[str], 
                                 variable_component_index : int = 6) -> bool:
        """Takes as inputs lists containing the component of the source and target file name (strings connecte by underscore).
           Checks whether those components match (files are equivalent). Variable component (number of nearest neighbor) must be less or equal
           to the target component.
        Args:
            source_components (List[str]): Components (substrings seperated by underscore) that constitute source file name
            target_components (List[str]): Components (substrings seperated by underscore) that constitute target file name
            variable_component_index (int, optional): Index in file name's components list where the variable component is placed (number of nearest neighbors)
        Returns:
            bool: Source and target file name components are equivalent (target file can be loaded for source file request)
        """
        number_pattern = r"[-+]?\d*\.\d+|\d+"
        zipped_components = list(zip(source_components, target_components))
        matching_components = True
        
        for index, components in enumerate(zipped_components):
            source_component, target_component = components
            
            if(index == variable_component_index):
                source_nns = int((re.findall(number_pattern, source_component))[0])
                target_nns = int((re.findall(number_pattern, target_component))[0])
                if(source_nns > target_nns):
                    matching_components = False
                    break
            else:
                if(source_component != target_component):
                    matching_components = False
                    break   
        return matching_components
    
    def retrieve_neighborhoods_from_disk(self) -> dict[List[Tuple[float, int]]]:
        """Attemps to retrieve a precalculated neighborhoods for indexed entities of the current experiment
        Returns:
            dict[List[Tuple[float, int]]]: Dictionary of neighborhoods for each indexed entity containing a sorted list of neighbords in descending similarity order
        """
        self._requested_file_components = [self._indexing,
                                      self.dataset_identifier,
                                      self.weighting_scheme,
                                      self.tokenizer.split('_')[0],
                                      self.similarity_function,
                                      "q" + str(self.qgram),
                                      "n" + str(self.number_of_nearest_neighbors) + ".json"]
        
        _neighbors_count_index : int = len(self._requested_file_components) - 1
        neighborhoods_directory_path : str = os.path.join(os.getcwd(), ".ngbs")
        _matching_neighborhood_file_name : str = None
        _matching_neighborhood : dict[List[Tuple[float, int]]] = None 
        
        os.makedirs(neighborhoods_directory_path, exist_ok=True)
        print(f"Searching for matching neighborhood file in -> {neighborhoods_directory_path}")
        
        if os.path.isdir(neighborhoods_directory_path):
            neighborhoods_file_names = os.listdir(neighborhoods_directory_path)
    
            for neighborhood_file_name in neighborhoods_file_names:
                _neighborhood_file_components = neighborhood_file_name.split('_')
                if(self.matching_file_components(source_components=self._requested_file_components,
                                                target_components=_neighborhood_file_components,
                                                variable_component_index=_neighbors_count_index)):
                    _matching_neighborhood_file_name = neighborhood_file_name
                    break
        
        if(_matching_neighborhood_file_name is not None):
            _matching_neighborhood_file_path = os.path.join(neighborhoods_directory_path, _matching_neighborhood_file_name)
            if(os.path.exists(_matching_neighborhood_file_path) and os.path.isfile(_matching_neighborhood_file_path)):
                with open(_matching_neighborhood_file_path, 'r') as neighborhood_file:
                    _matching_neighborhood = json.load(neighborhood_file)
                print(f"Retrieved matching neighborhood from -> {_matching_neighborhood_file_path}!")
        else:
            print(f"Matching Neighborhood File not found - Executing Joins Algorithm!")
            
        return _matching_neighborhood
                    
    def initialize_vectorizer(self) -> FrequencyEvaluator:
        self.vectorizer : FrequencyEvaluator = FrequencyEvaluator(vectorizer=self.weighting_scheme,
                                                                  tokenizer=self.tokenizer,
                                                                  qgram=self.qgram)
        
        d1 = self.data.dataset_1[self.data.attributes_1] if self.data.attributes_1 is not None else self.data.dataset_1
        self._entities_d1 = d1 \
                    .apply(" ".join, axis=1) \
                    .apply(lambda x: x.lower()) \
                    .values.tolist()
        d2 = self.data.dataset_2[self.data.attributes_2] if self.data.attributes_2 is not None else self.data.dataset_2
        self._entities_d2 = d2 \
                    .apply(" ".join, axis=1) \
                    .apply(lambda x: x.lower()) \
                    .values.tolist() if not self.data.is_dirty_er else None         
        self.vectorizer.fit(metric=self.similarity_function,
                            dataset_identifier=self.dataset_identifier,
                            indexing=self._indexing,
                            d1_entities=self._entities_d1,
                            d2_entities=self._entities_d2)
        return self.vectorizer

class_references = {
    'GlobalTopPM' : GlobalTopPM,
    'LocalTopPM' : LocalTopPM,
    'GlobalPSNM' : GlobalPSNM,
    'LocalPSNM' : LocalPSNM,
    'PESM' : PESM,
    'EmbeddingsNNBPM' : EmbeddingsNNBPM,
    'TopKJoinPM' : TopKJoinPM
} 

