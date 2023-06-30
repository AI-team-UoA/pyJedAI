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
from .vector_based_blocking import EmbeddingsNNBlockBuilding
from sklearn.metrics.pairwise import (
    cosine_similarity
)
from networkx import Graph
from py_stringmatching.similarity_measure.affine import Affine
from py_stringmatching.similarity_measure.bag_distance import BagDistance
from py_stringmatching.similarity_measure.cosine import Cosine
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.similarity_measure.editex import Editex
from py_stringmatching.similarity_measure.generalized_jaccard import \
    GeneralizedJaccard
from py_stringmatching.similarity_measure.hamming_distance import \
    HammingDistance
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.jaro import Jaro
from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
from py_stringmatching.similarity_measure.needleman_wunsch import \
    NeedlemanWunsch
from py_stringmatching.similarity_measure.overlap_coefficient import \
    OverlapCoefficient
from py_stringmatching.similarity_measure.partial_ratio import PartialRatio
from py_stringmatching.similarity_measure.token_sort import TokenSort
from py_stringmatching.similarity_measure.partial_token_sort import \
    PartialTokenSort
from py_stringmatching.similarity_measure.ratio import Ratio
from py_stringmatching.similarity_measure.smith_waterman import SmithWaterman
from py_stringmatching.similarity_measure.soundex import Soundex
from py_stringmatching.similarity_measure.tfidf import TfIdf
from py_stringmatching.similarity_measure.tversky_index import TverskyIndex
from py_stringmatching.tokenizer.alphabetic_tokenizer import \
    AlphabeticTokenizer
from py_stringmatching.tokenizer.alphanumeric_tokenizer import \
    AlphanumericTokenizer
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.whitespace_tokenizer import \
    WhitespaceTokenizer
from tqdm.autonotebook import tqdm

from .evaluation import Evaluation
from .datamodel import Data, PYJEDAIFeature
from .matching import EntityMatching
from .comparison_cleaning import AbstractMetablocking
from queue import PriorityQueue
from random import sample
from .utils import sorted_enumerate, canonical_swap
from abc import abstractmethod
from typing import Tuple, List
from .utils import SubsetIndexer, WhooshDataset, WhooshNeighborhood, is_infinite, PredictionData
import pandas as pd
import os
from whoosh.fields import TEXT, Schema, ID
from whoosh.index import create_in
from whoosh import qparser
from whoosh.scoring import TF_IDF, Frequency, PL2, BM25F


# Directory where the whoosh index is stored
INDEXER_DIR='.indexer'

# Package import from https://anhaidgroup.github.io/py_stringmatching/v0.4.2/index.html

available_tokenizers = [
    'white_space_tokenizer', 'qgram_tokenizer', 'delimiter_tokenizer',
    'alphabetic_tokenizer', 'alphanumeric_tokenizer'
]

metrics_mapping = {
    'levenshtein' : Levenshtein(),
    'edit_distance': Levenshtein(),
    'jaro_winkler' : JaroWinkler(),
    'bag_distance' : BagDistance(),
    'editex' : Editex(),
    'cosine' : Cosine(),
    'jaro' : Jaro(),
    'soundex' : Soundex(),
    'tfidf' : TfIdf(),
    'tversky_index':TverskyIndex(),
    'ratio' : Ratio(),
    'partial_token_sort' : PartialTokenSort(),
    'partial_ratio' : PartialRatio(),
    'hamming_distance' : HammingDistance(),
    'jaccard' : Jaccard(),
    'generalized_jaccard' : GeneralizedJaccard(),
    'dice': Dice(),
    'overlap_coefficient' : OverlapCoefficient(),
    'token_sort': TokenSort(),
    'cosine_vector_similarity': cosine_similarity,
    'TF-IDF' : TF_IDF(),
    'Frequency' : Frequency(),
    'PL2' : PL2(),
    'BM25F' : BM25F()
}

whoosh_similarity_function = {
    'TF-IDF' : TF_IDF(),
    'Frequency' : Frequency(),
    'PL2' : PL2(),
    'BM25F' : BM25F()
}

string_metrics = [
    'bag_distance', 'editex', 'hamming_distance', 'jaro', 'jaro_winkler', 'levenshtein',
    'edit_distance', 'partial_ratio', 'partial_token_sort', 'ratio', 'soundex', 'token_sort'
]

set_metrics = [
    'cosine', 'dice', 'generalized_jaccard', 'jaccard', 'overlap_coefficient', 'tversky_index'
]

bag_metrics = [
    'tfidf'
]

index_metrics = [
    'TF-IDF', 'Frequency', 'PL2', 'BM25F'
] 

vector_metrics = [
    'cosine_vector_similarity'
]

available_metrics = string_metrics + set_metrics + bag_metrics + vector_metrics + index_metrics

class ProgressiveMatching(EntityMatching):
    """Applies the matching process to a subset of available pairs progressively 
    """

    _method_name: str = "Progressive Matching"
    _method_info: str = "Applies the matching process to a subset of available pairs progressively "

    def __init__(
            self,
            budget: int = 0,
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)
        self._budget : int = budget

    def predict(self,
            blocks: dict,
            data: Data,
            comparison_cleaner: AbstractMetablocking = None,
            tqdm_disable: bool = False,
            method : str = 'HB',
            emit_all_tps_stop : bool = False) -> Graph:
        """Main method of  progressive entity matching. Inputs a set of blocks and outputs a graph \
            that contains of the entity ids (nodes) and the similarity scores between them (edges).
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
        self._comparison_cleaner: AbstractMetablocking = comparison_cleaner
        self._method = method
        self._emit_all_tps_stop = emit_all_tps_stop
        self.true_pair_checked = None 
        self._prediction_data : PredictionData = None

        if not blocks:
            raise ValueError("Empty blocks structure")
        self.data = data
        self.pairs = Graph()
        all_blocks = list(blocks.values())
        self._progress_bar = tqdm(total=len(blocks),
                                desc=self._method_name+" ("+self.metric+")",
                                disable=self.tqdm_disable)
        if 'Block' in str(type(all_blocks[0])):
            self._predict_raw_blocks(blocks)
        elif isinstance(all_blocks[0], set):
            if(self._comparison_cleaner == None):
                raise AttributeError("No precalculated weights were given from the CC step") 
            self._predict_prunned_blocks(blocks)
        else:
            raise AttributeError("Wrong type of Blocks")
        self.execution_time = time() - start_time
        self._progress_bar.close()

        return self.pairs

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
        
    def get_true_pair_checked(self):
        if(self.true_pair_checked is None):
            raise AttributeError("True positive pairs not defined in specified workflow.")
        else: return self.true_pair_checked   
        
        
    @abstractmethod
    def extract_tps_checked(self, **kwargs) -> dict:
        """Constructs a dictionary of the form [true positive pair] -> emitted status,
           containing all the true positive pairs that are emittable from the current subset of the dataset

        Returns:
            dict: Dictionary that shows whether a TP pair (key) has been emitted (value)
        """
        pass
    
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
        

class HashBasedProgressiveMatching(ProgressiveMatching):
    """Applies hash based candidate graph prunning, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Hash Based Progressive Matching"
    _method_info: str = "Applies hash based candidate graph prunning, sorts retained comparisons and applies Progressive Matching"

    def __init__(
            self,
            budget: int = 0,
            w_scheme: str = 'X2',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)
        self._w_scheme : str = w_scheme
        
    def extract_tps_checked(self, **kwargs) -> dict:
        _tps_checked = dict()
        for entity, neighbors in self.blocks.items():
            for neighbor in neighbors:
                entity_id = self.data._gt_to_ids_reversed_1[entity] if entity < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[entity]
                neighbor_id = self.data._gt_to_ids_reversed_1[neighbor] if neighbor < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[neighbor]
                _d1_entity, _d2_entity = (entity_id, neighbor_id) if entity < self.data.dataset_limit else (neighbor_id, entity_id)
            
                if _d2_entity in self.data.pairs_of[_d1_entity]:
                    _tps_checked[canonical_swap(_d1_entity, _d2_entity)] = False
        return _tps_checked
        

class GlobalTopPM(HashBasedProgressiveMatching):
    """Applies Progressive CEP, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Global Top Progressive Matching"
    _method_info: str = "Applies Progressive CEP, sorts retained comparisons and applies Progressive Matching"

    def __init__(
            self,
            budget: int = 0,
            w_scheme: str = 'X2',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, w_scheme, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)

    def _predict_raw_blocks(self, blocks: dict) -> None:
        pcep : ProgressiveCardinalityEdgePruning = ProgressiveCardinalityEdgePruning(self._w_scheme, self._budget)
        candidates : dict = pcep.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=None, emit_all_tps_stop=self._emit_all_tps_stop)
        self.blocks = candidates
        if(self._emit_all_tps_stop): self.true_pair_checked = self.extract_tps_checked()

        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, pcep.get_precalculated_weight(entity_id, candidate_id))
         
        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True) 
        return self.pairs.edges


    def _predict_prunned_blocks(self, blocks: dict) -> None:
        pcep : ProgressiveCardinalityEdgePruning = ProgressiveCardinalityEdgePruning(self._w_scheme, self._budget)
        candidates : dict = pcep.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=self._comparison_cleaner, emit_all_tps_stop=self._emit_all_tps_stop)
        self.blocks = candidates
        if(self._emit_all_tps_stop): self.true_pair_checked = self.extract_tps_checked()

        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, self._comparison_cleaner.get_precalculated_weight(entity_id, candidate_id))

        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        return self.pairs.edges

class LocalTopPM(HashBasedProgressiveMatching):
    """Applies Progressive CNP, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Global Top Progressive Matching"
    _method_info: str = "Applies Progressive CNP, sorts retained comparisons and applies Progressive Matching"

    def __init__(
            self,
            budget: int = 0,
            w_scheme: str = 'X2',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, w_scheme, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)


    def _predict_raw_blocks(self, blocks: dict) -> None:
        pcnp : ProgressiveCardinalityNodePruning = ProgressiveCardinalityNodePruning(self._w_scheme, self._budget)
        candidates : dict = pcnp.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=None, emit_all_tps_stop=self._emit_all_tps_stop)
        self.blocks = candidates
        if(self._emit_all_tps_stop): self.true_pair_checked = self.extract_tps_checked()
        
        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, pcnp.get_precalculated_weight(entity_id, candidate_id))

        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True) 
        return self.pairs.edges

    def _predict_prunned_blocks(self, blocks: dict) -> None:

        pcnp : ProgressiveCardinalityNodePruning = ProgressiveCardinalityNodePruning(self._w_scheme, self._budget)
        candidates : dict = pcnp.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=self._comparison_cleaner, emit_all_tps_stop=self._emit_all_tps_stop)
        self.blocks = candidates
        if(self._emit_all_tps_stop): self.true_pair_checked = self.extract_tps_checked()

        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, self._comparison_cleaner.get_precalculated_weight(entity_id, candidate_id))

        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True) 
        return self.pairs.edges


class EmbeddingsNNBPM(ProgressiveMatching):
    """Utilizes/Creates entity embeddings, constructs neighborhoods via NN Approach and applies Progressive Matching
    """

    _method_name: str = "Embeddings NN Blocking Based Progressive Matching"
    _method_info: str = "Utilizes/Creates entity embeddings, constructs neighborhoods via NN Approach and applies Progressive Matching"

    def __init__(
            self,
            budget: int = 0,
            vectorizer: str = 'bert',
            similarity_search: str = 'faiss',
            vector_size: int = 200,
            num_of_clusters: int = 5,
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)
        self._vectorizer = vectorizer
        self._similarity_search = similarity_search
        self._vector_size = vector_size
        self._num_of_clusters = num_of_clusters
        
        
    def predict(self,
        data: Data,
        blocks: dict = None,
        comparison_cleaner: AbstractMetablocking = None,
        tqdm_disable: bool = False,
        method : str = 'HB',
        emit_all_tps_stop : bool = False) -> Graph:
        """Main method of  progressive entity matching. Inputs a set of blocks and outputs a graph \
            that contains of the entity ids (nodes) and the similarity scores between them (edges).
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
        self._comparison_cleaner: AbstractMetablocking = comparison_cleaner
        self._method = method
        self._emit_all_tps_stop = emit_all_tps_stop
        self.true_pair_checked = None 
        self._prediction_data : PredictionData = None
        self.data = data
        self.pairs = Graph()
        
        if blocks is None:
        # applying the process to the whole dataset
            self._predict_raw_blocks(blocks)
        else:
            all_blocks = list(blocks.values())
            self._progress_bar = tqdm(total=len(blocks),
                                    desc=self._method_name+" ("+self.metric+")",
                                    disable=self.tqdm_disable)
            if 'Block' in str(type(all_blocks[0])):
                self._predict_raw_blocks(blocks)
            elif isinstance(all_blocks[0], set):
                if(self._comparison_cleaner == None):
                    raise AttributeError("No precalculated weights were given from the CC step") 
                self._predict_prunned_blocks(blocks)
            else:
                raise AttributeError("Wrong type of Blocks")
            self._progress_bar.close()
            
        self.execution_time = time() - start_time
        return self.pairs

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

        self.pairs = sorted(self.pairs, key=lambda x: x[2], reverse=True)
        self.pairs = [(x[0], x[1]) for x in self.pairs]

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

        self.pairs = [(x[0], x[1]) for x in self.pairs]
        
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

        self.pairs = [(x[0], x[1]) for x in _first_emissions] + [(x[0], x[1]) for x in _remaining_emissions]
        
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
                    
        self.pairs = [(x[0], x[1]) for x in self.pairs]

    def _produce_pairs(self):
        """Calls pairs emission based on the requested approach
        Raises:
            AttributeError: Given emission technique hasn't been defined
        """
        if(self._method == 'DFS'):
            self._dfs_pair_emission()
        elif(self._method == 'HB'):
            self._hb_pair_emission()
        elif(self._method == 'BFS'):
            self._bfs_pair_emission()
        elif(self._method == 'TOP'):
            self._top_pair_emission()
        else:
            raise AttributeError(self._method + ' emission technique is undefined!')

    def _predict_raw_blocks(self, blocks: dict = None) -> None:
        self.ennbb : EmbeddingsNNBlockBuilding = EmbeddingsNNBlockBuilding(self._vectorizer, self._similarity_search)
        self.final_blocks = self.ennbb.build_blocks(data = self.data,
                     num_of_clusters = self._num_of_clusters,
                     top_k = int(max(1, int(self._budget / self.data.num_of_entities) + (self._budget % self.data.num_of_entities > 0)))
                     if not self._emit_all_tps_stop else self._budget,
                     return_vectors = False,
                     tqdm_disable = False,
                     save_embeddings = True,
                     load_embeddings_if_exist = True,
                     with_entity_matching = False,
                     input_cleaned_blocks = blocks)

        self.scores = self.ennbb.distances
        self.neighbors = self.ennbb.neighbors
        self.final_vectors = (self.ennbb.vectors_1, self.ennbb.vectors_2)

        self._produce_pairs()
        if(self._emit_all_tps_stop):
            self.true_pair_checked = self.extract_tps_checked()
        return self.pairs

    def _predict_prunned_blocks(self, blocks: dict = None) -> None:
        return self._predict_raw_blocks(blocks)
    
    def extract_tps_checked(self, **kwargs) -> dict:
        _tps_checked = dict()
        _neighbors = self.neighbors
        
        for row in range(_neighbors.shape[0]):
            entity = self.ennbb._si.d1_retained_ids[row] \
                    if self.data.is_dirty_er \
                    else self.ennbb._si.d2_retained_ids[row]
            entity_id = self.data._gt_to_ids_reversed_1[entity] if entity < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[entity]
            for column in range(_neighbors.shape[1]):
                if(_neighbors[row][column] != -1):
                    neighbor = self.ennbb._si.d1_retained_ids[_neighbors[row][column]]
                    neighbor_id = self.data._gt_to_ids_reversed_1[neighbor] if neighbor < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[neighbor]
                    _d1_entity, _d2_entity = (entity_id, neighbor_id) if entity < self.data.dataset_limit else (neighbor_id, entity_id)
                if _d2_entity in self.data.pairs_of[_d1_entity]:
                    _tps_checked[canonical_swap(_d1_entity, _d2_entity)] = False
        
        return _tps_checked
    
class SimilarityBasedProgressiveMatching(ProgressiveMatching):
    """Applies similarity based candidate graph prunning, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Similarity Based Progressive Matching"
    _method_info: str = "Applies similarity based candidate graph prunning, sorts retained comparisons and applies Progressive Matching"

    def __init__(
            self,
            budget: int = 0,
            pwScheme: str = 'ACF',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)
        self._pwScheme : str = pwScheme
        
    def extract_tps_checked(self, **kwargs) -> dict:
        pass
        
class GlobalPSNM(SimilarityBasedProgressiveMatching):
    """Applies Global Progressive Sorted Neighborhood Matching
    """

    _method_name: str = "Global Progressive Sorted Neighborhood Matching"
    _method_info: str = "For each entity sorted accordingly to its block's tokens, " + \
                        "evaluates its neighborhood pairs defined within shifting windows of incremental size" + \
                        " and retains the globally best candidate pairs"

    def __init__(
            self,
            budget: int = 0,
            pwScheme: str = 'ACF',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, pwScheme, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)

    def _predict_raw_blocks(self, blocks: dict):
        gpsn : GlobalProgressiveSortedNeighborhood = GlobalProgressiveSortedNeighborhood(self._pwScheme, self._budget)
        candidates :  PriorityQueue = gpsn.process(blocks=blocks, data=self.data, tqdm_disable=True, emit_all_tps_stop=self._emit_all_tps_stop)
        self.pairs = []
        while(not candidates.empty()):
            _, entity_id, candidate_id = candidates.get()
            self.pairs.append((entity_id, candidate_id))
            if(self._emit_all_tps_stop): self.true_pair_checked = self.extract_tps_checked(entity=entity_id, neighbor=candidate_id)
          
        return self.pairs

    def _predict_prunned_blocks(self, blocks: dict):
        raise NotImplementedError("Sorter Neighborhood Algorithms don't support prunned blocks")
    
    def extract_tps_checked(self, **kwargs) -> dict:
        self.true_pair_checked = dict() if self.true_pair_checked is None else self.true_pair_checked
        entity = kwargs['entity']
        neighbor = kwargs['neighbor']
        
        entity_id = self.data._gt_to_ids_reversed_1[entity] if entity < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[entity]
        neighbor_id = self.data._gt_to_ids_reversed_1[neighbor] if neighbor < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[neighbor]
        _d1_entity, _d2_entity = (entity_id, neighbor_id) if entity < self.data.dataset_limit else (neighbor_id, entity_id)
        if _d2_entity in self.data.pairs_of[_d1_entity]:
            self.true_pair_checked[canonical_swap(_d1_entity, _d2_entity)] = False
            
        return self.true_pair_checked
    
class LocalPSNM(SimilarityBasedProgressiveMatching):
    """Applies Local Progressive Sorted Neighborhood Matching
    """

    _method_name: str = "Global Progressive Sorted Neighborhood Matching"
    _method_info: str = "For each entity sorted accordingly to its block's tokens, " + \
                        "evaluates its neighborhood pairs defined within shifting windows of incremental size" + \
                        " and retains the globally best candidate pairs"

    def __init__(
            self,
            budget: int = 0,
            pwScheme: str = 'ACF',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, pwScheme, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)

    def _predict_raw_blocks(self, blocks: dict):
        lpsn : LocalProgressiveSortedNeighborhood = LocalProgressiveSortedNeighborhood(self._pwScheme, self._budget)
        candidates : list = lpsn.process(blocks=blocks, data=self.data, tqdm_disable=True, emit_all_tps_stop=self._emit_all_tps_stop)
        self.pairs = candidates
        if(self._emit_all_tps_stop): self.true_pair_checked = self.extract_tps_checked(candidates=candidates) 
        return self.pairs

    def _predict_prunned_blocks(self, blocks: dict):
        raise NotImplementedError("Sorter Neighborhood Algorithms don't support prunned blocks " + \
                                "(pre comparison-cleaning entities per block distribution required")
        
    def extract_tps_checked(self, **kwargs) -> dict:
        _tps_checked = dict()
        _candidates = kwargs['candidates']
        
        for entity, neighbor in _candidates:
            entity_id = self.data._gt_to_ids_reversed_1[entity] if entity < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[entity]
            neighbor_id = self.data._gt_to_ids_reversed_1[neighbor] if neighbor < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[neighbor]
            _d1_entity, _d2_entity = (entity_id, neighbor_id) if entity < self.data.dataset_limit else (neighbor_id, entity_id)
            if _d2_entity in self.data.pairs_of[_d1_entity]:
                _tps_checked[canonical_swap(_d1_entity, _d2_entity)] = False  
        return _tps_checked
class RandomPM(ProgressiveMatching):
    """Picks a number of random comparisons equal to the available budget
    """

    _method_name: str = "Random Progressive Matching"
    _method_info: str = "Picks a number of random comparisons equal to the available budget"

    def __init__(
            self,
            budget: int = 0,
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)

    def _predict_raw_blocks(self, blocks: dict) -> None:
        cp : ComparisonPropagation = ComparisonPropagation()
        cleaned_blocks = cp.process(blocks=blocks, data=self.data, tqdm_disable=True)
        self._predict_prunned_blocks(cleaned_blocks)

    def _predict_prunned_blocks(self, blocks: dict) -> None:
        _all_pairs = [(id1, id2) for id1 in blocks for id2 in blocks[id1]]
        _total_pairs = len(_all_pairs)
        random_pairs = sample(_all_pairs, self._budget) if self._budget <= _total_pairs and not self._emit_all_tps_stop else _all_pairs
        if(self._emit_all_tps_stop): self.true_pair_checked = self.extract_tps_checked(candidates=random_pairs)
        self.pairs.add_edges_from(random_pairs)
        
    def extract_tps_checked(self, **kwargs) -> dict:
        _tps_checked = dict()
        _candidates = kwargs['candidates']
        
        for entity, neighbor in _candidates:
            entity_id = self.data._gt_to_ids_reversed_1[entity] if entity < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[entity]
            neighbor_id = self.data._gt_to_ids_reversed_1[neighbor] if neighbor < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[neighbor]
            _d1_entity, _d2_entity = (entity_id, neighbor_id) if entity < self.data.dataset_limit else (neighbor_id, entity_id)
            if _d2_entity in self.data.pairs_of[_d1_entity]:
                _tps_checked[canonical_swap(_d1_entity, _d2_entity)] = False  
        return _tps_checked
        
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
            budget: int = 0,
            w_scheme: str = 'X2',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:
        
        super().__init__(budget, w_scheme, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)


    def _predict_raw_blocks(self, blocks: dict) -> None:
        
        pes : ProgressiveEntityScheduling = ProgressiveEntityScheduling(self._w_scheme, self._budget)
        pes.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=None, method=self._method, emit_all_tps_stop=self._emit_all_tps_stop)
        self.pairs = pes.produce_pairs()
        if(self._emit_all_tps_stop): self.true_pair_checked = self.extract_tps_checked(candidates=self.pairs)

    def _predict_prunned_blocks(self, blocks: dict):
        return self._predict_raw_blocks(blocks)
        # raise NotImplementedError("Sorter Neighborhood Algorithms doesn't support prunned blocks (lack of precalculated weights)")
        
    def extract_tps_checked(self, **kwargs) -> dict:
        _tps_checked = dict()
        _candidates = kwargs['candidates']
        
        for entity, neighbor in _candidates:
            entity_id = self.data._gt_to_ids_reversed_1[entity] if entity < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[entity]
            neighbor_id = self.data._gt_to_ids_reversed_1[neighbor] if neighbor < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[neighbor]
            _d1_entity, _d2_entity = (entity_id, neighbor_id) if entity < self.data.dataset_limit else (neighbor_id, entity_id)
            if _d2_entity in self.data.pairs_of[_d1_entity]:
                _tps_checked[canonical_swap(_d1_entity, _d2_entity)] = False  
        return _tps_checked
    
    
class WhooshPM(ProgressiveMatching):
    """Applies progressive index based matching using whoosh library 
    """

    _method_name: str = "Whoosh Progressive Matching"
    _method_info: str = "Applies Whoosh Progressive Matching - Indexes the entities of the second dataset, " + \
                        "stores their specified attributes, " + \
                        "defines a query for each entity of the first dataset, " + \
                        "and retrieves its pair candidates from the indexer within specified budget"

    def __init__(
            self,
            budget: int = 0,
            metric: str = 'TF-IDF',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:
        # budget set to float('inf') implies unlimited budget
        super().__init__(budget, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)
     
    def _set_whoosh_datasets(self) -> None:
        """Saves the rows of both datasets corresponding to the indices of the entities that have been retained after comparison cleaning
        """
        
        self._whoosh_d1 = self.data.dataset_1[self.attributes + [self.data.id_column_name_1]] if self.attributes else self.data.dataset_1
        self._whoosh_d1 = self._whoosh_d1[self._whoosh_d1[self.data.id_column_name_1].isin(self._whoosh_d1_retained_index)]
        if(not self.data.is_dirty_er):  
            self._whoosh_d2 = self.data.dataset_2[self.attributes + [self.data.id_column_name_2]] if self.attributes else self.data.dataset_2
            self._whoosh_d2 = self._whoosh_d2[self._whoosh_d2[self.data.id_column_name_2].isin(self._whoosh_d2_retained_index)]
        

    def _set_retained_entries(self) -> None:
        """Saves the indices of entities of both datasets that have been retained after comparison cleaning
        """
        self._whoosh_d1_retained_index = pd.Index([self.data._gt_to_ids_reversed_1[id] 
        for id in self._si.d1_retained_ids])
        
        if(not self.data.is_dirty_er):
            self._whoosh_d2_retained_index = pd.Index([self.data._gt_to_ids_reversed_2[id] 
        for id in self._si.d2_retained_ids])
    
    
    def _initialize_index_path(self):
        """Creates index directory if non-existent, constructs the absolute path to the current whoosh index
        """
        global INDEXER_DIR
        INDEXER_DIR = os.path.abspath(INDEXER_DIR)  
        _d1_name = self.data.dataset_name_1 if self.data.dataset_name_1 is not None else 'd3'    
        self._index_path = os.path.join(INDEXER_DIR, _d1_name if self.data.is_dirty_er else (_d1_name + (self.data.dataset_name_2 if self.data.dataset_name_2 is not None else 'd4')))
        if not os.path.exists(self._index_path):
            print('Created index directory at: ' + self._index_path)
            os.makedirs(self._index_path, exist_ok=True)
        
    
    def _create_index(self):
        """Defines the schema [ID, CONTENT], creates the index in the defined path 
           and populates it with all the entities of the target dataset (first - Dirty ER, second - Clean ER)
        """
        self._schema = Schema(ID=ID(stored=True), content=TEXT(stored=True))
        self._index = create_in(self._index_path, self._schema)
        writer = self._index.writer()
        
        _target_dataset = self._whoosh_d1 if self.data.is_dirty_er else self._whoosh_d2
        _id_column_name = self.data.id_column_name_1 if self.data.is_dirty_er else self.data.id_column_name_2
        
        for _, entity in _target_dataset.iterrows():
            entity_values = [str(entity[column]) for column in _target_dataset.columns if column != _id_column_name]
            writer.add_document(ID=entity[_id_column_name], content=' '.join(entity_values))
        writer.commit()
    
    def _populate_whoosh_dataset(self) -> None:
        """For each retained entity in the first dataset, construct a query with its text content,
           parses it to the indexers, retrieves best candidates and stores them in entity's neighborhood.
           Finally, neighborhoods are sorted in descending order of their average weight
        """
        # None value for budget implies unlimited budget in whoosh 
        _query_budget = None if is_infinite(self._budget) else max(1, 2 * self._budget / len(self._whoosh_d1))
        
        if(self.metric not in whoosh_similarity_function):
            print(f'{self.metric} Similarity Function is Undefined')
            self.metric = 'Frequency'
        print(f'Applying {self.metric} Similarity Function')
        _scorer = whoosh_similarity_function[self.metric]
        
        with self._index.searcher(weighting=_scorer) as searcher:
            self._parser = qparser.QueryParser('content', schema=self._index.schema, group=qparser.OrGroup)
            for _, entity in self._whoosh_d1.iterrows():
                entity_values = [str(entity[column]) for column in self._whoosh_d1.columns if column != self.data.id_column_name_1]
                entity_string = ' '.join(entity_values)
                entity_id = entity[self.data.id_column_name_1]
                entity_query = self._parser.parse(entity_string)
                query_results = searcher.search(entity_query, limit = _query_budget)
                
                for neighbor in query_results:
                    _score = neighbor.score
                    _neighbor_id = neighbor['ID']
                    self._sorted_dataset._insert_entity_neighbor(entity=entity_id, neighbor=_neighbor_id, weight=_score)
        
        self._sorted_dataset._sort_neighborhoods_by_avg_weight()
        
    def _emit_pairs(self) -> None:
        """Returns a list of candidate pairs that have been emitted following the requested method"""       
        self.pairs = self._sorted_dataset._emit_pairs(method=self._method, data=self.data)
                       
    def _predict_raw_blocks(self, blocks: dict) -> None:
        self._start_time = time()
        self._si = SubsetIndexer(blocks=blocks, data=self.data, subset=False)
        self._set_retained_entries()
        self._set_whoosh_datasets()
        self._initialize_index_path()
        self._create_index()
        self._to_emit_pairs : List[Tuple[int, int]] = []
        self._budget = float('inf') if self._emit_all_tps_stop else self._budget
        self._sorted_dataset = WhooshDataset(list(self._whoosh_d1_retained_index), self._budget)
        self._populate_whoosh_dataset()
        self._emit_pairs()
        self.execution_time = time() - self._start_time
        if(self._emit_all_tps_stop): self.true_pair_checked = self.extract_tps_checked(candidates=self.pairs)
        
    def _predict_prunned_blocks(self, blocks: dict) -> None:
        self._predict_raw_blocks(blocks)    
        
    def extract_tps_checked(self, **kwargs) -> dict:
        _tps_checked = dict()
        _candidates = kwargs['candidates']
        
        for entity, neighbor in _candidates:
            entity_id = self.data._gt_to_ids_reversed_1[entity] if entity < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[entity]
            neighbor_id = self.data._gt_to_ids_reversed_1[neighbor] if neighbor < self.data.dataset_limit else self.data._gt_to_ids_reversed_2[neighbor]
            _d1_entity, _d2_entity = (entity_id, neighbor_id) if entity < self.data.dataset_limit else (neighbor_id, entity_id)
            if _d2_entity in self.data.pairs_of[_d1_entity]:
                _tps_checked[canonical_swap(_d1_entity, _d2_entity)] = False  
        return _tps_checked 
        
        

