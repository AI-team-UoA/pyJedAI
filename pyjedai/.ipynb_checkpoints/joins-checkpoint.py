import pandas as pd
import tqdm
from tqdm.notebook import tqdm
import networkx
import os
import sys

import strsimpy
from strsimpy.levenshtein import Levenshtein
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from strsimpy.weighted_levenshtein import WeightedLevenshtein
from strsimpy.damerau import Damerau
from strsimpy.optimal_string_alignment import OptimalStringAlignment
from strsimpy.jaro_winkler import JaroWinkler
from strsimpy.longest_common_subsequence import LongestCommonSubsequence
from strsimpy.metric_lcs import MetricLCS
from strsimpy.ngram import NGram
from strsimpy.qgram import QGram
from strsimpy.overlap_coefficient import OverlapCoefficient
from strsimpy.cosine import Cosine
from strsimpy.jaccard import Jaccard
from strsimpy.sorensen_dice import SorensenDice
from strsimpy import SIFT4

# pyJedAI
from datamodel import Data
from utils import EMPTY

from queue import PriorityQueue

class AbstractJoin:
    
    def __init__(self, metric: str, qgrams: int = 2) -> None:
        self.metric = metric
        self.qgrams = qgrams
        self._initialize_metric()

    def _initialize_metric(self):
        if self.metric == 'levenshtein' or self.metric == 'edit_distance':
            self._metric = Levenshtein().distance
        elif self.metric == 'nlevenshtein':
            self._metric = NormalizedLevenshtein().distance
        elif self.metric == 'jaro_winkler':
            self._metric = JaroWinkler().distance
        elif self.metric == 'metric_lcs':
            self._metric = MetricLCS().distance
        elif self.metric == 'qgram':
            self._metric = NGram(self.qgrams).distance
        # elif self.metric == 'cosine':
        #     cosine = Cosine(self.qgram)
        #     self._metric = cosine.similarity_profiles(cosine.get_profile(entity_1), cosine.get_profile(entity_2))
        elif self.metric == 'jaccard':
            self._metric = Jaccard(self.qgrams).distance
        elif self.metric == 'sorensen_dice':
            self._metric = SorensenDice().distance
        elif self.metric == 'overlap_coefficient':
            self._metric = OverlapCoefficient().distance
        
    def _tokenize_entity(self, entity: str) -> list:
        
        if self.tokenization == 'qgrams':
            return [' '.join(grams) for grams in nltk.ngrams(entity, n=self.qgrams)]
        elif self.tokenization == 'standard':
            return entity.split()
        elif self.tokenization == 'suffix_arrays':
            return [' '.join(grams) for grams in nltk.ngrams(entity, n=self.qgrams)]
        else:
            print("Tokenization not found")
            # TODO error             
        
    def fit(
        self, data: Data, 
        attributes_1: list=None,
        attributes_2: list=None
    ) -> networkx.Graph:
        
        self.attributes_1 = attributes_1
        self.attributes_2 = attributes_2
        self.data = data
        self._progress_bar = tqdm(total=self.data.num_of_entities, desc=self._method_name)
        self._create_similarity_graph()
        return self.pairs
    
    def _similarity(self, entity_id1: int, entity_id2: int, attributes: any=None) -> float:

        similarity: float = 0.0

        if isinstance(attributes, dict):
            for attribute, weight in self.attributes.items():
                similarity += weight*self._metric(
                    self.data.entities.iloc[entity_id1][attribute],
                    self.data.entities.iloc[entity_id2][attribute]
                )
        if isinstance(attributes, list):
            for attribute in self.attributes:
                similarity += self._metric(
                    self.data.entities.iloc[entity_id1][attribute],
                    self.data.entities.iloc[entity_id2][attribute]
                )
                similarity /= len(self.attributes)
        else:
            # print(self.data.entities.iloc[entity_id1].str.cat(sep=' '),
                # self.data.entities.iloc[entity_id2].str.cat(sep=' '))
            # concatenated row string
            similarity = self._metric(
                self.data.entities.iloc[entity_id1].str.cat(sep=' '),
                self.data.entities.iloc[entity_id2].str.cat(sep=' ')
            )

        return similarity
            
class SchemaAgnosticJoin(AbstractJoin):
    '''
    SchemaAgnosticJoin
    '''
    
    _method_name = "Schema Agnostic Join"

    def __init__(
        self, threshold: float, metric: str, 
        tokenization: str, qgrams: int = None) -> None:
        
        super().__init__(metric, qgrams)

        self.similarity_threshold = threshold
        self.tokenization = tokenization
        self.qgrams = qgrams
        self.metric = metric
    
    def _create_similarity_graph(self) -> None:
        self.pairs = networkx.Graph()
        if self.attributes_1 and isinstance(self.attributes_1, dict):
            self.attributes_1 = list(self.attributes_1.keys())

        entity_index_d1 = {}
        for i in range(0, self.data.num_of_entities_1, 1):
            record = self.data.dataset_1.iloc[i, self.attributes_1] if self.attributes_1 else self.data.entities_d1.iloc[i]
            for token in self._tokenize_entity(record):
                entity_index_d1.setdefault(token, set())
                if self.data.is_dirty_er and len(entity_index_d1[token])>0:
                    for candidate_id in entity_index_d1[token]:
                        sim = self._similarity(i, candidate_id, self.attributes_1)
                        self._insert_to_graph(i, candidate_id, sim)
                entity_index_d1[token].add(i)
            self._progress_bar.update(1)
        
        if not self.data.is_dirty_er:
            if self.attributes_2 and isinstance(self.attributes_2, dict):
                self.attributes_2 = list(self.attributes_2.keys())
            for i in range(0, self.data.num_of_entities_2, 1):
                record = self.data.dataset_2.iloc[i, self.attributes_2] if self.attributes_2 else self.data.entities_d2.iloc[i]
                for token in self._tokenize_entity(record):
                    if token in entity_index_d1 and len(entity_index_d1[token])>0:
                        for candidate_id in entity_index_d1[token]:
                            sim = self._similarity(i+self.data.dataset_limit, candidate_id, self.attributes_2)
                            self._insert_to_graph(i+self.data.dataset_limit, candidate_id, sim)
                self._progress_bar.update(1)

    def _insert_to_graph(self, entity_id1, entity_id2, similarity):
        if self.similarity_threshold and similarity > self.similarity_threshold:
            self.pairs.add_edge(entity_id1, entity_id2, weight=similarity)    
        
            
class TopKSchemaAgnosticJoin(AbstractJoin):
    '''
    TopKSchemaAgnosticJoin
    '''
    
    _method_name = "Top-K Schema Agnostic Join"

    def __init__(
        self, K: int, metric: str, 
        tokenization: str, qgrams: int = 2) -> None:
        
        super().__init__(metric, qgrams)

        self.K = K
        self.tokenization = tokenization
    
    def _create_similarity_graph(self) -> None:
        self.pairs = networkx.Graph()
        if self.attributes_1 and isinstance(self.attributes_1, dict):
            self.attributes_1 = list(self.attributes_1.keys())

        entity_index_d1 = {}
        for i in range(0, self.data.num_of_entities_1, 1):
            priority_queue = PriorityQueue()
            minimum_weight = 0.0
            record = self.data.dataset_1.iloc[i, self.attributes_1] if self.attributes_1 else self.data.entities_d1.iloc[i]
            tokens = self._tokenize_entity(record)
            
            
            for token in self._tokenize_entity(record):
                entity_index_d1.setdefault(token, set())
                if self.data.is_dirty_er and len(entity_index_d1[token])>0:
                    for candidate_id in entity_index_d1[token]:
                        sim = self._similarity(i, candidate_id, self.attributes_1)
                        if sim  > minimum_weight:
                            priority_queue.put(sim)
                            if self.K < len(priority_queue):
                                minimum_weight = priority_queue.get()
                        self._insert_to_graph(i, candidate_id, sim)
                entity_index_d1[token].add(i)
            self._progress_bar.update(1)
        
        if not self.data.is_dirty_er:
            if self.attributes_2 and isinstance(self.attributes_2, dict):
                self.attributes_2 = list(self.attributes_2.keys())
            for i in range(0, self.data.num_of_entities_2, 1):
                record = self.data.dataset_2.iloc[i, self.attributes_2] if self.attributes_2 else self.data.entities_d2.iloc[i]
                for token in self._tokenize_entity(record):
                    if token in entity_index_d1 and len(entity_index_d1[token])>0:
                        for candidate_id in entity_index_d1[token]:
                            sim = self._similarity(i+self.data.dataset_limit, candidate_id, self.attributes_2)
                            self._insert_to_graph(i+self.data.dataset_limit, candidate_id, sim)
                self._progress_bar.update(1)
        # else:

    def _insert_to_graph(self, entity_id1, entity_id2, similarity):
        if self.similarity_threshold and similarity > self.similarity_threshold:
            self.pairs.add_edge(entity_id1, entity_id2, weight=similarity)   