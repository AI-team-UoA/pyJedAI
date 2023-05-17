import itertools
import math
import re
from collections import defaultdict
from queue import PriorityQueue
from time import time

import networkx
import nltk
import numpy as np
import pandas as pd
import tqdm
from tqdm.autonotebook import tqdm

from .datamodel import Data, PYJEDAIFeature
from .evaluation import Evaluation

class AbstractJoin(PYJEDAIFeature):
    """Abstract class of Joins module
    """

    def __init__(
            self,
            metric: str,
            tokenization: str,
            qgrams: int = 2,
            similarity_threshold: float = None
    ) -> None:
        """AbstractJoin Constructor

        Args:
            metric (str): String similarity metric
            tokenization (str): Tokenizer
            qgrams (int, optional): For Jaccard metric. Defaults to 2.
            similarity_threshold (float, optional): Threshold for preserving pair. Defaults to None.
        """
        self.metric = metric
        self.qgrams = qgrams
        self.tokenization = tokenization
        self._source_frequency: np.array
        self.similarity_threshold: float = similarity_threshold
        self.reverse_order: bool
        self.attributes_1: list
        self.attributes_2: list
        self._flags: np.array
        self.pairs: networkx.Graph

    def fit(self,
            data: Data,
            reverse_order: bool = False,
            attributes_1: list = None,
            attributes_2: list = None,
            tqdm_disable: bool = False
    ) -> networkx.Graph:
        """Joins main method

            Args:
                data (Data): dataset module
                reverse_order (bool, optional): _description_. Defaults to False.
                attributes_1 (list, optional): _description_. Defaults to None.
                attributes_2 (list, optional): _description_. Defaults to None.
                tqdm_disable (bool, optional): _description_. Defaults to False.

            Returns:
                networkx.Graph: graph containg nodes as entities and edges as similarity score
        """
        if reverse_order and data.is_dirty_er:
            raise ValueError("Can't have reverse order in Dirty Entity Resolution")

        start_time = time()
        self.tqdm_disable, self.reverse_order, self.attributes_1, self.attributes_2, self.data = \
            tqdm_disable, reverse_order, attributes_1, attributes_2, data

        self._entities_d1 = data.dataset_1[attributes_1 if attributes_1 else data.attributes_1] \
                            .apply(" ".join, axis=1) \
                            .apply(self._tokenize_entity) \
                            .values.tolist()

        if not data.is_dirty_er:
            self._entities_d2 = data.dataset_2[attributes_2 if attributes_2 else data.attributes_2] \
                    .apply(" ".join, axis=1) \
                    .apply(self._tokenize_entity) \
                    .values.tolist()

        num_of_entities = self.data.num_of_entities_2 if reverse_order else self.data.num_of_entities_1

        self._progress_bar = tqdm(
            total=self.data.num_of_entities if not self.data.is_dirty_er else num_of_entities*2,
            desc=self._method_name+" ("+self.metric+")", disable=self.tqdm_disable
        )

        self._flags, \
        self._counters, \
        self._sims, \
        self._source_frequency, \
        self.pairs = np.empty([num_of_entities]), \
                    np.zeros([num_of_entities]), \
                    np.empty([self.data.num_of_entities_1*self.data.num_of_entities_2]), \
                    np.empty([num_of_entities]), \
                    networkx.Graph()
        self._flags[:] = -1
        entity_index = self._create_entity_index(
                self._entities_d2 if reverse_order else self._entities_d1
            )

        if self.data.is_dirty_er:
            eid = 0
            for entity in self._entities_d1:
                candidates = set()
                for token in entity:
                    if token in entity_index:
                        current_candidates = entity_index[token]
                        for candidate_id in current_candidates:
                            if self._flags[candidate_id] != eid:
                                self._counters[candidate_id] = 0
                                self._flags[candidate_id] = eid
                            self._counters[candidate_id] += 1
                            candidates.add(candidate_id)
                self._process_candidates(candidates, eid, len(entity))
                self._progress_bar.update(1)
                eid += 1
        else:
            if reverse_order:
                entities = self._entities_d1
                num_of_entities = self.data.num_of_entities_1
            else:
                entities = self._entities_d2
                num_of_entities = self.data.num_of_entities_2

            for i in range(0, num_of_entities):
                candidates = set()
                record = entities[i]
                entity_id = i if reverse_order else i+self.data.dataset_limit
                for token in record:
                    if token in entity_index:
                        current_candidates = entity_index[token]
                        for candidate_id in current_candidates:
                            if self._flags[candidate_id] != entity_id:
                                self._counters[candidate_id] = 0
                                self._flags[candidate_id] = entity_id
                            self._counters[candidate_id] += 1
                            candidates.add(candidate_id)
                if 0 < len(candidates):
                    self._process_candidates(candidates, entity_id, len(record))
                self._progress_bar.update(1)
        self._progress_bar.close()
        self.execution_time = time() - start_time

        return self.pairs

    def _tokenize_entity(self, entity: str) -> set:
        if self.tokenization == 'qgrams':
            return set([' '.join(grams) for grams in nltk.ngrams(entity.lower(), n=self.qgrams)])
        elif self.tokenization == 'standard':
            return set(filter(None, re.split('[\\W_]', entity.lower())))
        elif self.tokenization == 'standard_multiset':
            tok_ids_index = {}
            multiset = set()
            for tok in set(filter(None, re.split('[\\W_]', entity.lower()))):
                tok_id =  tok_ids_index[tok] if tok in tok_ids_index else 0
                multiset.add(tok+str(tok_id))
                tok_ids_index[tok] = tok_id+1
            return multiset
        elif self.tokenization == 'qgrams_multiset':
            grams_ids_index = {}
            qgrams = set()
            for gram in set([' '.join(grams) for grams in nltk.ngrams(entity.lower(), n=self.qgrams)]):
                gram_id =  grams_ids_index[gram] if gram in grams_ids_index else 0
                qgrams.add(gram+str(gram_id))
                grams_ids_index[gram] = gram_id+1
            return qgrams
        else:
            raise AttributeError("Tokenization not found")

    def _calc_similarity(
            self,
            common_tokens: int,
            source_frequency: int,
            tokens_size: int
        ) -> float:
        """Similarity score

        Args:
            common_tokens (int): number of common tokens
            source_frequency (int): frequency
            tokens_size (int): size of tokens

        Returns:
            float: similarity
        """
        if self.metric == 'cosine':
            return common_tokens / math.sqrt(source_frequency*tokens_size)
        elif self.metric == 'dice':
            return 2 * common_tokens / (source_frequency+tokens_size)
        elif self.metric == 'jaccard':
            return common_tokens / (source_frequency+tokens_size-common_tokens)

    def _create_entity_index(self, entities: list) -> dict:
        entity_index = defaultdict(set)
        entity_id = itertools.count()
        for entity in entities:
            eid = next(entity_id)
            for token in entity:
                entity_index[token].add(eid)
            self._source_frequency[eid] = len(entity)
            self._progress_bar.update(1)

        return entity_index

#     def _similarity(self, entity_id1: int, entity_id2: int, attributes: any=None) -> float:
#         similarity: float = 0.0
#         if isinstance(attributes, dict):
#             for attribute, weight in self.attributes.items():
#                 similarity += weight*self._metric(
#                     self.data.entities.iloc[entity_id1][attribute],
#                     self.data.entities.iloc[entity_id2][attribute]
#                 )
#         if isinstance(attributes, list):
#             for attribute in self.attributes:
#                 similarity += self._metric(
#                     self.data.entities.iloc[entity_id1][attribute],
#                     self.data.entities.iloc[entity_id2][attribute]
#                 )
#                 similarity /= len(self.attributes)
#         else:
#             # print(self.data.entities.iloc[entity_id1].str.cat(sep=' '),
#                 # self.data.entities.iloc[entity_id2].str.cat(sep=' '))
#             # concatenated row string
#             similarity = self._metric(
#                 self.data.entities.iloc[entity_id1].str.cat(sep=' '),
#                 self.data.entities.iloc[entity_id2].str.cat(sep=' ')
#             )
#         return similarity

    def _insert_to_graph(self, entity_id1, entity_id2, similarity):
        if self.similarity_threshold <= similarity:
            self.pairs.add_edge(entity_id1, entity_id2, weight=similarity)

    def evaluate(self, prediction=None, export_to_df: bool = False,
                 export_to_dict: bool = False, with_classification_report: bool = False,
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

    def stats(self) -> None:
        pass

    def _configuration(self) -> dict:
        return {
            "similarity_threshold" : self.similarity_threshold,
            "metric" : self.metric,
            "tokenization" : self.tokenization,
            "qgrams": self.qgrams
        }    

class ΕJoin(AbstractJoin):
    """
     Ε Join algorithm
    """
    _method_name = "EJoin"
    _method_info = " ΕJoin algorithm"
    _method_short_name = "EJ"

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        metric: str = 'cosine',
        tokenization: str = 'qgrams',
        qgrams: int = 2
    ) -> None:
        super().__init__(metric, tokenization, qgrams, similarity_threshold)

    def _process_candidates(self, candidates: set, entity_id: int, tokens_size: int) -> None:
        for candidate_id in candidates:
            self._insert_to_graph(
                candidate_id+self.data.dataset_limit if self.reverse_order \
                                                        and not self.data.is_dirty_er \
                                                    else candidate_id,
                entity_id,
                self._calc_similarity(
                    self._counters[candidate_id],
                    self._source_frequency[candidate_id],
                    tokens_size
                )
            )

class TopKJoin(AbstractJoin):
    """Top-K Join algorithm
    """

    _method_name = "Top-K Join"
    _method_info = "Top-K Join algorithm"
    _method_short_name = "TopKJ"

    def __init__(self,
                 K: int,
                 metric: str,
                 tokenization: str,
                 qgrams: int = 2
        ) -> None:
        super().__init__(metric, tokenization, qgrams)
        self.K = K

    def _process_candidates(self, candidates: set, entity_id: int, tokens_size: int) -> None:
        minimum_weight=0
        pq = PriorityQueue()
        for candidate_id in candidates:
            sim = self._calc_similarity(
                self._counters[candidate_id], self._source_frequency[candidate_id], tokens_size
            )
            if minimum_weight < sim:
                pq.put(sim)
                if self.K < pq.qsize():
                    minimum_weight = pq.get()

        minimum_weight = pq.get()
        for candidate_id in candidates:
            self.similarity_threshold = minimum_weight
            self._insert_to_graph(
                candidate_id + self.data.dataset_limit if self.reverse_order else candidate_id,
                entity_id,
                self._calc_similarity(
                    self._counters[candidate_id], 
                    self._source_frequency[candidate_id],
                    tokens_size
                )
            )

    def _configuration(self) -> dict:
        return {
            "similarity_threshold" : self.similarity_threshold,
            "K" : self.K,
            "metric" : self.metric,
            "tokenization" : self.tokenization,
            "qgrams": self.qgrams
        }
