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
from .utils import FrequencyEvaluator

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
        self.vectorizer = None

    def vectorizer_based(self) -> bool:
        """
            Checks whether current instance of Joins algorithm is using a frequency vectorizer

            Returns:
                bool: Candidate scores are being calculated through frequency vectorizer
        """
        return (self.vectorizer is not None)
   
    def dirty_indexing(self):
        """Applies Dirty Indexing - Evaluates the similarity of all the entities of the target dataset
        """
        eid = 0
        for entity in self.indexed_entities:
            candidates = set()
            for token in entity:
                if token in self.entity_index:
                    current_candidates = self.entity_index[token]
                    for candidate_id in current_candidates:
                        if(not self.vectorizer_based()):
                            if self._flags[candidate_id] != eid:
                                self._counters[candidate_id] = 0
                                self._flags[candidate_id] = eid
                            self._counters[candidate_id] += 1
                        candidates.add(candidate_id)
            self._process_candidates(candidates, eid, len(entity))
            self._progress_bar.update(1)
            eid += 1
    
    def get_id_from_index(self, index : int):
        return (i if self.reverse_order else (index+self.data.dataset_limit))
    
       
    def clean_indexing(self):
        """Applies Dirty Indexing - One of the datasets (depends on the order of indexing) is set as the indexer.
           For each entry of that dataset, its similarity scores are being calculated with each entity of the target dataset.
           The top-K best results for each source entity are chosen.
        """
        for i in range(0, self.indexed_entities_count):
            candidates = set()
            record = self.indexed_entities[i]
            entity_id =  self.get_id_from_index(i)
            for token in record:
                if token in self.entity_index:
                    current_candidates = self.entity_index[token]
                    for candidate_id in current_candidates:
                        if(not self.vectorizer_based()):
                            if self._flags[candidate_id] != entity_id:
                                self._counters[candidate_id] = 0
                                self._flags[candidate_id] = entity_id
                            self._counters[candidate_id] += 1
                        candidates.add(candidate_id)
            if 0 < len(candidates):
                self._process_candidates(candidates, entity_id, len(record))
            self._progress_bar.update(1)
        
    def setup_indexing(self):
        """Defines the indexed and target entities, as well as their total count
        
        """
        self.indexed_entities, self.indexed_entities_count = (self._entities_d1, self.data.num_of_entities_1) if (self.reverse_order or self.data.is_dirty_er) \
                                                              else (self._entities_d2, self.data.num_of_entities_2)
                                                              
        self.target_entities, self.target_entities_count = (self._entities_d1, self.data.num_of_entities_1) if (not self.reverse_order or self.data.is_dirty_er) \
                                                              else (self._entities_d2, self.data.num_of_entities_2)                                                      
                                                    
    def fit(self,
            data: Data,
            vectorizer: FrequencyEvaluator = None,
            reverse_order: bool = False,
            attributes_1: list = None,
            attributes_2: list = None,
            tqdm_disable: bool = False,
            store_neighborhoods : bool = False
    ) -> networkx.Graph:
        """Joins main method

            Args:
                data (Data): dataset module
                vectorizer (FrequencyEvaluator, optional): Vectorizer will be used for similarity evaluation
                reverse_order (bool, optional): _description_. Defaults to False.
                attributes_1 (list, optional): _description_. Defaults to None.
                attributes_2 (list, optional): _description_. Defaults to None.
                tqdm_disable (bool, optional): _description_. Defaults to False.
                save_to_json (bool, optional): Store indexed dataset neighborhoods in a dictionary of form
                                               [indexed dataset entity id] -> [ids of top-k neighbors in target dataset]
            Returns:
                networkx.Graph: graph containg nodes as entities and edges as similarity score
        """
        if reverse_order and data.is_dirty_er:
            raise ValueError("Can't have reverse order in Dirty Entity Resolution")

        start_time = time()
        self.tqdm_disable, self.reverse_order, self.attributes_1, self.attributes_2, self.data, self.vectorizer, self.store_neighborhoods = \
            tqdm_disable, reverse_order, attributes_1, attributes_2, data, vectorizer, store_neighborhoods

        self._entities_d1 = data.dataset_1[attributes_1 if attributes_1 else data.attributes_1] \
                            .apply(" ".join, axis=1) \
                            .apply(self._tokenize_entity) \
                            .values.tolist()

        if not data.is_dirty_er:
            self._entities_d2 = data.dataset_2[attributes_2 if attributes_2 else data.attributes_2] \
                    .apply(" ".join, axis=1) \
                    .apply(self._tokenize_entity) \
                    .values.tolist()

        self.neighborhoods = defaultdict(list) if self.store_neighborhoods else None
        self.setup_indexing()
        
        self._progress_bar = tqdm(
            total=self.indexed_entities_count,
            desc=self._method_name+" ("+self.metric+")", disable=self.tqdm_disable
        )
        
        self._flags = np.empty([self.target_entities_count]) if (not self.vectorizer_based()) else None
        self._counters = np.zeros([self.target_entities_count]) if (not self.vectorizer_based()) else None
        self._source_frequency = np.empty([self.target_entities_count]) if (not self.vectorizer_based()) else None
        if(not self.vectorizer_based()) : self._flags[:] = -1
        self.pairs = networkx.Graph()
        self.entity_index = self._create_entity_index()

        if self.data.is_dirty_er:
            self.dirty_indexing()
        else:
            self.clean_indexing()
            
        if(self.store_neighborhoods): self._process_neighborhoods()   
                
        self._progress_bar.close()
        self.execution_time = time() - start_time
        return self.pairs

    def _tokenize_entity(self, entity: str) -> set:
        if self.vectorizer is not None:
            return entity.lower()
        elif self.tokenization == 'qgrams':
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
        
    def _calc_vector_similarity(self, id1 : int, id2 : int) -> float:
        """Vector based similarity score

        Args:
            id1 (int): D1 entity ID
            id2 (int): D2 entity ID

        Returns:
            float: vector based similarity
        """
        return self.vectorizer.predict(id1=id1, id2=id2)

    def _create_entity_index(self) -> dict:
        entity_index = defaultdict(set)
        for eid, entity in enumerate(self.target_entities):
            for token in entity:
                entity_index[token].add(eid)
                
            if(not self.vectorizer_based()):
                self._source_frequency[eid] = len(entity)
            self._progress_bar.update(1)

        return entity_index   

    def _insert_to_graph(self, entity_id1, entity_id2, similarity):
        if self.similarity_threshold <= similarity:
            self.pairs.add_edge(entity_id1, entity_id2, weight=similarity)
            
    def _store_neighborhood(self, entity_id1, entity_id2, similarity):
        if self.similarity_threshold <= similarity:
            self.neighborhoods[entity_id2].append((similarity, entity_id1))
            
    def _process_neighborhoods(self):
        """Sorts the candidates of each indexed entity's neighborhood in descending order
           of similarity. 
        """
        for d1_id, d2_ids in self.neighborhoods.items():
            self.neighborhoods[d1_id] = sorted(d2_ids, key=lambda x: (-x[0], x[1]))
            

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

    def export_to_df(self, prediction, tqdm_enable=False) -> pd.DataFrame:
        """creates a dataframe with the predicted pairs

        Args:
            prediction (any): Predicted candidate pairs,
            tqdm_enable (bool, optional): Enable tqdm. Defaults to False.

        Returns:
            pd.DataFrame: Dataframe with the predicted pairs
        """
        pairs_list = []

        is_dirty_er = self.data.is_dirty_er
        dataset_limit = self.data.dataset_limit
        gt_to_ids_reversed_1 = self.data._gt_to_ids_reversed_1
        if not is_dirty_er:
            gt_to_ids_reversed_2 = self.data._gt_to_ids_reversed_2

        for edge in tqdm(prediction.edges, disable=not tqdm_enable, desc="Exporting to DataFrame"):
            node1, node2 = edge

            if not is_dirty_er:
                if node1 < dataset_limit:
                    id1 = gt_to_ids_reversed_1[node1]
                    id2 = gt_to_ids_reversed_2[node2]
                else:
                    id1 = gt_to_ids_reversed_1[node2]
                    id2 = gt_to_ids_reversed_2[node1]
            else:
                id1 = gt_to_ids_reversed_1[node1]
                id2 = gt_to_ids_reversed_1[node2]

            pairs_list.append((id1, id2))

        pairs_df = pd.DataFrame(pairs_list, columns=['id1', 'id2'])

        return pairs_df
    
class EJoin(AbstractJoin):
    """
     E Join algorithm
    """
    _method_name = "EJoin"
    _method_info = " EJoin algorithm"
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
            sim = self._calc_similarity(
                  self._counters[candidate_id],
                  self._source_frequency[candidate_id],
                  tokens_size
                )
            d1_id = candidate_id+self.data.dataset_limit if (self.reverse_order \
                                                    and not self.data.is_dirty_er) \
                                                    else candidate_id
            d2_id = entity_id
            self._insert_to_graph(d1_id, d2_id, sim)
            if(self.store_neighborhoods): self._store_neighborhood(d1_id, d2_id, sim)

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
        pq.put(minimum_weight)
        for index, candidate_id in enumerate(candidates):
            if(self.vectorizer is None):
                sim = self._calc_similarity(self._counters[candidate_id], self._source_frequency[candidate_id], tokens_size)
            else:
                sim = self._calc_vector_similarity(((candidate_id + self.data.dataset_limit) if self.reverse_order else candidate_id), entity_id)
            if minimum_weight < sim:
                pq.put(sim)
                if self.K < pq.qsize():
                    minimum_weight = pq.get()

        minimum_weight = pq.get()
        for index, candidate_id in enumerate(candidates):
            self.similarity_threshold = minimum_weight
            if(self.vectorizer is None):
                sim = self._calc_similarity(self._counters[candidate_id], self._source_frequency[candidate_id], tokens_size)
            else:
                sim = self._calc_vector_similarity(((candidate_id + self.data.dataset_limit) if self.reverse_order else candidate_id), entity_id)
            self._insert_to_graph(
                candidate_id + self.data.dataset_limit if self.reverse_order else candidate_id,
                entity_id,
                sim
            )
            if(self.store_neighborhoods): self._store_neighborhood(candidate_id + self.data.dataset_limit if self.reverse_order else candidate_id, \
                                                                   entity_id, \
                                                                   sim)

    def _configuration(self) -> dict:
        return {
            "similarity_threshold" : self.similarity_threshold,
            "K" : self.K,
            "metric" : self.metric,
            "tokenization" : self.tokenization,
            "qgrams": self.qgrams
        }
    
class PETopKJoin(TopKJoin):
    """Progressive Entity Resolution Top-K class of Joins module
    """
    _method_name = "Progressive Top-K Join"
    _method_info = "Progressive Top-K Join algorithm"
    _method_short_name = "PETopKJ"    

    def __init__(
            self,
            K: int,
            metric: str,
            tokenization: str,
            qgrams: int = 2
    ) -> None:
        """AbstractJoin Constructor

        Args:
            K (int): Number of candidates per entity
            metric (str): String similarity metric
            tokenization (str): Tokenizer
            qgrams (int, optional): For Jaccard metric. Defaults to 2.
        """
        super().__init__(K=K,
                        metric=metric,
                        tokenization=tokenization,
                        qgrams=qgrams)
    
    
    def _get_similarity(self, target_id : int, indexed_id : int, tokens_size : int):
        return self._calc_similarity(self._counters[target_id], self._source_frequency[target_id], tokens_size) \
               if (self.vectorizer is None) else \
               self._calc_vector_similarity(target_id , indexed_id)
        
    def _process_candidates(self, candidates: set, entity_id: int, tokens_size: int) -> None:
        minimum_weight=0
        pq = PriorityQueue()
        for index, candidate_id in enumerate(candidates):
            
            _target_id = candidate_id
            _indexed_id = entity_id + self.data.dataset_limit
            
            sim : float = self._get_similarity(target_id=_target_id,
                                               indexed_id=_indexed_id,
                                               tokens_size=tokens_size)
            
            # target dataset entity id set to negative
            # so higher identifier kicked out first (simulating descending order with ascending PQ)
            _pair = (sim, -_target_id, _indexed_id)

            if minimum_weight <= sim:
                pq.put(_pair)
                if self.K < pq.qsize():
                    minimum_weight, _, _ = pq.get()
        
        if(self.store_neighborhoods):
            _first_element = True
            while(not pq.empty()):
                _sim, _target_id, _indexed_id = pq.get()
                if _first_element:
                    self.similarity_threshold = _sim
                    _first_element = False
                    
                self._store_neighborhood(entity_id1= -_target_id,
                                         entity_id2= _indexed_id,
                                         similarity= _sim)
                self._insert_to_graph(entity_id1=-_target_id,
                                      entity_id2=_indexed_id,
                                      similarity=_sim) 
        else:
            self.similarity_threshold, _, _ = pq.get()
            for index, candidate_id in enumerate(candidates):
                _target_id = candidate_id
                _indexed_id = entity_id + self.data.dataset_limit
                self._insert_to_graph(entity_id1=_target_id,
                                      entity_id2=_indexed_id,
                                      similarity=self._get_similarity(target_id=_target_id,
                                                                      indexed_id=_indexed_id,
                                                                      tokens_size=tokens_size))

    def _process_neighborhoods(self, strict_top_k : bool = True):
        """Sorts the candidates of each indexed entity's neighborhood in descending order
           of similarity. If strict top-K instance is chosen, it retains max K best candidates
           per entity.
        Args:
            strict_top_k (bool, optional): Retain strictly (max) top-K candidates per entity
        """
        for d1_id, d2_ids in self.neighborhoods.items():
            _sorted_neighborhood = sorted(d2_ids, key=lambda x: (-x[0], x[1])) 
            self.neighborhoods[d1_id] = _sorted_neighborhood[:self.K] if strict_top_k else \
                                        _sorted_neighborhood

    def setup_indexing(self):
        """Defines the indexed and target entities, as well as their total count
        
        """
        # self.indexed_entities, self.indexed_entities_count = (self._entities_d2, self.data.num_of_entities_2) if (self.reverse_order) \
        #                                                       else (self._entities_d1, self.data.num_of_entities_1)
                                                              
        # self.target_entities, self.target_entities_count = (self._entities_d1, self.data.num_of_entities_1) if (self.reverse_order or self.data.is_dirty_er) \
        #                                                       else (self._entities_d2, self.data.num_of_entities_2)     
        self.indexed_entities, self.indexed_entities_count = (self._entities_d2, self.data.num_of_entities_2)
                                                              
        self.target_entities, self.target_entities_count = (self._entities_d1, self.data.num_of_entities_1)                                                      

    def get_id_from_index(self, index : int):
        return index

    def _configuration(self) -> dict:
        return {
            "similarity_threshold" : self.similarity_threshold,
            "K" : self.K,
            "metric" : self.metric,
            "tokenization" : self.tokenization,
            "qgrams": self.qgrams
        }






        


