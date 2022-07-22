import pandas as pd
import tqdm
from tqdm.notebook import tqdm
import networkx
from queue import PriorityQueue
import nltk, re
import numpy as np
import math
from queue import PriorityQueue

# pyJedAI
from .datamodel import Data
from .utils import EMPTY

class AbstractJoin:
    
    def __init__(
        self, 
        metric: str, 
        tokenization: str, 
        qgrams: int = 2,
        similarity_threshold: float = None
    ) -> None:
        self.metric = metric
        self.qgrams = qgrams
        self.tokenization = tokenization
        self._source_frequency: np.array
        self.similarity_threshold: float = similarity_threshold

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
            print("Tokenization not found")
            # TODO error
        
    def fit(
        self, data: Data, 
        reverse_order: bool=False,
        attributes_1: list=None,
        attributes_2: list=None
    ) -> networkx.Graph:
        self.reverse_order = reverse_order
        self.attributes_1 = attributes_1
        self.attributes_2 = attributes_2
        self.data = data
        
        if reverse_order and data.is_dirty_er:
            print("Can't have reverse order in Dity ER") #TODO
        
        if reverse_order:
            num_of_entities = self.data.num_of_entities_2
        else:
            num_of_entities = self.data.num_of_entities_1
        
        self._progress_bar = tqdm(
            total=self.data.num_of_entities if not self.data.is_dirty_er else num_of_entities*2,
            desc=self._method_name+" ("+self.metric+")"
        )
        
        self._flags = np.empty([num_of_entities])
        self._flags[:] = -1
        self._counters = np.zeros([num_of_entities])
        self._sims = np.empty([self.data.num_of_entities_1*self.data.num_of_entities_2])
        self._source_frequency = np.empty([num_of_entities])

        self.pairs = networkx.Graph()
        if self.attributes_1 and isinstance(self.attributes_1, dict):
            self.attributes_1 = list(self.attributes_1.keys())

        if reverse_order:
            entity_index = self._create_entity_index_d2()
        else:
            entity_index = self._create_entity_index_d1()
        
        if self.attributes_2 and isinstance(self.attributes_2, dict):
            self.attributes_2 = list(self.attributes_2.keys())
        
        if self.data.is_dirty_er:
            for i in range(0, self.data.num_of_entities_1):
                candidates = set()
                record = self.data.dataset_1.iloc[i, self.attributes_1] \
                            if self.attributes_1 else self.data.entities_d1.iloc[i]
                tokens = self._tokenize_entity(record)
                for token in tokens:
                    if token in entity_index:
                        current_candidates = entity_index[token]
                        for candidate_id in current_candidates:
                            if self._flags[candidate_id] != i:
                                self._counters[candidate_id] = 0
                                self._flags[candidate_id] = i
                            self._counters[candidate_id] += 1
                            candidates.add(candidate_id)
                            
                self._process_candidates(candidates, i, len(tokens))
                self._progress_bar.update(1)  
        else:
            if reverse_order:
                dataset = self.data.dataset_1
                entities = self.data.entities_d1
                num_of_entities = self.data.num_of_entities_1
                attr = self.attributes_1
            else:
                dataset = self.data.dataset_2
                entities = self.data.entities_d2
                num_of_entities = self.data.num_of_entities_2
                attr = self.attributes_2
                
            for i in range(0, num_of_entities):
                candidates = set()
                record = dataset.iloc[i, attr] \
                            if attr else entities.iloc[i]
                tokens = self._tokenize_entity(record)
                entity_id = i if reverse_order else i+self.data.dataset_limit
                for token in tokens:
                    if token in entity_index:
                        current_candidates = entity_index[token]
                        for candidate_id in current_candidates:
                            if self._flags[candidate_id] != entity_id:
                                self._counters[candidate_id] = 0
                                self._flags[candidate_id] = entity_id
                            self._counters[candidate_id] += 1
                            candidates.add(candidate_id)
                self._process_candidates(candidates, entity_id, len(tokens))
                self._progress_bar.update(1)
        self._progress_bar.close()
        return self.pairs
    
    def _calc_similarity(self, common_tokens: int, source_frequency: int, tokens_size: int) -> float:
        if self.metric == 'cosine':
            return common_tokens / math.sqrt(source_frequency*tokens_size)
        elif self.metric == 'dice':
            return 2 * common_tokens / (source_frequency+tokens_size)
        elif self.metric == 'jaccard':
            return common_tokens / (source_frequency+tokens_size-common_tokens)        
    
    def _create_entity_index_d1(self):
        entity_index_d1 = {}
        for i in range(0, self.data.num_of_entities_1, 1):
            record = self.data.dataset_1.iloc[i, self.attributes_1]  \
                            if self.attributes_1 else self.data.entities_d1.iloc[i]
            tokens = self._tokenize_entity(record)
            for token in tokens:
                entity_index_d1.setdefault(token, set())
                entity_index_d1[token].add(i)
            self._source_frequency[i] = len(tokens)
            self._progress_bar.update(1)
        return entity_index_d1
    
    def _create_entity_index_d2(self):
        entity_index = {}
        for i in range(0, self.data.num_of_entities_2, 1):
            record = self.data.dataset_2.iloc[i, self.attributes_2]  \
                            if self.attributes_2 else self.data.entities_d2.iloc[i]
            tokens = self._tokenize_entity(record)
            for token in tokens:
                entity_index.setdefault(token, set())
                entity_index[token].add(i)
            self._source_frequency[i] = len(tokens)
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
            
class SchemaAgnosticΕJoin(AbstractJoin):
    '''
    SchemaAgnosticΕJoin
    '''
    
    _method_name = "Schema Agnostic Join"

    def __init__(
        self, 
        threshold: float = 0.82, 
        metric: str = 'cosine', 
        tokenization: str = 'qgrams', 
        qgrams: int = 2
    ) -> None:
        super().__init__(metric, tokenization, qgrams, threshold)
    
    def _process_candidates(self, candidates: set, entity_id: int, tokens_size: int) -> None:
        for candidate_id in candidates:
            self._insert_to_graph(
                candidate_id+self.data.dataset_limit if self.reverse_order and not self.data.is_dirty_er \
                                else candidate_id,
                entity_id, 
                self._calc_similarity(
                    self._counters[candidate_id], 
                    self._source_frequency[candidate_id],
                    tokens_size
                )
            )

        
class TopKSchemaAgnosticJoin(AbstractJoin):
    '''
    TopKSchemaAgnosticJoin
    '''
    
    _method_name = "Top-K Schema Agnostic Join"

    def __init__(
        self, K: int, 
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
                candidate_id,
                entity_id,
                self._calc_similarity(
                    self._counters[candidate_id], 
                    self._source_frequency[candidate_id],
                    tokens_size
                )
            )
