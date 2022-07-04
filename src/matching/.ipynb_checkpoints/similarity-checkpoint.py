'''
TODO
'''

from html import entities
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


import gensim
from gensim import corpora
from pprint import pprint

import pandas as pd
import tqdm
from tqdm import tqdm
import networkx
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import core
from datamodel import Block, Data
from blocks.utils import drop_single_entity_blocks, create_entity_index, print_blocks
from utils.constants import EMBEDING_TYPES


class EntityMatching:
    '''
    TODO
    '''
    _method_name: str = "Entity Matching"
    _method_info: str = ": Calculates similarity from 0. to 1. for all blocks"

    def __init__(self, metric: str, qgram: int = 2, embedings: str = None, attributes: list = None, similarity_threshold: float = None) -> None:
        self.data: Data
        self.pairs: networkx.Graph
        self.metric = metric
        self.qgram: int = 2
        self.embedings: str = embedings
        self.attributes: list = attributes
        self.similarity_threshold = similarity_threshold
        self._progress_bar: tqdm

        if self.metric == 'levenshtein' or self.metric == 'edit_distance':
            self._metric = Levenshtein().distance
        elif self.metric == 'nlevenshtein':
            self._metric = NormalizedLevenshtein().distance
        elif self.metric == 'jaro_winkler':
            self._metric = JaroWinkler().distance
        elif self.metric == 'metric_lcs':
            self._metric = MetricLCS().distance
        elif self.metric == 'qgram':
            self._metric = NGram(self.qgram).distance
        # elif self.metric == 'cosine':
        #     cosine = Cosine(self.qgram)
        #     self._metric = cosine.similarity_profiles(cosine.get_profile(entity_1), cosine.get_profile(entity_2))
        elif self.metric == 'jaccard':
            self._metric = Jaccard(self.qgram).distance
        elif self.metric == 'sorensen_dice':
            self._metric = SorensenDice().distance
        elif self.metric == 'overlap_coefficient':
            self._metric = OverlapCoefficient().distance

    def predict(self, blocks: dict, data: Data) -> networkx.Graph:
        '''
        TODO
        '''
        if len(blocks) == 0:
            # TODO: Error
            return None
        self.data = data
        self.pairs = networkx.Graph()
        all_blocks = list(blocks.values())
        self._progress_bar = tqdm(total=len(blocks), desc=self._method_name+" ("+self.metric+")") 
        
        if 'Block' in str(type(all_blocks[0])):
            self._predict_raw_blocks(blocks)
        elif isinstance(all_blocks[0], set):
            self._predict_prunned_blocks(blocks)
        else:
            # TODO: Error
            print("Error")

        # if self.embedings in EMBEDING_TYPES:
        # TODO: Add GENSIM

        return self.pairs

    def _predict_raw_blocks(self, blocks: dict) -> None:
        '''
        TODO comment
        '''
        if self.data.is_dirty_er:
            for _, block in blocks.items():
                entities_array = list(block.entities_D1)
                for entity_id1 in range(0, len(entities_array), 1):
                    for entity_id2 in range(entity_id1+1, len(entities_array), 1):
                        similarity = self._similarity(
                            entity_id1, entity_id2
                        )
                        self._insert_to_graph(entity_id1, entity_id2, similarity)
                self._progress_bar.update(1)
        else:
            for _, block in blocks.items():
                for entity_id1 in block.entities_D1:
                    for entity_id2 in block.entities_D2:
                        similarity = self._similarity(
                            entity_id1, entity_id2
                        )
                        self._insert_to_graph(entity_id1, entity_id2, similarity)
                self._progress_bar.update(1)

    def _predict_prunned_blocks(self, blocks: dict) -> None:
        '''
        TODO comment
        '''
        for entity_id, candidates in blocks.items():
            for candidate_id in candidates:
                similarity = self._similarity(
                    entity_id, candidate_id
                )
                self._insert_to_graph(entity_id, candidate_id, similarity)
            self._progress_bar.update(1)

    def _insert_to_graph(self, entity_id1, entity_id2, similarity):
        if self.similarity_threshold is None or \
            (self.similarity_threshold and similarity > self.similarity_threshold):
            self.pairs.add_edge(entity_id1, entity_id2, weight=similarity)

    def _similarity(self, entity_id1: int, entity_id2: int) -> float:

        similarity: float = 0.0

        if isinstance(self.attributes, dict):
            for attribute, weight in self.attributes.items():
                similarity += weight*self._metric(
                    self.data.entities.iloc[entity_id1][attribute],
                    self.data.entities.iloc[entity_id2][attribute]
                )
        if isinstance(self.attributes, list):
            for attribute in self.attributes:
                similarity += self._metric(
                    self.data.entities.iloc[entity_id1][attribute],
                    self.data.entities.iloc[entity_id2][attribute]
                )
                similarity /= len(self.attributes)
        else:
            # concatenated row string
            similarity = self._metric(
                self.data.entities.iloc[entity_id1].str.cat(sep=' '),
                self.data.entities.iloc[entity_id2].str.cat(sep=' ')
            )

        return similarity