'''
Blocking methods
---

One block is consisted of 1 set if Dirty ER and
2 sets if Clean-Clean ER.

TODO: Change dict instertion like cleaning or use method insert_to_dict
TODO: ids to CC as 0...n-1 and n..m can be merged in one set, no need of 2 sets?
'''
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from operator import methodcaller
import os
import sys

import pandas as pd
import nltk
import numpy as np
# nltk.download('punkt')
import tqdm
from tqdm import tqdm

import math

from typing import Dict, List, Callable

info = logging.info
error = logging.error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datamodel import Block, Data
from blocks.utils import drop_single_entity_blocks


class AbstractBlockBuilding:
    '''
    Abstract class for the block building method
    '''

    _method_name: str
    _method_info: str

    def __init__(self) -> any:
        self.blocks: dict

    def build_blocks(
            self, data: Data,
            attributes_1: list=None,
            attributes_2: list=None,
    ) -> dict:
        '''
        Main method of Standard Blocking
        ---
        Input: Dirty/Clean-1 dataframe, Clean-2 dataframe
        Returns: dict of token -> Block
        '''
        self.blocks: dict = dict()
        self.attributes_1 = attributes_1
        self.attributes_2 = attributes_2
        if data.is_dirty_er:
            tqdm_desc_1 = self._method_name + " - Dirty ER"
        else:
            tqdm_desc_1 = self._method_name + " - Clean-Clean ER (1)"
            tqdm_desc_2 = self._method_name + " - Clean-Clean ER (2)"
            

        for i in tqdm(range(0, data.num_of_entities_1, 1), desc=tqdm_desc_1):
            record = data.dataset_1.iloc[i, attributes_1] if attributes_1 else data.entities_d1.iloc[i] 
            for token in self._tokenize_entity(record):
                self.blocks.setdefault(token, Block())
                self.blocks[token].entities_D1.add(i)
        if not data.is_dirty_er:
            for i in tqdm(range(0, data.num_of_entities_2, 1), desc=tqdm_desc_2):
                record = data.dataset_2.iloc[i, attributes_2] if attributes_2 else data.entities_d2.iloc[i]
                for token in self._tokenize_entity(record):
                    self.blocks.setdefault(token, Block())
                    self.blocks[token].entities_D2.add(data.dataset_limit+i)
        self.blocks = drop_single_entity_blocks(self.blocks, data.is_dirty_er)

        return self.blocks

    def _tokenize_entity(self, entity: str) -> list:
        pass

    def __str__(self) -> str:
        pass

class StandardBlocking(AbstractBlockBuilding):
    '''
    Standard Blocking
    ---
    Creates one block for every token in the attribute values of at least two entities.
    '''

    _method_name = "Standard Blocking"
    _method_info = _method_name + ": it creates one block for every token in the attribute \
                                    values of at least two entities."

    def __init__(self) -> any:
        super().__init__()

    def _tokenize_entity(self, entity) -> list:
        return entity.split()


class QGramsBlocking(AbstractBlockBuilding):
    '''
    Q-Grams Blocking
    ---
    Creates one block for every q-gram that is extracted from any token in the attribute \
    values of any entity. The q-gram must be shared by at least two entities.
    '''

    _method_name = "Q-Grams Blocking"
    _method_info = _method_name + ": it creates one block for every q-gram that is extracted \
                from any token in the attribute values of any entity.\n" + \
                "The q-gram must be shared by at least two entities."

    def __init__(
            self,
            qgrams=None,
    ) -> any:
        super().__init__()

        self.qgrams = qgrams

    def _tokenize_entity(self, entity) -> list:
        return [' '.join(grams) for grams in nltk.ngrams(entity, n=self.qgrams)]


class SuffixArraysBlocking(AbstractBlockBuilding):
        
    _method_name = "Suffix Arrays Blocking"
    _method_info = _method_name + ": it creates one block for every suffix that appears in the attribute value tokens of at least two entities."

    def __init__(
            self, suffix_length: int = 3,
    ) -> any:
        super().__init__()

        self.suffix_length = suffix_length

    def _tokenize_entity(self, entity) -> list:
        return [word[:self.suffix_length] if len(word) > self.suffix_length else word for word in entity.split()]

    
class ExtendedSuffixArraysBlocking(SuffixArraysBlocking):
    _method_name = "Extended Suffix Arrays Blocking"
    _method_info = _method_name + ": it creates one block for every substring (not just suffix) that appears in the tokens of at least two entities."

    def __init__(
            self, suffix_length: int = 3,
    ) -> any:
        super().__init__(suffix_length)
        self.suffix_length = suffix_length


    def _tokenize_entity(self, entity) -> list:
        tokens = []
        for word in entity.split():
            if len(word) > self.suffix_length:
                for token in list(nltk.ngrams(word,n=self.suffix_length)):
                    tokens.append("".join(token))
            else:
                tokens.append("".join(word))
        return tokens

class ExtendedQGramsBlocking(QGramsBlocking):
    
    _method_name = "Extended Suffix Arrays Blocking"
    _method_info = _method_name + ": it creates one block for every substring (not just suffix) that appears in the tokens of at least two entities."
    
    def __init__(
        self, qgrams: int, threshold: float = 0.95
    ) -> any:
        super().__init__(qgrams)
        self.threshold: float = threshold
        self.MAX_QGRAMS: int = 15

    def _tokenize_entity(self, entity) -> list:
        tokens = []
        for word in entity.split():
            qgrams = [''.join(qgram) for qgram in nltk.ngrams(word, n=self.qgrams)]
            if len(qgrams) == 1:
                tokens += qgrams
            else:
                if len(qgrams) > self.MAX_QGRAMS:
                    qgrams = qgrams[:self.MAX_QGRAMS]

                minimum_length = math.floor(len(qgrams) * self.threshold)

                for i in range(minimum_length, len(qgrams)):
                    tokens += self._qgrams_combinations(qgrams, i)
        
        return tokens
    
    def _qgrams_combinations(self, sublists: list, sublist_length: int) -> list:
        
        if not sublists or len(sublists) < sublist_length:
            return []
        
        remaining_elements = sublists.copy()
        last_sublist = remaining_elements.pop(len(sublists)-1)
        combinations_exclusive_x = self._qgrams_combinations(remaining_elements, sublist_length)
        combinations_inclusive_x = self._qgrams_combinations(remaining_elements, sublist_length-1)
        
        resulting_combinations = combinations_exclusive_x.copy()
        if not resulting_combinations:
            resulting_combinations.append(last_sublist)
        else:
            for combination in combinations_inclusive_x:
                resulting_combinations.append(combination+last_sublist)
            
        return resulting_combinations

class LSHSuperBitBlocking(AbstractBlockBuilding):
    pass


class LSHMinHashBlocking(LSHSuperBitBlocking):
    pass
