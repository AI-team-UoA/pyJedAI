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
    _is_dirty_er: bool
    blocks: dict = dict()

    def __init__(self) -> any:
        self._num_of_entities_2 = None

    def build_blocks(
            self,
            data: Data
    ) -> dict:
        '''
        Main method of Standard Blocking
        ---
        Input: Dirty/Clean-1 dataframe, Clean-2 dataframe
        Returns: dict of token -> Block
        '''
        
        if not data.is_dirty_er:
            tqdm_desc_1 = self._method_name + " - Clean-Clean ER (1)"
            tqdm_desc_2 = self._method_name + " - Clean-Clean ER (2)"
        else:
            tqdm_desc_1 = self._method_name + " - Dirty ER"

        for i in tqdm(range(0, data.num_of_entities_1, 1), desc=tqdm_desc_1):
            record = data.entities_d1[i]
            for token in self._tokenize_entity(record):
                self.blocks.setdefault(token, Block())
                self.blocks[token].entities_D1.add(i)

        if not data.is_dirty_er:
            for i in tqdm(range(0, data.num_of_entities_2, 1), desc=tqdm_desc_2):
                record = data.entities_d2[i]
                for token in self._tokenize_entity(record):
                    self.blocks.setdefault(token, Block())
                    self.blocks[token].entities_D2.add(data.num_of_entities_1+i)

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
        return nltk.word_tokenize(entity)


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
    pass

class ExtendedSuffixArraysBlocking(AbstractBlockBuilding):
    pass

class ExtendedQGramsBlocking(AbstractBlockBuilding):
    pass

class LSHSuperBitBlocking(AbstractBlockBuilding):
    pass


class LSHMinHashBlocking(LSHSuperBitBlocking):
    pass
