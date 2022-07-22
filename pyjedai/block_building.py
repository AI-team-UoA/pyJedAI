'''
Blocking methods
---
One block is consisted of 1 set if Dirty ER and
2 sets if Clean-Clean ER.
'''
import nltk
import math
import re
import time

from tqdm.notebook import tqdm

# pyJedAI
from .datamodel import Block, Data
from .utils import drop_big_blocks_by_size, drop_single_entity_blocks

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
        start_time = time.time()
        
        self.blocks: dict = dict()
        self.attributes_1 = attributes_1
        self.attributes_2 = attributes_2
        self._progress_bar = tqdm(total=data.num_of_entities, desc=self._method_name)        
        for i in range(0, data.num_of_entities_1, 1):
            record = data.dataset_1.iloc[i, attributes_1] if attributes_1 else data.entities_d1.iloc[i]
            for token in self._tokenize_entity(record):
                self.blocks.setdefault(token, Block())
                self.blocks[token].entities_D1.add(i)
            self._progress_bar.update(1)
        if not data.is_dirty_er:
            for i in range(0, data.num_of_entities_2, 1):
                record = data.dataset_2.iloc[i, attributes_2] if attributes_2 else data.entities_d2.iloc[i]
                for token in self._tokenize_entity(record):
                    self.blocks.setdefault(token, Block())
                    self.blocks[token].entities_D2.add(data.dataset_limit+i)
                self._progress_bar.update(1)
                
        self.blocks = drop_single_entity_blocks(self.blocks, data.is_dirty_er)
        self.blocks = self._clean_blocks(self.blocks)
        
        self.execution_time = time.time() - start_time
        self._progress_bar.close()
        
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

    def _tokenize_entity(self, entity) -> set:
        return set(filter(None, re.split('[\\W_]', entity.lower())))

    def _clean_blocks(self, blocks: dict) -> dict:
        return blocks

class QGramsBlocking(StandardBlocking):
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
            self, qgrams: int = 6,
    ) -> any:
        super().__init__()
        self.qgrams = qgrams

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            if len(token) < self.qgrams:
                keys.add(token)
            else:
                keys.update(''.join(qg) for qg in nltk.ngrams(token, n=self.qgrams))
        return keys
    
    def _clean_blocks(self, blocks: dict) -> dict:
        return blocks


class SuffixArraysBlocking(StandardBlocking):
    '''
    Suffix Arrays Blocking
    ---
    It creates one block for every suffix that appears in the attribute value tokens of at least two entities.
    '''
    _method_name = "Suffix Arrays Blocking"
    _method_info = _method_name + ": it creates one block for every suffix that appears in the attribute value tokens of at least two entities."

    def __init__(
            self, suffix_length: int = 6, max_block_size : int = 53
    ) -> any:
        super().__init__()
        self.suffix_length = suffix_length
        self.max_block_size = max_block_size
        
    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            if len(token) < self.suffix_length:
                keys.add(token)
            else:
                for length in range(0, len(token) - self.suffix_length + 1):
                    keys.add(token[length:])
        return keys
    
    def _clean_blocks(self, blocks: dict) -> dict:
        return drop_big_blocks_by_size(blocks, self.max_block_size)

    
class ExtendedSuffixArraysBlocking(StandardBlocking):
    '''
    Extended Suffix Arrays Blocking
    ---
    It creates one block for every substring (not just suffix) that appears in the tokens of at least two entities..
    '''
    _method_name = "Extended Suffix Arrays Blocking"
    _method_info = _method_name + ": it creates one block for every substring (not just suffix) that appears in the tokens of at least two entities."

    def __init__(
            self, suffix_length: int = 6, max_block_size : int = 39
    ) -> any:
        super().__init__()
        self.suffix_length = suffix_length
        self.max_block_size = max_block_size

    def _tokenize_entity(self, entity) -> set:
       keys = set()
       for token in super()._tokenize_entity(entity):
           keys.add(token)
           if len(token) > self.suffix_length:
                for current_size in range(self.suffix_length, len(token)): 
                    for letters in list(nltk.ngrams(token, n=current_size)):
                        keys.add("".join(letters))
       return keys
    
    def _clean_blocks(self, blocks: dict) -> dict:
        return drop_big_blocks_by_size(blocks, self.max_block_size)
    
class ExtendedQGramsBlocking(StandardBlocking):
    '''
    Extended Q-Grams Blocking
    ---
    It creates one block for every combination of q-grams that represents at least two entities.
    The q-grams are extracted from any token in the attribute values of any entity.
    '''
    _method_name = "Extended QGramsBlocking"
    _method_info = _method_name + ": it creates one block for every substring (not just suffix) that        appears in the tokens of at least two entities."
    
    def __init__(
        self, qgrams: int = 6, threshold: float = 0.95
    ) -> any:
        super().__init__()
        self.threshold: float = threshold
        self.MAX_QGRAMS: int = 15
        self.qgrams = qgrams

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            if len(token) < self.qgrams:
                keys.add(token)
            else:    
                qgrams = [''.join(qgram) for qgram in nltk.ngrams(token, n=self.qgrams)]
                if len(qgrams) == 1:
                    keys.update(qgrams)
                else:
                    if len(qgrams) > self.MAX_QGRAMS:
                        qgrams = qgrams[:self.MAX_QGRAMS]

                    minimum_length = max(1, math.floor(len(qgrams) * self.threshold))
                    for i in range(minimum_length, len(qgrams) + 1):
                        keys.update(self._qgrams_combinations(qgrams, i))
        
        return keys
    
    def _qgrams_combinations(self, sublists: list, sublist_length: int) -> list:
        if sublist_length == 0 or len(sublists) < sublist_length:
            return []
        
        remaining_elements = sublists.copy()
        last_sublist = remaining_elements.pop(len(sublists)-1)
        
        combinations_exclusive_x = self._qgrams_combinations(remaining_elements, sublist_length)
        combinations_inclusive_x = self._qgrams_combinations(remaining_elements, sublist_length-1)
        
        resulting_combinations = combinations_exclusive_x.copy() if combinations_exclusive_x else []

        if not combinations_inclusive_x: # is empty
            resulting_combinations.append(last_sublist)
        else:
            for combination in combinations_inclusive_x:
                resulting_combinations.append(combination+last_sublist)
            
        return resulting_combinations
    
    def _clean_blocks(self, blocks: dict) -> dict:
        return blocks