import itertools
import logging as log
import math
import re
import time
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple

import nltk
import numpy as np
from tqdm.auto import tqdm

from .datamodel import Block, Data, PYJEDAIFeature
from .utils import (are_matching, drop_big_blocks_by_size, create_entity_index,
                    drop_single_entity_blocks, get_blocks_cardinality)
from .evaluation import Evaluation

class AbstractBlockProcessing(PYJEDAIFeature):
    """Abstract class for the block building method
    """

    def __init__(self):
        super().__init__()
        self.blocks: dict
        self.attributes_1: list
        self.attributes_2: list
        self.num_of_blocks_dropped: int
        self.original_num_of_blocks: int
        self.sum_of_sizes: int = 0
        self.total_num_of_comparisons: int = 0
        self.min_block_size: int = None
        self.max_block_size: int = None
        self.min_block_comparisons: int = None
        self.max_block_comparisons: int = None
            
    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes from D1:\n\t" + ', '.join(c for c in (self.attributes_1 if self.attributes_1 is not None \
                else self.data.dataset_1.columns)) +
            ("\nAttributes from D2:\n\t" + ', '.join(c for c in (self.attributes_2 if self.attributes_2 is not None \
                else self.data.dataset_2.columns)) if not self.data.is_dirty_er else "") +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

    def evaluate(self,
                 prediction,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True,
                 with_stats: bool = False) -> any:

        if prediction is None:
            if self.blocks is None:
                raise AttributeError("Can not proceed to evaluation without build_blocks.")
            else:
                eval_blocks = self.blocks
        else:
            eval_blocks = prediction
            
        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " + 
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        true_positives = 0
        entity_index = eval_obj._create_entity_index_from_blocks(eval_blocks)
        for _, (id1, id2) in self.data.ground_truth.iterrows():
            id1 = self.data._ids_mapping_1[id1]
            id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er else self.data._ids_mapping_2[id2]
            if id1 in entity_index and    \
                id2 in entity_index and are_matching(entity_index, id1, id2):
                true_positives += 1

        total_matching_pairs = get_blocks_cardinality(eval_blocks, self.data.is_dirty_er)
        eval_obj.calculate_scores(true_positives=true_positives, total_matching_pairs=total_matching_pairs)
        eval_result = eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)
        if with_stats:
            self.stats(eval_blocks)
        return eval_result

    def stats(self, blocks: dict, verbose: bool = True) -> dict:

        # Atomic features
        self.portion_of_singleton_entites =  0
        self.portion_of_duplicate_blocks = 0 # contain the same entities
        self.num_of_block_assignments = 0
        self.num_of_minimal_blocks = 0 # one-comparison blocks
        self.num_of_blocks_per_entity = 0
        self.average_number_of_block_assignments_per_comparison = 0
        self.optimality_distance = 0
        self.entities_in_blocks = set()
        self.size_per_block = []
        self.cardinalities = []
        self.num_of_blocks = len(blocks)
        for block in blocks.values():
            self.sum_of_sizes += block.get_size()
            self.min_block_size = min(self.min_block_size, block.get_size()) if self.min_block_size else block.get_size()
            self.max_block_size = max(self.max_block_size, block.get_size()) if self.max_block_size else block.get_size()
            self.min_block_comparisons = min(self.min_block_comparisons, block.get_cardinality(self.data.is_dirty_er)) if self.min_block_comparisons else block.get_cardinality(self.data.is_dirty_er)
            self.max_block_comparisons = max(self.max_block_comparisons, block.get_cardinality(self.data.is_dirty_er)) if self.max_block_comparisons else block.get_cardinality(self.data.is_dirty_er)
            self.size_per_block.append(block.get_size())
            self.entities_in_blocks = self.entities_in_blocks.union(block.entities_D1)
            if not self.data.is_dirty_er:
                self.entities_in_blocks = self.entities_in_blocks.union(block.entities_D2)
            cardinality = block.get_cardinality(self.data.is_dirty_er)
            self.cardinalities.append(cardinality)
            if cardinality == 1:
                self.num_of_minimal_blocks += 1
                
        self.num_of_minimal_blocks /= self.num_of_blocks
        self.num_of_entities_in_blocks = len(self.entities_in_blocks)
        self.num_of_block_assignments = self.total_num_of_comparisons = sum(self.cardinalities)
        self.average_block_size = int(self.sum_of_sizes / self.num_of_blocks)
        self.size_per_block = sorted(self.size_per_block)
        self.num_of_blocks_per_entity = self.num_of_blocks / self.num_of_entities_in_blocks
        self.average_number_of_block_assignments_per_comparison = self.num_of_block_assignments / (2*self.total_num_of_comparisons)
        median = self.size_per_block[int(len(self.size_per_block)/2)]

        entity_index = create_entity_index(blocks, self.data.is_dirty_er)
        
        # Distributional features
        self.blocks_frequency = []
        self.relative_block_frequency = []
        self.comparison_frequency = []
        self.relative_comparison_frequency = []
        
        for entity in entity_index:
            if len(entity_index[entity]) == 1:
                self.portion_of_singleton_entites += 1
            self.blocks_frequency.append(len(entity_index[entity]))
            self.relative_block_frequency.append(len(entity_index[entity]) / self.num_of_blocks)
            self.comparison_frequency.append(sum([blocks[block_key].get_cardinality(self.data.is_dirty_er) for block_key in entity_index[entity]]))
            self.relative_comparison_frequency.append(sum([blocks[block_key].get_cardinality(self.data.is_dirty_er) for block_key in entity_index[entity]]) / self.total_num_of_comparisons)
        
        self.portion_of_singleton_entites /= self.num_of_entities_in_blocks
        self.portion_of_minimal_blocks = self.num_of_minimal_blocks / self.num_of_blocks
        
        # Distributional features
        self.average_blocks_per_entity = np.mean(self.blocks_frequency)
        self.average_number_of_block_assignments_per_entity = np.mean(self.relative_block_frequency)
        self.average_comparison_per_entity = np.mean(self.comparison_frequency)
        self.average_relative_number_of_comparisons_per_entity = np.mean(self.relative_comparison_frequency)

        self.entropy_of_blocks_per_entity = -np.sum([p * np.log2(p) for p in self.blocks_frequency])
        self.entropy_of_comparison_per_entity = -np.sum([p * np.log2(p) for p in self.comparison_frequency])
        
        self.kurtosis_of_blocks_per_entity = np.sum([(p - self.average_blocks_per_entity)**4 for p in self.blocks_frequency]) /\
                                                    (self.num_of_blocks * self.average_blocks_per_entity**4)
        self.kurtosis_of_comparison_per_entity = np.sum([(p - self.average_comparison_per_entity)**4 for p in self.comparison_frequency]) /\
                                                    (self.num_of_blocks * self.average_comparison_per_entity**4)
        
        self.skewness_of_blocks_per_entity = np.sum([(p - self.average_blocks_per_entity)**3 for p in self.blocks_frequency]) /\
                                                (self.num_of_blocks * self.average_blocks_per_entity**3)
        self.skewness_of_comparison_per_entity = np.sum([(p - self.average_comparison_per_entity)**3 for p in self.comparison_frequency]) /\
                                                (self.num_of_blocks * self.average_comparison_per_entity**3)
        
        
        if verbose:
            print(
                "Statistics:" +
                "\n\tNumber of blocks: " + str(self.num_of_blocks) +
                "\n\tAverage block size: " + str(self.average_block_size) +
                "\n\tMedian block size: " + str(median) +
                "\n\tMax block size: " + str(self.max_block_size) +
                "\n\tMin block size: " + str(self.min_block_size) +
                "\n\tNumber of blocks dropped: " + str(self.num_of_blocks_dropped) +
                "\n\tNumber of comparisons: " + str(self.total_num_of_comparisons) +
                "\n\tMax comparisons per block: " + str(self.max_block_comparisons) +
                "\n\tMin comparisons per block: " + str(self.min_block_comparisons) +
                "\n\tEntities in blocks: " + str(len(self.entities_in_blocks))
            )
            print(u'\u2500' * 123)
            print(
                "\tAtomic feautures" +
                "\n\t\tNumber of entities in blocks: " + str(self.num_of_entities_in_blocks) +
                "\n\t\tNumber of blocks: " + str(self.num_of_blocks) +
                "\n\t\tPortion of singleton entities: " + str(self.portion_of_singleton_entites) +
                "\n\t\tTotal number of comparisons: " + str(self.total_num_of_comparisons) +
                "\n\t\tNumber of blocks: " + str(self.num_of_blocks) +
                "\n\t\tNumber of block assignments: " + str(self.num_of_block_assignments) +
                "\n\t\tPortion of minimal blocks: " + str(self.portion_of_minimal_blocks) +
                "\n\t\tNumber of blocks per entity: " + str(self.num_of_blocks_per_entity) +
                "\n\t\tAverage number of block assignments per comparison: " + str(self.average_number_of_block_assignments_per_comparison)                
            )
            print(u'\u2500' * 123)
            print(
                "\tDistributional feautures" +
                "\n\t\tAverage blocks per entity: " + str(self.average_blocks_per_entity) +
                "\n\t\tAverage number of block assignments per entity: " + str(self.average_number_of_block_assignments_per_entity) +
                "\n\t\tAverage comparison per entity: " + str(self.average_comparison_per_entity) +
                "\n\t\tAverage relative number of comparisons per entity: " + str(self.average_relative_number_of_comparisons_per_entity) +
                "\n\t\tEntropy of blocks per entity: " + str(self.entropy_of_blocks_per_entity) +
                "\n\t\tEntropy of comparison per entity: " + str(self.entropy_of_comparison_per_entity) +
                "\n\t\tKurtosis of blocks per entity: " + str(self.kurtosis_of_blocks_per_entity) +
                "\n\t\tKurtosis of comparison per entity: " + str(self.kurtosis_of_comparison_per_entity) +
                "\n\t\tSkewness of blocks per entity: " + str(self.skewness_of_blocks_per_entity) +
                "\n\t\tSkewness of comparison per entity: " + str(self.skewness_of_comparison_per_entity)
            )
            print(u'\u2500' * 123)
        
        return {
            'num_of_blocks': self.num_of_blocks,
            'average_block_size': self.average_block_size,
            'median_block_size': median,
            'max_block_size': self.max_block_size,
            'min_block_size': self.min_block_size,
            'num_of_blocks_dropped': self.num_of_blocks_dropped,
            'total_num_of_comparisons': self.total_num_of_comparisons,
            'max_block_comparisons': self.max_block_comparisons,
            'min_block_comparisons': self.min_block_comparisons,
            'entities_in_blocks': len(self.entities_in_blocks),
            'average_blocks_per_entity': self.average_blocks_per_entity,
            'average_number_of_block_assignments_per_entity': self.average_number_of_block_assignments_per_entity,
            'average_comparison_per_entity': self.average_comparison_per_entity,
            'average_relative_number_of_comparisons_per_entity': self.average_relative_number_of_comparisons_per_entity,
            'entropy_of_blocks_per_entity': self.entropy_of_blocks_per_entity,
            'entropy_of_comparison_per_entity': self.entropy_of_comparison_per_entity,
            'kurtosis_of_blocks_per_entity': self.kurtosis_of_blocks_per_entity,
            'kurtosis_of_comparison_per_entity': self.kurtosis_of_comparison_per_entity,
            'skewness_of_blocks_per_entity': self.skewness_of_blocks_per_entity,
            'skewness_of_comparison_per_entity': self.skewness_of_comparison_per_entity
        }

    def export_to_df(self, blocks: dict, tqdm_enable:bool = False) -> pd.DataFrame:
        """Creates a dataframe for the evaluation report.

        Args:
            blocks (dict): Predicted blocks.

        Returns:
            pd.DataFrame: Dataframe with the predicted pairs (can be exported to CSV).
        """
        pairs_list = []

        is_dirty_er = self.data.is_dirty_er
        gt_to_ids_reversed_1 = self.data._gt_to_ids_reversed_1
        if not is_dirty_er:
            gt_to_ids_reversed_2 = self.data._gt_to_ids_reversed_2

        for block in tqdm(blocks.values(), desc="Exporting to DataFrame", disable=not tqdm_enable):
            if is_dirty_er:
                lblock = list(block.entities_D1)
                
                for i1 in range(len(lblock)):
                    for i2 in range(i1 + 1, len(lblock)):
                        id1 = gt_to_ids_reversed_1[lblock[i1]]
                        id2 = gt_to_ids_reversed_1[lblock[i2]]
                        pairs_list.append((id1, id2))
            else:
                for i1 in block.entities_D1:
                    for i2 in block.entities_D2:
                        id1 = gt_to_ids_reversed_1[i1]
                        id2 = gt_to_ids_reversed_2[i2]
                        pairs_list.append((id1, id2))

        pairs_df = pd.DataFrame(pairs_list, columns=['id1', 'id2'])

        return pairs_df


class AbstractBlockBuilding(AbstractBlockProcessing):
    """Abstract class for the block building method
    """

    _method_name: str
    _method_info: str
    _method_short_name: str

    def __init__(self):
        super().__init__()
        self.blocks: dict
        self._progress_bar: tqdm
        self.attributes_1: list
        self.attributes_2: list
        self.execution_time: float
        self.data: Data
        self.list_of_sizes: list = []

    def build_blocks(
            self,
            data: Data,
            attributes_1: list = None,
            attributes_2: list = None,
            tqdm_disable: bool = False
    ) -> Tuple[dict, dict]:
        """Main method of Blocking in a dataset

            Args:
                data (Data): Data module that contaiins the processed dataset
                attributes_1 (list, optional): Attribute columns of the dataset 1 \
                    that will be processed. Defaults to None. \
                    If not provided, all attributes are selected.
                attributes_2 (list, optional): Attribute columns of the dataset 2. \
                    Defaults to None. If not provided, all attributes are selected.
                tqdm_disable (bool, optional): Disables all tqdm at processing. Defaults to False.

            Returns:
                Tuple[dict, dict]: Dictionary of blocks, Dict of entity index (reversed blocks).
        """

        _start_time = time.time()
        self.data, self.attributes_1, self.attributes_2 = data, attributes_1, attributes_2
        self._progress_bar = tqdm(
            total=data.num_of_entities, desc=self._method_name, disable=tqdm_disable
        )

        # TODO Text process function can be applied in this step (.apply)
        self._entities_d1 = data.dataset_1[attributes_1 if attributes_1 else data.attributes_1] \
                            .apply(" ".join, axis=1) \
                            .apply(self._tokenize_entity) \
                            .values.tolist()
                        # if attributes_1 else data.entities_d1.apply(self._tokenize_entity)

        self._all_tokens = set(itertools.chain.from_iterable(self._entities_d1))

        if not data.is_dirty_er:
            self._entities_d2 = data.dataset_2[attributes_2 if attributes_2 else data.attributes_2] \
                    .apply(" ".join, axis=1) \
                    .apply(self._tokenize_entity) \
                    .values.tolist()
            self._all_tokens.union(set(itertools.chain.from_iterable(self._entities_d2)))

        entity_id = itertools.count()
        blocks = {}
        
        for entity in self._entities_d1:
            eid = next(entity_id)
            for token in entity:
                blocks.setdefault(token, Block())
                blocks[token].entities_D1.add(eid)
            self._progress_bar.update(1)

        if not data.is_dirty_er:
            for entity in self._entities_d2:
                eid = next(entity_id)
                for token in entity:
                    blocks.setdefault(token, Block())
                    blocks[token].entities_D2.add(eid)
                self._progress_bar.update(1)

        self.original_num_of_blocks = len(blocks)
        self.blocks = self._clean_blocks(blocks)
        self.num_of_blocks_dropped = len(blocks) - len(self.blocks)
        self.execution_time = time.time() - _start_time
        self._progress_bar.close()

        return self.blocks

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes from D1:\n\t" + ', '.join(c for c in (self.attributes_1 if self.attributes_1 is not None \
                else self.data.dataset_1.columns)) +
            ("\nAttributes from D2:\n\t" + ', '.join(c for c in (self.attributes_2 if self.attributes_2 is not None \
                else self.data.dataset_2.columns)) if not self.data.is_dirty_er else "") +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

    @abstractmethod
    def _clean_blocks(self, blocks: dict) -> dict:
        pass

    @abstractmethod
    def _configuration(self) -> dict:
        pass
    
class StandardBlocking(AbstractBlockBuilding):
    """ Creates one block for every token in \
        the attribute values of at least two entities.
    """

    _method_name = "Standard Blocking"
    _method_short_name: str = "SB"
    _method_info = "Creates one block for every token in " + \
        "the attribute values of at least two entities."

    def __init__(self) -> any:
        super().__init__()

    def _tokenize_entity(self, entity: str) -> list:
        """Produces a list of workds of a given string

        Args:
            entity (str): String representation  of an entity

        Returns:
            list: List of words
        """
        return list(set(filter(None, re.split('[\\W_]', entity.lower()))))

    def _clean_blocks(self, blocks: dict) -> dict:
        """No cleaning"""
        return drop_single_entity_blocks(blocks, self.data.is_dirty_er)

    def _configuration(self) -> dict:
        """No configuration"""
        return {}

class QGramsBlocking(StandardBlocking):
    """ Creates one block for every q-gram that is extracted \
        from any token in the attribute values of any entity. \
            The q-gram must be shared by at least two entities.
    """

    _method_name = "Q-Grams Blocking"
    _method_short_name: str = "QGB"
    _method_info = "Creates one block for every q-gram that is extracted " + \
                    "from any token in the attribute values of any entity. " + \
                    "The q-gram must be shared by at least two entities."

    def __init__(
            self, qgrams: int = 6
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
        return drop_single_entity_blocks(blocks, self.data.is_dirty_er)

    def _configuration(self) -> dict:
        return {
            "Q-Gramms" : self.qgrams
        }

class SuffixArraysBlocking(StandardBlocking):
    """ It creates one block for every suffix that appears \
        in the attribute value tokens of at least two entities.
    """

    _method_name = "Suffix Arrays Blocking"
    _method_short_name: str = "SAB"
    _method_info = "Creates one block for every suffix that appears in the " + \
        "attribute value tokens of at least two entities."

    def __init__(
            self,
            suffix_length: int = 6,
            max_block_size: int = 53
    ) -> any:
        super().__init__()
        self.suffix_length, self.max_block_size = suffix_length, max_block_size

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
        return drop_big_blocks_by_size(blocks, self.max_block_size, self.data.is_dirty_er)

    def _configuration(self) -> dict:
        return {
            "Suffix length" : self.suffix_length,
            "Maximum Block Size" : self.max_block_size
        }

class ExtendedSuffixArraysBlocking(StandardBlocking):
    """ It creates one block for every substring \
        (not just suffix) that appears in the tokens of at least two entities.
    """

    _method_name = "Extended Suffix Arrays Blocking"
    _method_short_name: str = "ESAB"
    _method_info = "Creates one block for every substring (not just suffix) " + \
        "that appears in the tokens of at least two entities."

    def __init__(
            self,
            suffix_length: int = 6,
            max_block_size: int = 39
    ) -> any:
        super().__init__()
        self.suffix_length, self.max_block_size = suffix_length, max_block_size

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
        return drop_big_blocks_by_size(blocks, self.max_block_size, self.data.is_dirty_er)

    def _configuration(self) -> dict:
        return {
            "Suffix length" : self.suffix_length,
            "Maximum Block Size" : self.max_block_size
        }

class ExtendedQGramsBlocking(StandardBlocking):
    """It creates one block for every combination of q-grams that represents at least two entities.
    The q-grams are extracted from any token in the attribute values of any entity.
    """

    _method_name = "Extended QGramsBlocking"
    _method_short_name: str = "EQGB"
    _method_info = "Creates one block for every substring (not just suffix) " + \
        "that appears in the tokens of at least two entities."

    def __init__(
            self,
            qgrams: int = 6,
            threshold: float = 0.95
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
        return drop_single_entity_blocks(blocks, self.data.is_dirty_er)

    def _configuration(self) -> dict:
        return {
            "Q-Gramms" : self.qgrams,
            "Threshold" : self.threshold
        }
