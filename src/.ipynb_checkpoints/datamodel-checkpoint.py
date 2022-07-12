import colorama
from colorama import Fore
import logging
from typing import Dict
import pandas as pd
import sys, os

class Data:

    def __init__(
            self, 
            dataset_1: pd.DataFrame,
            attributes_1: list,
            id_column_name_1: str,
            dataset_2: pd.DataFrame=None,
            attributes_2: list=None,
            id_column_name_2: str=None,
            ground_truth: pd.DataFrame=None,
            with_header: bool=None
    ) -> None:
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        if dataset_2 is not None and (id_column_name_2 is None or attributes_2 is None):
            print("ERROR")
            # TODO: error
        self.entities_d1: pd.DataFrame
        self.entities_d2: pd.DataFrame = None
        self.ground_truth = ground_truth.astype(str)
        self.is_dirty_er = True if dataset_2 is None else False
        self.dataset_limit = self.num_of_entities_1 = len(dataset_1)
        self.num_of_entities_2: int = len(dataset_2) if dataset_2 is not None else 0
        self.num_of_entities: int = self.num_of_entities_1 + self.num_of_entities_2
        self.attributes_1: list = attributes_1
        # self.attributes_1.remove(id_column_name_1)
        
        if dataset_2 is not None: self.attributes_2: list = attributes_2
            # self.attributes_2.remove(id_column_name_2)

        self.entities: pd.DataFrame
        self.id_column_name_1 = id_column_name_1
        self.id_column_name_2 = id_column_name_2

    def process(self, text_cleaning_method=None) -> None:
        
        self.entities = self.dataset_1 = self.dataset_1.astype(str).apply(text_cleaning_method)
        self.entities_d1 = self.dataset_1[self.attributes_1].apply(" ".join, axis=1)
        
        if not self.is_dirty_er:
            self.dataset_2 = self.dataset_2.astype(str).apply(text_cleaning_method)
            self.entities_d2 = self.dataset_2[self.attributes_2].apply(" ".join, axis=1)
            self.entities = pd.concat([self.dataset_1, self.dataset_2])
        self._create_gt_mapping()
        
    def _create_gt_mapping(self) -> None:
        
        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.astype(str)

        self._ids_mapping_1 = dict(zip(
            self.dataset_1[self.id_column_name_1].tolist(), range(0, self.num_of_entities_1)
        ))
        
        if not self.is_dirty_er:
            self._ids_mapping_2 = dict(zip(
                self.dataset_2[self.id_column_name_2].tolist(), range(self.num_of_entities_1, self.num_of_entities_1+self.num_of_entities_2)
            ))      

    def print_specs(self):
        print("Type of Entity Resolution: ", "Dirty" if self.is_dirty_er else "Clean-Clean" )
        print("Number of entities in D1: ", self.num_of_entities_1)
        print("Attributes provided  for D1: ", self.attributes_1)
        if not self.is_dirty_er: 
            print("\nNumber of entities in D2: ", self.num_of_entities_2)
            print("Attributes provided  for D2: ", self.attributes_2)
        print("\nTotal number of entities: ", self.num_of_entities)
        
        if self.ground_truth is not None:
            print("Number of matching pairs in ground-truth: ", len(self.ground_truth))
        
        
class Block:
    '''
    Block entity
    ---
    Consists of 2 sets of profile entities (1 for Dirty ER and 2 for Clean-Clean ER)
    '''

    def __init__(self) -> None:
        self.entities_D1: set = set()
        self.entities_D2: set = set()

    def get_cardinality(self, is_dirty_er) -> int:
        if is_dirty_er:
            return len(self.entities_D1)*(len(self.entities_D1)-1)/2
        return len(self.entities_D1) * len(self.entities_D2)

    def get_size(self) -> int:
        return len(self.entities_D1) + len(self.entities_D2)

    def verbose(self, key, is_dirty_er):
        print("\nBlock ", "\033[1;32m"+key+"\033[0m", " contains entities with ids: ")
        if is_dirty_er:
            print("Dirty dataset: " + "[\033[1;34m" + \
             str(len(self.entities_D1)) + " entities\033[0m]")
            print(self.entities_D1)
        else:
            print("Clean dataset 1: " + "[\033[1;34m" + \
             str(len(self.entities_D1)) + " entities\033[0m]")
            print(self.entities_D1)
            print("Clean dataset 2: " + "[\033[1;34m" + str(len(self.entities_D2)) + \
            " entities\033[0m]")
            print(self.entities_D2)
