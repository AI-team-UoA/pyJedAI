import colorama
from colorama import Fore
import logging
from typing import Dict
import pandas as pd
import sys, os

class Data:

    def __init__(
            self, dataset_1,
            dataset_2=None,
            ground_truth=None,
            attributes=None,
            with_header=None
        ) -> None:

        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.entities_d1: pd.DataFrame
        self.entities_d2: pd.DataFrame = None
        self.ground_truth = ground_truth
        if dataset_2 is None:
            self.is_dirty_er = True
        else:
            self.is_dirty_er = False
        self.dataset_limit: int = None
        self.num_of_entities_1: int = None
        self.num_of_entities_2: int = None
        self.num_of_entities: int = None
        self.attributes: list = attributes if attributes else dataset_1.columns.values.tolist()
        self.entities: pd.DataFrame

    def _init_entities(self) -> None:
        self.entities = self.entities_d1
        if not self.is_dirty_er:
            self.entities = pd.concat([self.entities_d1,  self.entities_d2])
    
    def process(self, text_cleaning_method=None) -> None:
        
        self.dataset_1[self.attributes] = self.dataset_1[self.attributes].apply(text_cleaning_method)
        self.dataset_1 = self.dataset_1[self.attributes] if self.attributes is not None else self.dataset_1
        if not self.is_dirty_er:
            self.dataset_2 = self.dataset_2[self.attributes] \
                                if self.attributes is not None else self.dataset_2

        self.entities_d1 = self.dataset_1.apply(" ".join, axis=1)
        self.dataset_limit = self.num_of_entities = self.num_of_entities_1 = len(self.entities_d1)
        self.entities = self.dataset_1
        
        
        if self.dataset_2 is not None:
            self.dataset_2[self.attributes] = self.dataset_2[self.attributes].apply(text_cleaning_method)

            if self.attributes:
                self.entities_d2 = self.dataset_2[self.attributes].apply(" ".join, axis=1)
            else:
                self.entities_d2 = self.dataset_2.apply(" ".join, axis=1)

            self.num_of_entities_2 = len(self.entities_d2)
            self.is_dirty_er = False
            self.num_of_entities += self.num_of_entities_2
            self.entities = pd.concat([self.dataset_1, self.dataset_2])
        else:
            self.is_dirty_er = True

    def print_specs(self):
        print("Type of Entity Resolution: ", "Dirty" if self.is_dirty_er else "Clean-Clean" )
        print("Number of entities in D1: ", self.num_of_entities_1)
        if not self.is_dirty_er:
            print("Number of entities in D1: ", self.num_of_entities_2)
        print("Total number of entities: ", self.num_of_entities)
        print("Attributes provided: ", self.dataset_1.columns.values.tolist())
        
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

    def get_size(self, is_dirty_er: bool) -> int:
        if is_dirty_er:
            return len(self.entities_D1)
        return len(self.entities_D1) + len(self.entities_D2)

    def verbose(self, is_dirty_er):
        print("\nBlock ", "\033[1;32m"+self.key+"\033[0m", " contains entities with ids: ")
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
