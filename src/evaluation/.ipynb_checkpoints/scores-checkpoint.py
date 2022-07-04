'''
TODO info
'''

import logging
import os
import sys

import pandas as pd
import nltk
import numpy as np
import tqdm
from tqdm import tqdm

from typing import Dict, List, Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datamodel import Block, Data
from blocks.utils import drop_single_entity_blocks

class Evaluation:

    def __init__(self) -> None:
        self.f1: float
        self.recall: float
        self.precision: float
        self.accuracy: float
        self.num_of_comparisons: int
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_matching_pairs = 0
        self.data: Data
    
    def report(self, prediction: any, data: Data) -> None:
        
        self.data = data
        gt = self.data.ground_truth

        if isinstance(prediction, dict) and isinstance(list(prediction.values())[0], set):
            self.total_matching_pairs = sum([len(block) for block in prediction.values()])
            for _, (id1, id2) in gt.iterrows():
                if (id1 in prediction and id2 in prediction[id1]) or   \
                    (id2 in prediction and id1 in prediction[id2]):
                    self.true_positives += 1
                else:
                    self.false_negatives += 1
        else:
            entity_index: dict = self._create_entity_index(prediction)

            for _, (id1, id2) in gt.iterrows():
                if id1 in entity_index and    \
                    id2 in entity_index and     \
                        self._are_matching(entity_index, id1, id2):
                    self.true_positives += 1
                else:
                    self.false_negatives += 1

        self.false_positives = self.total_matching_pairs - self.true_positives
        self.precision = self.true_positives / self.total_matching_pairs
        self.recall = self.true_positives / len(gt)
        self.f1 = 2*((self.precision*self.recall)/(self.precision+self.recall))
        
        print("+----------+\n Evaluation\n+----------+\nPrecision: {:9.2f}% \nRecall:    {:9.2f}%\nF1-score:  {:9.2f}%".format(
            self.precision*100, self.recall*100, self.f1*100)
        )

    def _create_entity_index(self, groups: any) -> dict:
        
        if len(groups) < 1:
            print("error")
            # TODO: error
        
        if isinstance(groups, list):
            return self._create_entity_index_from_clusters(groups)
        elif 'Block' in str(type(list(groups.values())[0])):
            return self._create_entity_index_from_blocks(groups)
        else:
            print("Not supported type")
            # TODO: error
    
    
    def _create_entity_index_from_clusters(self, clusters: list) -> dict:
       
        entity_index = dict()
        for cluster, cluster_id in zip(clusters, range(0, len(clusters))):
            cluster_entities_d1 = 0
            cluster_entities_d2 = 0
            for id in cluster:
                entity_index[id] = cluster_id

                if not self.data.is_dirty_er:
                    if id < self.data.dataset_limit:
                        cluster_entities_d1 += 1
                    else:
                        cluster_entities_d2 += 1

            if self.data.is_dirty_er:
                self.total_matching_pairs += len(cluster)*(len(cluster)-1)/2
            else:
                self.total_matching_pairs += cluster_entities_d1*cluster_entities_d2
                    
        return entity_index
    
    def _create_entity_index_from_blocks(self, blocks: dict) -> dict:
        
        entity_index = dict()
        for block_id, block in blocks.items():
            block_entities_d1 = 0
            block_entities_d2 = 0
            
            for id in block.entities_D1:
                entity_index.setdefault(id, set())
                entity_index[id] = block_id
                
            if not self.data.is_dirty_er:
                for id in block.entities_D2:
                    entity_index.setdefault(id, set())
                    entity_index[id] = block_id
                    
            if self.data.is_dirty_er:
                self.total_matching_pairs += len(block.entities_D1)*(len(block.entities_D1)-1)/2
            else:
                self.total_matching_pairs += len(block.entities_D1)*len(block.entities_D2)

        return entity_index
    
    
    def _are_matching(self, entity_index, id1, id2) -> bool:
        if len(entity_index) < 1:
            print("error") # TODO: error
            return None
        
        return True if (isinstance([entity_index.values()][0], set) and \
                        entity_index[id1].intersection(entity_index[id2]) > 0) or \
                        entity_index[id1] == entity_index[id2] else False