"""Evaluation module
This file contains all the methods for evaluating every module in pyjedai.
"""
from decimal import DivisionByZero
from typing import Type
from typing import List, Tuple
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import numpy as np

from .datamodel import Data
from .utils import are_matching
from .utils import batch_pairs
from .utils import canonical_swap
from math import inf
from .utils import PredictionData
import random
import matplotlib.pyplot as plt


class Evaluation:
    """Evaluation class. Contains multiple methods for all the fitted & predicted data.
    """
    def __init__(self, data) -> None:
        self.f1: float
        self.recall: float
        self.precision: float
        self.num_of_comparisons: int
        self.total_matching_pairs: int = 0
        self.true_positives: int
        self.true_negatives: int
        self.false_positives: int
        self.false_negatives: int
        self.num_of_true_duplicates: int
        self.data: Data = data

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " + 
                    "Data object has not been initialized with the ground-truth file")

        self.true_positives = self.true_negatives = self.false_positives = self.false_negatives = 0
        self.all_gt_ids = set(self.data._ids_mapping_1.values()) if self.data.is_dirty_er else \
                        set(self.data._ids_mapping_1.values()).union(set(self.data._ids_mapping_2.values()))

    def _set_true_positives(self, true_positives) -> None:
        self.true_positives = true_positives

    def _set_total_matching_pairs(self, total_matching_pairs) -> None:
        self.total_matching_pairs = total_matching_pairs

    def calculate_scores(self, true_positives=None, total_matching_pairs=None) -> None:
        if true_positives is not None:
            self.true_positives = true_positives
            
        if total_matching_pairs is not None:
            self.total_matching_pairs = total_matching_pairs

        if self.total_matching_pairs == 0:
            warn("Evaluation: No matches found", Warning)
            self.num_of_true_duplicates = self.false_negatives \
                = self.false_positives = self.total_matching_pairs \
                    = self.true_positives = self.true_negatives \
                        = self.recall = self.f1 = self.precision = 0
        else:
            self.num_of_true_duplicates = len(self.data.ground_truth)
            self.false_negatives = self.num_of_true_duplicates - self.true_positives
            self.false_positives = self.total_matching_pairs - self.true_positives
            cardinality = (self.data.num_of_entities_1*(self.data.num_of_entities_1-1))/2 \
                if self.data.is_dirty_er else (self.data.num_of_entities_1 * self.data.num_of_entities_2)
            self.true_negatives = cardinality - self.false_negatives - self.num_of_true_duplicates
            self.precision = self.true_positives / self.total_matching_pairs
            self.recall = self.true_positives / self.num_of_true_duplicates
            if self.precision == 0.0 or self.recall == 0.0:
                self.f1 = 0.0
            else:
                self.f1 = 2*((self.precision*self.recall)/(self.precision+self.recall))

    def report(
            self,
            configuration: dict = None,
            export_to_df=False,
            export_to_dict=False,
            with_classification_report=False,
            verbose=True
        ) -> any:

        results_dict = {
                'Precision %': self.precision*100,
                'Recall %': self.recall*100,
                'F1 %': self.f1*100,
                'True Positives': self.true_positives,
                'False Positives': self.false_positives,
                'True Negatives': self.true_negatives,
                'False Negatives': self.false_negatives
            }

        if verbose:
            if configuration:
                print('*' * 123)
                print(' ' * 40, 'Μethod: ', configuration['name'])
                print('*' * 123)
                print(
                    "Method name: " + configuration['name'] +
                    "\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in configuration['parameters'].items()]) +
                    "Runtime: {:2.4f} seconds".format(configuration['runtime'])
                )
            else:
                print(" " + (configuration['name'] if configuration else "") + " Evaluation \n---")


            print(u'\u2500' * 123)
            print("Performance:\n\tPrecision: {:9.2f}% \n\tRecall:    {:9.2f}%\n\tF1-score:  {:9.2f}%".format(self.precision*100, self.recall*100, self.f1*100))
            print(u'\u2500' * 123)
            if with_classification_report:
                print("Classification report:\n\tTrue positives: {:d}\n\tFalse positives: {:d}\n\tTrue negatives: {:d}\n\tFalse negatives: {:d}\n\tTotal comparisons: {:d}".format(
                    int(self.true_positives), int(self.false_positives), int(self.true_negatives), \
                    int(self.false_negatives), int(self.total_matching_pairs))
                )
                print(u'\u2500' * 123)
                
        if export_to_df:
            pd.set_option("display.precision", 2)
            results = pd.DataFrame.from_dict(results_dict, orient='index').T
            return results

        return results_dict

    def _create_entity_index_from_clusters(
            self,
            clusters: list,
    ) -> dict:
        entity_index = dict()
        for cluster, cluster_id in zip(clusters, range(0, len(clusters))):
            cluster_entities_d1 = 0
            cluster_entities_d2 = 0
            for entity_id in cluster.intersection(self.all_gt_ids):
                entity_index[entity_id] = cluster_id

                if not self.data.is_dirty_er:
                    if entity_id < self.data.dataset_limit:
                        cluster_entities_d1 += 1
                    else:
                        cluster_entities_d2 += 1

            if self.data.is_dirty_er:
                self.total_matching_pairs += len(cluster)*(len(cluster)-1)/2
            else:
                self.total_matching_pairs += cluster_entities_d1*cluster_entities_d2

        return entity_index

    def _create_entity_index_from_blocks(
            self,
            blocks: dict
    ) -> dict:
        entity_index = dict()
        for block_id, block in blocks.items():
            for entity_id in block.entities_D1:
                entity_index.setdefault(entity_id, set())
                entity_index[entity_id].add(block_id)

            if not self.data.is_dirty_er:
                for entity_id in block.entities_D2:
                    entity_index.setdefault(entity_id, set())
                    entity_index[entity_id].add(block_id)

            if self.data.is_dirty_er:
                self.total_matching_pairs += len(block.entities_D1)*(len(block.entities_D1)-1)/2
            else:
                self.total_matching_pairs += len(block.entities_D1)*len(block.entities_D2)

        return entity_index

    def confusion_matrix(self):
        """Generates a confusion matrix based on the classification report.
        """
        heatmap = [
            [int(self.true_positives), int(self.false_positives)],
            [int(self.false_negatives), int(self.true_negatives)]
        ]
        # plt.colorbar(heatmap)
        sns.heatmap(
            heatmap,
            annot=True,
            cmap='Blues',
            xticklabels=['Non-Matching', 'Matching'],
            yticklabels=['Non-Matching', 'Matching'],
            fmt='g'
        )
        plt.title("Confusion Matrix", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted pairs", fontsize=10, fontweight='bold')
        plt.ylabel("Real matching pairs", fontsize=10, fontweight='bold')
        plt.show()
        
    def visualize_roc(method_names : List[str], methods_data : List[Tuple[str, float, List[float]]], proportional : bool =True) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))  # set the size of the plot
        colors = []
        normalized_aucs = []
        # for each method layout its plot
        for method_name, normalized_auc, cumulative_recall in methods_data:
            x_values = range(len(cumulative_recall))
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            colors.append(color)
            normalized_aucs.append(normalized_auc)
            if proportional: sizes = [cr * 100 for cr in cumulative_recall]
            else: sizes = [10] * len(cumulative_recall)
            ax.scatter(x_values, cumulative_recall, marker='o', s=0.05, color=color, label=method_name)
            ax.plot(x_values, cumulative_recall, color=color)

        ax.set_xlabel('ec*', fontweight='bold', labelpad=10)
        ax.set_ylabel('Cumulative Recall', fontweight='bold', labelpad=10)
        ax.set_xlim(0, len(cumulative_recall))
        ax.set_ylim(0, 1)

        # add a legend showing the name of each curve and its color
        legend = ax.legend(ncol=2, loc='lower left', title='Methods', bbox_to_anchor=(0, -0.4))
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontsize(12)
        plt.setp(legend.get_lines(), linewidth=4)

        # add AUC score legend
        handles, _ = ax.get_legend_handles_labels()
        auc_legend_labels = ['AUC: {:.2f}'.format(nauc) for nauc in normalized_aucs]
        auc_legend = ax.legend(handles, auc_legend_labels, loc='lower left', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=True, title='AUC', title_fontsize=12)
        auc_legend.get_title().set_fontweight('bold')
        for i, text in enumerate(auc_legend.get_texts()):
            plt.setp(text, color=colors[i])
        ax.add_artist(legend)

        # set the figure background color to the RGB color of the solarized terminal theme
        fig.patch.set_facecolor((0.909, 0.909, 0.909))

        # adjust the margins of the figure to move the graph to the right
        fig.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)

        plt.show()

    def calculate_ideal_auc(self, pairs_num : int, true_duplicates_num : int) -> float:
        """Calculates the ideal AUC for the given number of candidate pairs
        Args:
            pairs_num (int): Total number of candidate pairs
            true_duplicates_num (int): The number of true duplicates 
        Returns:
            float: Ideal AUC
        """
        ideal_auc : float

        if(pairs_num == true_duplicates_num):
            ideal_auc = 0.5
        else:
            ideal_auc = (pairs_num % true_duplicates_num) / true_duplicates_num * 0.5
            if(pairs_num > true_duplicates_num): ideal_auc += (pairs_num - true_duplicates_num) / true_duplicates_num

        return ideal_auc

    def _till_full_tps_emission(self) -> bool:
        """Checks if emission should be stopped once all TPs have been found (TPs dict supplied)
        Returns:
            bool: Stop emission on all TPs found / Emit all pairs
        """
        return self._true_positive_checked is not None
    
    def _all_tps_emitted(self) -> bool:
        """Checks if all TPs have been emitted (Defaults to False in the case of all pairs emission approach)
        Returns:
            bool: All TPs emitted / not emitted
        """
        if(self._till_full_tps_emission()): return self._tps_found >= len(self._true_positive_checked)
        else: False
        
    def _update_true_positive_entry(self, entity : int, candidate : int) -> None:
        """Updates the checked status of the given true positive

        Args:
            entity (int): Entity ID
            candidate (int): Candidate ID
        """
        if(self._till_full_tps_emission()):
            if(not self._true_positive_checked[canonical_swap(entity, candidate)]):
                self._true_positive_checked[canonical_swap(entity, candidate)] = True
                self._tps_found += 1
                return
    

    def calculate_roc_auc_data(self, data: Data, pairs, batch_size : int  = 1, true_positive_checked : dict = None) -> List[Tuple[int, int]]:
        """Progressively calculates total recall, AUC for each batch of candidate pairs
        Args:
            data (Data): Data Module
            pairs: List containing pairs in form (entity1 id, entity2 id, score)
        Raises:
            AttributeError: Ground Truth file hasn't been supplied, cannot calculate ROC AUC
        Returns:
            List[Tuple[int, int]]: List of ROC graph points information (recall up to e, normalized auc up to e)
        """

        if(true_positive_checked is not None): 
            for pair in true_positive_checked.keys():
                true_positive_checked[pair] = False

        if(data.ground_truth is None):
            raise AttributeError("Can calculate ROC AUC without a ground-truth file. \
                Data object mush have initialized with the ground-truth file")

        if(len(data.ground_truth) == 0):
            raise AttributeError("Cannot calculate AUC score, number of true duplicates is equal to 0.")
        
        _true_positives: int = 0
        _normalized_auc: int = 0
        _current_recall: int = 0
        _new_recall: int = 0
        self._tps_found : int = 0
        self._true_positive_checked : dict = true_positive_checked
        self.num_of_true_duplicates = len(data.ground_truth)
        _recall_progress = [0]

        batches = batch_pairs(pairs, batch_size)
        # ideal_auc = self.calculate_ideal_auc(len(pairs), self.num_of_true_duplicates)
        self._total_emissions : int = 0
        for batch in batches:
            _current_batch_size : int = 0
            for entity, candidate in batch:
                if(self._all_tps_emitted()): break
                entity_id = data._gt_to_ids_reversed_1[entity] if entity < data.dataset_limit else data._gt_to_ids_reversed_2[entity]
                candidate_id = data._gt_to_ids_reversed_1[candidate] if candidate < data.dataset_limit else data._gt_to_ids_reversed_2[candidate]
                _d1_entity, _d2_entity = (entity_id, candidate_id) if entity < data.dataset_limit else (candidate_id, entity_id)
                
                if _d2_entity in self.data.pairs_of[_d1_entity]:
                    self._update_true_positive_entry(entity_id, candidate_id)
                    _true_positives += 1  
                _current_batch_size += 1
            self._total_emissions += 1
            _new_recall = _true_positives / self.num_of_true_duplicates
            # _normalized_auc += ((_new_recall + _current_recall) / 2) * (_current_batch_size / self.num_of_true_duplicates)
            _current_recall = _new_recall
            _recall_progress.append(_current_recall)
            if(self._all_tps_emitted()): break
            

        # _normalized_auc = 0 if(ideal_auc == 0) else _normalized_auc / ideal_auc
        _normalized_auc = sum(_recall_progress) / (len(pairs) + 1.0)
        return _recall_progress, _normalized_auc
    
    def evaluate_auc_roc(self, matchers_data : List[Tuple], batch_size : int = 1, proportional : bool = True) -> None:
        """For each matcher, takes its prediction data, calculates cumulative recall and auc, plots the corresponding ROC curve, populates prediction data with performance info
        Args:
            matchers_data List[Tuple[str, ProgressiveMatching]]: Progressive Matchers and their names
            data (Data) : Data Module
            batch_size (int, optional): Emitted pairs step at which cumulative recall is recalculated. Defaults to 1.
            proportional (bool) : Proportional Visualization
        Raises:
            AttributeError: No Data object
            AttributeError: No Ground Truth file
        """
        
        if self.data is None:
            raise AttributeError("Can not proceed to AUC ROC evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to AUC ROC evaluation without a ground-truth file. " +
                    "Data object has not been initialized with the ground-truth file")

        self._matchers_auc_roc_data = []

        for matcher_data in matchers_data:
            
            matcher_name, progressive_matcher = matcher_data
            matcher_prediction_data : PredictionData = PredictionData(matcher_name, progressive_matcher.pairs, progressive_matcher.true_pair_checked)
            
            matcher_predictions = matcher_prediction_data.get_predictions()
            matcher_tps_checked = matcher_prediction_data.get_tps_checked()
            
            cumulative_recall, normalized_auc = self.calculate_roc_auc_data(self.data, matcher_predictions, batch_size, matcher_tps_checked)
                    
            self._matchers_auc_roc_data.append((matcher_name, normalized_auc, cumulative_recall))
            matcher_prediction_data.set_total_emissions(self._total_emissions)
            matcher_prediction_data.set_normalized_auc(normalized_auc)
            matcher_prediction_data.set_cumulative_recall(cumulative_recall[-1])
            progressive_matcher.set_prediction_data(matcher_prediction_data)

        self.visualize_roc(methods_data = self._matchers_auc_roc_data, proportional = proportional)
