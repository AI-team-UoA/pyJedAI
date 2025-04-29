from abc import ABC, abstractmethod
from itertools import count
from time import time
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from networkx import Graph
from tqdm.autonotebook import tqdm

from .datamodel import Data
from .evaluation import Evaluation
from .block_building import StandardBlocking
from .block_cleaning import BlockFiltering, BlockPurging
from .comparison_cleaning import *
from .matching import EntityMatching
from .clustering import ConnectedComponentsClustering, UniqueMappingClustering
from .vector_based_blocking import EmbeddingsNNBlockBuilding
from .joins import EJoin, TopKJoin

from .prioritization import ProgressiveMatching, BlockIndependentPM, class_references
from .utils import new_dictionary_from_keys, get_class_function_arguments, generate_unique_identifier



class PYJEDAIWorkFlow(ABC):
    """Main module of the pyjedAI and the simplest way to create an end-to-end ER workflow.
    """

    _id = count()

    def __init__(
            self,
            name: str = None
    ) -> None:
        self.f1: list = []
        self.recall: list = []
        self.precision: list = []
        self.runtime: list = []
        self.configurations: list = []
        self.workflow_exec_time: float
        self._id: int = next(self._id)
        self.name: str = name if name else "Workflow-" + str(self._id)
        self._workflow_bar: tqdm
        self.final_pairs = None

    @abstractmethod
    def run(self,
            data: Data,
            verbose: bool = False,
            with_classification_report: bool = False,
            workflow_step_tqdm_disable: bool = True,
            workflow_tqdm_enable: bool = False) -> None:
        pass

    def _init_experiment(self) -> None:
        self.f1: list = []
        self.recall: list = []
        self.precision: list = []
        self.runtime: list = []
        self.configurations: list = []
        self.workflow_exec_time: float

    def visualize(
            self,
            f1: bool = True,
            recall: bool = True,
            precision: bool = True,
            separate: bool = False
    ) -> None:
        """Performance Visualization of the workflow.

        Args:
            f1 (bool, optional): F-Measure. Defaults to True.
            recall (bool, optional): Recall. Defaults to True.
            precision (bool, optional): Precision. Defaults to True.
            separate (bool, optional): Separate plots. Defaults to False.
        """
        method_names = [conf['name'] for conf in self.configurations]
        exec_time = []
        prev = 0

        for i, _ in enumerate(self.runtime):
            exec_time.append(prev + self.runtime[i])
            prev = exec_time[i]

        if separate:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
            fig.suptitle(self.name + " Visualization", fontweight='bold', fontsize=14)
            fig.subplots_adjust(top=0.88)
            axs[0, 0].plot(method_names,
                           self.precision,
                           linewidth=2.0,
                           label="Precision",
                           marker='o',
                           markersize=10)
            axs[0, 0].set_ylabel("Scores %", fontsize=12)
            axs[0, 0].set_title("Precision", fontsize=12)
            axs[0, 1].plot(method_names,
                           self.recall,
                           linewidth=2.0,
                           label="Recall",
                           marker='*',
                           markersize=10)
            axs[0, 1].set_ylabel("Scores %", fontsize=12)
            axs[0, 1].set_title("Recall", fontsize=12)            
            axs[1, 0].plot(method_names,
                           self.f1,
                           linewidth=2.0,
                           label="F1-Score",
                           marker='x',
                           markersize=10)
            axs[1, 0].set_ylabel("Scores %", fontsize=12)
            axs[1, 0].set_title("F1-Score", fontsize=12)
            # axs[0, 0].legend(loc='lower right')
            axs[1, 1].plot(method_names,
                           exec_time,
                           linewidth=2.0,
                           label="Time",
                           marker='.',
                           markersize=10,
                           color='r')
            axs[1, 1].set_ylabel("Time (sec)", fontsize=12)
            axs[1, 1].set_title("Execution time", fontsize=12)
            fig.autofmt_xdate()
        else:
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
            fig.suptitle(self.name + " Visualization", fontweight='bold', fontsize=14)
            fig.subplots_adjust(top=0.88)
            if precision:
                axs[0].plot(method_names,
                            self.precision,
                            linewidth=2.0,
                            label="Precision",
                            marker='o',
                            markersize=10)
            if recall:
                axs[0].plot(method_names,
                            self.recall,
                            linewidth=2.0,
                            label="Recall",
                            marker='*',
                            markersize=10)
            if f1:
                axs[0].plot(method_names,
                            self.f1, linewidth=2.0,
                            label="F1-Score",
                            marker='x',
                            markersize=10)
            axs[0].set_xlabel("Models", fontsize=12)
            axs[0].set_ylabel("Scores %", fontsize=12)
            axs[0].set_title("Performance per step", fontsize=12)
            axs[0].legend(loc='lower right')
            exec_time = []
            prev = 0
            for i, _ in enumerate(self.runtime):
                exec_time.append(prev + self.runtime[i])
                prev = exec_time[i]
            axs[1].plot(method_names,
                        exec_time,
                        linewidth=2.0,
                        label="F1-Score",
                        marker='.',
                        markersize=10,
                        color='r')
            axs[1].set_ylabel("Time (sec)", fontsize=12)
            axs[1].set_title("Execution time", fontsize=12)
            fig.autofmt_xdate()
        plt.show()

    def to_df(self) -> pd.DataFrame:
        """Transform results into a pandas.DataFrame

        Returns:
            pd.DataFrame: Results
        """
        workflow_df = pd.DataFrame(
            columns=['Algorithm', 'F1', 'Recall', 'Precision', 'Runtime (sec)', 'Params'])
        workflow_df['F1'], workflow_df['Recall'], \
        workflow_df['Precision'], workflow_df['Runtime (sec)'] = \
            self.f1, self.recall, self.precision, self.runtime
        workflow_df['Algorithm'] = [c['name'] for c in self.configurations]
        workflow_df['Params'] = [c['parameters'] for c in self.configurations]

        return workflow_df

    def export_pairs(self) -> pd.DataFrame:
        """Export pairs to file.

        Returns:
            pd.DataFrame: pairs as a DataFrame
        """
        return self.final_step_method.export_to_df(self.final_pairs)

    def _save_step(self, results: dict, configuration: dict) -> None:
        self.f1.append(results['F1 %'])
        self.recall.append(results['Recall %'])
        self.precision.append(results['Precision %'])
        self.configurations.append(configuration)
        self.runtime.append(configuration['runtime'])

    def get_final_scores(self) -> Tuple[float, float, float]:
        """Final scores in the last step of the workflow.

        Returns:
            Tuple[float, float, float]: F-Measure, Precision, Recall.
        """
        return self.f1[-1], self.precision[-1], self.recall[-1]

class ProgressiveWorkFlow(PYJEDAIWorkFlow):
    """Main module of the pyjedAI and the simplest way to create an end-to-end PER workflow.
    """

    def __init__(
            self,
            name: str = None
    ) -> None:
        self.f1: list = []
        self.recall: list = []
        self.precision: list = []
        self.runtime: list = []
        self.configurations: list = []
        self.workflow_exec_time: float
        self._id: int = next(self._id)
        self.name: str = name if name else "Workflow-" + str(self._id)
        self._workflow_bar: tqdm
        self.final_pairs = None

    def run(self,
            data: Data,
            verbose: bool = False,
            with_classification_report: bool = False,
            workflow_step_tqdm_disable: bool = True,
            workflow_tqdm_enable: bool = False,
            block_building : dict = None,
            block_purging : dict = None,
            block_filtering : dict = None,
            **matcher_arguments
        ) -> None:
        """Main function for creating an Progressive ER workflow.

        Args:
            data (Data): Dataset Module, used to derive schema-awereness status
            verbose (bool, optional): Print detailed report for each step. Defaults to False.
            with_classification_report (bool, optional): Print pairs counts. Defaults to False.
            workflow_step_tqdm_disable (bool, optional):  Tqdm progress bar in each step. Defaults to True.
            workflow_tqdm_enable (bool, optional): Overall progress bar. Defaults to False.
            number_of_nearest_neighbors (int, optional): Number of nearest neighbours in cardinality based algorithms. Defaults to None.
            indexing (str, optional): Inorder/Reverse/Bilateral indexing of datasets. Defaults to None.
            similarity_function (str, optional): Function used to evaluate the similarity of two vector based representations of entities. Defaults to None.
            language_model (str, optional): Language model used to vectorize the entities. Defaults to None.
            tokenizer (str, optional): Text tokenizer used. Defaults to None.
            weighting_scheme (str, optional): Scheme used to evaluate the weight between nodes of intermediate representation graph. Defaults to None.
            block_building (dict, optional): Algorithm and its parameters used to construct the blocks. Defaults to None.
            block_purging (dict, optional): Algorithm and its parameters used to delete obsolete blocks. Defaults to None.
            block_filtering (dict, optional): Algorithm and its parameters used to lower the cardinality of blocks. Defaults to None.
            window_size (dict, optional): Window size in the Sorted Neighborhood Progressive ER workflows. Defaults to None.
        """
        self.block_building, self.block_purging, self.block_filtering, self.algorithm = \
        block_building, block_purging, block_filtering, matcher_arguments['algorithm']
        steps = [self.block_building, self.block_purging, self.block_filtering, self.algorithm]
        num_of_steps = sum(x is not None for x in steps)
        self._workflow_bar = tqdm(total=num_of_steps,
                                  desc=self.name,
                                  disable=not workflow_tqdm_enable)
         
        self.data : Data = data
        self._init_experiment()
        start_time = time()
        self.matcher_arguments = matcher_arguments
        self.matcher_name = self.matcher_arguments['matcher']
        self.dataset_name = self.matcher_arguments['dataset']
        matcher = class_references[matcher_arguments['matcher']]
        self.constructor_arguments = new_dictionary_from_keys(dictionary=self.matcher_arguments, keys=get_class_function_arguments(class_reference=matcher, function_name='__init__'))
        self.predictor_arguments = new_dictionary_from_keys(dictionary=self.matcher_arguments, keys=get_class_function_arguments(class_reference=matcher, function_name='predict'))
        print(self.constructor_arguments)
        print(self.predictor_arguments)
        
        progressive_matcher : ProgressiveMatching = matcher(**self.constructor_arguments)
        self.progressive_matcher : ProgressiveMatching = progressive_matcher
        #
        # Block Building step: Only one algorithm can be performed
        #
        block_building_method = (self.block_building['method'](**self.block_building["params"]) \
                                                    if "params" in self.block_building \
                                                    else self.block_building['method']()) if self.block_building \
                                                    else (None if not self._blocks_required() else StandardBlocking())

        bblocks = None
        block_building_blocks = None
        if block_building_method:
            block_building_blocks = \
                block_building_method.build_blocks(data,
                                                attributes_1=self.block_building["attributes_1"] \
                                                                    if(self.block_building is not None and "attributes_1" in self.block_building) else None,
                                                    attributes_2=self.block_building["attributes_2"] \
                                                                    if(self.block_building is not None and "attributes_2" in self.block_building) else None,
                                                    tqdm_disable=workflow_step_tqdm_disable)
            self.final_pairs = bblocks = block_building_blocks
            res = block_building_method.evaluate(block_building_blocks,
                                                export_to_dict=True,
                                                with_classification_report=with_classification_report,
                                                verbose=verbose)
            self._save_step(res, block_building_method.method_configuration())
            self._workflow_bar.update(1)

        if(block_building_blocks is not None):
            #
            # Block Purging step [optional]
            #
            bblocks = block_building_blocks
            block_purging_blocks = None
            if(self.block_purging is not None):
                block_purging_method = self.block_purging['method'](**self.block_purging["params"]) \
                                                if "params" in self.block_purging \
                                                else self.block_purging['method']()
                block_purging_blocks = block_purging_method.process(bblocks,
                                                                    data,
                                                                    tqdm_disable=workflow_step_tqdm_disable)
                self.final_pairs = bblocks = block_purging_blocks
                res = block_purging_method.evaluate(bblocks,
                                                    export_to_dict=True,
                                                    with_classification_report=with_classification_report,
                                                    verbose=verbose)
                self._save_step(res, block_purging_method.method_configuration())
                self._workflow_bar.update(1)
            #
            # Block Filtering step [optional]
            #
            block_filtering_blocks = None
            if(self.block_filtering is not None):
                block_filtering_method = self.block_filtering['method'](**self.block_filtering["params"]) \
                                                if "params" in self.block_filtering \
                                                else self.block_filtering['method']()
                block_filtering_blocks = block_filtering_method.process(bblocks,
                                                                        data,
                                                                        tqdm_disable=workflow_step_tqdm_disable)
                self.final_pairs = bblocks = block_filtering_blocks
                res = block_filtering_method.evaluate(bblocks,
                                                    export_to_dict=True,
                                                    with_classification_report=with_classification_report,
                                                    verbose=verbose)
                self._save_step(res, block_filtering_method.method_configuration())
                self._workflow_bar.update(1)

        #
        # Progressive Matching step
        #
        self.final_pairs : List[Tuple[float, int, int]] = progressive_matcher.predict(data=data, blocks=bblocks, dataset_identifier=self.dataset_name, **self.predictor_arguments)
        evaluator = Evaluation(self.data)
        self.tp_indices, self.total_emissions = evaluator.calculate_tps_indices(pairs=self.final_pairs,duplicate_of=progressive_matcher.duplicate_of, duplicate_emitted=progressive_matcher.duplicate_emitted)
        self.total_candidates = len(self.final_pairs)       
        self._workflow_bar.update(1)
        self.workflow_exec_time = time() - start_time

    def _blocks_required(self):
        return not isinstance(self.progressive_matcher, BlockIndependentPM)

    def _init_experiment(self) -> None:
        self.f1: list = []
        self.recall: list = []
        self.precision: list = []
        self.runtime: list = []
        self.configurations: list = []
        self.workflow_exec_time: float

    def visualize(
            self,
            f1: bool = True,
            recall: bool = True,
            precision: bool = True,
            separate: bool = False
    ) -> None:
        pass

    def to_df(self) -> pd.DataFrame:
        pass

    def export_pairs(self) -> pd.DataFrame:
        pass

    def _save_step(self, results: dict, configuration: dict) -> None:
        pass

    def get_final_scores(self) -> Tuple[float, float, float]:
        pass
    
    def retrieve_matcher_workflows(self, workflows : dict, arguments : dict) -> list:
        """Retrieves the list of already executed workflows for the matcher/model of current workflow 

        Args:
            workflows (dict): Dictionary of script's executed workflows' information
            arguments (dict): Arguments that have been supplied for current workflow execution

        Returns:
            list: List of already executed workflows for given workflow's arguments' matcher/model
        """
        dataset : str = self.dataset_name
        matcher : str = self.matcher_name
        
        workflows[dataset] = workflows[dataset] if dataset in workflows else dict()
        matcher_results = workflows[dataset]
        matcher_results[matcher] = matcher_results[matcher] if matcher in matcher_results \
                                else ([] if('language_model' not in arguments) else {})
                
        matcher_info = matcher_results[matcher]
        workflows_info = matcher_info
        if(isinstance(matcher_info, dict)):
            lm_name = arguments['language_model']
            matcher_info[lm_name] = matcher_info[lm_name] if lm_name in matcher_info else []
            workflows_info = matcher_info[lm_name]  
            
        return workflows_info
    
    
    
    def save(self, arguments : dict, path : str = None, results = None) -> dict:
        """Stores argument / execution information for current workflow within a workflows dictionary.
        
        Args:
            arguments (dict): Arguments that have been supplied for current workflow execution
            path (str): Path where the workflows results are stored at (Default to None),
            results (str): A dictionary of workflows results at which we want to store current workflow's arguments/info
        Returns:
            dict: Dictionary containing the information about the given workflow
        """
        if(path is None and results is None):
            raise ValueError(f"No dictionary path or workflows dictionary given - Cannot save workflow.")
        
        if(results is not None):
            workflows = results
        elif(not os.path.exists(path) or os.path.getsize(path) == 0):
            workflows = {}
        else:
            with open(path, 'r', encoding="utf-8") as file:
                workflows = json.load(file)
                
        category_workflows = self.retrieve_matcher_workflows(workflows=workflows, arguments=arguments)
        self.save_workflow_info(arguments=arguments) 
        category_workflows.append(self.info)
        
        if(path is not None):
            with open(path, 'w', encoding="utf-8") as file:
                json.dump(workflows, file, indent=4)
            
        return self.info
    
    def save_workflow_info(self, arguments : dict) -> dict:
        """Stores current workflow argument values and execution related data (like execution time and total emissions)

        Args:
            arguments (dict): Arguments that were passed to progressive workflow at hand
        """
        
        workflow_info : dict = {k: v for k, v in arguments.items()}
        workflow_info['total_candidates'] = self.total_candidates
        workflow_info['total_emissions'] = self.total_emissions
        workflow_info['time'] = self.workflow_exec_time
        workflow_info['name'] = generate_unique_identifier()
        workflow_info['tp_idx'] = self.tp_indices
        workflow_info['dataset'] = self.dataset_name
        workflow_info['matcher'] = self.matcher_name

        self.info = workflow_info  
    
    def print_info(self, info : dict):
        for attribute in info:
            value = info[attribute]
            if(attribute != 'tp_idx'):
                print(f"{attribute} : {value}")
            else:
                print(f"true_positives : {len(value)}")
    

def compare_workflows(workflows: List[PYJEDAIWorkFlow], with_visualization=True) -> pd.DataFrame:
    """Compares workflows by creating multiple plots and tables with results.

    Args:
        workflows (List[PYJEDAIWorkFlow]): Different workflows
        with_visualization (bool, optional): Diagram generation. Defaults to True.

    Returns:
        pd.DataFrame: Results
    """
    workflow_df = pd.DataFrame(columns=['Name', 'F1', 'Recall', 'Precision', 'Runtime (sec)'])
    if with_visualization:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        fig.suptitle("Workflows Performance Visualization", fontweight='bold', fontsize=14)
        fig.subplots_adjust(top=0.88)
        axs[0, 0].set_ylabel("Scores %", fontsize=12)
        axs[0, 0].set_title("Precision", fontsize=12)
        axs[0, 0].set_ylim([0, 100])
        axs[0, 1].set_ylabel("Scores %", fontsize=12)
        axs[0, 1].set_title("Recall", fontsize=12)
        axs[0, 1].set_ylim([0, 100])
        axs[1, 0].set_ylabel("Scores %", fontsize=12)
        axs[1, 0].set_title("F1-Score", fontsize=12)
        axs[1, 0].set_ylim([0, 100])
        axs[1, 1].set_ylabel("Time (sec)", fontsize=12)
        axs[1, 1].set_title("Execution time", fontsize=12)

    for w in workflows:
        workflow_df.loc[len(workflow_df)] = \
            [w.name, w.f1[-1], w.recall[-1], w.precision[-1], w.workflow_exec_time]

    if with_visualization:
        axs[0, 0].bar(workflow_df['Name'],
                      workflow_df['Precision'],
                      label=workflow_df['Name'],
                      color='b')
        axs[0, 1].bar(workflow_df['Name'],
                      workflow_df['Recall'],
                      label=workflow_df['Name'],
                      color='g')
        axs[1, 0].bar(workflow_df['Name'], workflow_df['F1'], color='orange')
        axs[1, 1].bar(workflow_df['Name'], workflow_df['Runtime (sec)'], color='r')
    fig.autofmt_xdate()
    plt.show()

    return workflow_df

class BlockingBasedWorkFlow(PYJEDAIWorkFlow):
    """Blocking-based workflow.
    """

    def __init__(
            self,
            block_building: dict = None,
            entity_matching: dict = None,
            block_cleaning: dict = None,
            comparison_cleaning: dict = None,
            clustering: dict = None,
            joins: dict = None,
            name: str = None
    ) -> None:
        super().__init__()
        self.block_cleaning, self.block_building, self.comparison_cleaning, \
            self.clustering, self.joins, self.entity_matching = \
            block_cleaning, block_building, comparison_cleaning, clustering, joins, entity_matching
        self.name: str = name if name else "BlockingBasedWorkFlow-" + str(self._id)

    def run(self,
            data: Data,
            verbose: bool = False,
            with_classification_report: bool = False,
            workflow_step_tqdm_disable: bool = True,
            workflow_tqdm_enable: bool = False
        ) -> None:
        """Main function for creating an Entity resolution workflow.

        Args:
            data (Data): Dataset module.
            verbose (bool, optional): Print detailed report for each step. Defaults to False.
            with_classification_report (bool, optional): Print pairs counts. Defaults to False.
            workflow_step_tqdm_disable (bool, optional):  Tqdm progress bar in each step. Defaults to True.
            workflow_tqdm_enable (bool, optional): Overall progress bar. Defaults to False.
        """
        steps = [self.block_building, self.entity_matching, self.clustering, self.joins, self.block_cleaning, self.comparison_cleaning]
        num_of_steps = sum(x is not None for x in steps)
        self._workflow_bar = tqdm(total=num_of_steps,
                                  desc=self.name,
                                  disable=not workflow_tqdm_enable)
        self.data = data
        self._init_experiment()
        start_time = time()
        #
        # Block building step: Only one algorithm can be performed
        #
        block_building_method = self.block_building['method'](**self.block_building["params"]) \
                                                    if "params" in self.block_building \
                                                    else self.block_building['method']()
        block_building_blocks = \
            block_building_method.build_blocks(data,
                                               attributes_1=self.block_building["attributes_1"] \
                                                                if "attributes_1" in self.block_building else None,
                                                attributes_2=self.block_building["attributes_2"] \
                                                                if "attributes_2" in self.block_building else None,
                                                tqdm_disable=workflow_step_tqdm_disable)
        self.final_pairs = block_building_blocks
        self.final_step_method = block_building_method
        if data.ground_truth is not None:
            res = block_building_method.evaluate(block_building_blocks,
                                                export_to_dict=True,
                                                with_classification_report=with_classification_report,
                                                verbose=verbose)
            self._save_step(res, block_building_method.method_configuration())
        self._workflow_bar.update(1)
        #
        # Block cleaning step [optional]: Multiple algorithms
        #
        block_cleaning_blocks = None
        if self.block_cleaning:
            if isinstance(self.block_cleaning, dict):
                self.block_cleaning = list(self.block_cleaning)
            bblocks = block_building_blocks
            for block_cleaning in self.block_cleaning:
                block_cleaning_method = block_cleaning['method'](**block_cleaning["params"]) \
                                                    if "params" in block_cleaning \
                                                    else block_cleaning['method']()
                block_cleaning_blocks = block_cleaning_method.process(bblocks,
                                                                      data,
                                                                      tqdm_disable=workflow_step_tqdm_disable)
                
                self.final_pairs = bblocks = block_cleaning_blocks
                # self.final_pairs = block_cleaning_method.export_to_df(self.final_pairs)
                if data.ground_truth is not None:
                    res = block_cleaning_method.evaluate(bblocks,
                                                        export_to_dict=True,
                                                        with_classification_report=with_classification_report,
                                                        verbose=verbose)
                    self._save_step(res, block_cleaning_method.method_configuration())
                self._workflow_bar.update(1)
        #
        # Comparison cleaning step [optional]
        #
        comparison_cleaning_blocks = None
        if self.comparison_cleaning:
            comparison_cleaning_method = self.comparison_cleaning['method'](**self.comparison_cleaning["params"]) \
                                            if "params" in self.comparison_cleaning \
                                            else self.comparison_cleaning['method']()
            self.final_pairs = \
            comparison_cleaning_blocks = \
            comparison_cleaning_method.process(block_cleaning_blocks if block_cleaning_blocks is not None \
                                                    else block_building_blocks,
                                                data,
                                                tqdm_disable=workflow_step_tqdm_disable)
            self.final_step_method = comparison_cleaning_method

            if data.ground_truth is not None:
                res = comparison_cleaning_method.evaluate(comparison_cleaning_blocks,
                                                        export_to_dict=True,
                                                        with_classification_report=with_classification_report,
                                                        verbose=verbose)
                self._save_step(res, comparison_cleaning_method.method_configuration())
            self._workflow_bar.update(1)
        #
        # Entity Matching step
        #
        entity_matching_method = self.entity_matching['method'](**self.entity_matching["params"]) \
                                        if "params" in self.entity_matching \
                                        else self.entity_matching['method']()
                                        
        if "exec_params" not in self.entity_matching:
            self.final_pairs = em_graph = entity_matching_method.predict(
                comparison_cleaning_blocks if comparison_cleaning_blocks is not None \
                    else block_building_blocks,
                data,
                tqdm_disable=workflow_step_tqdm_disable)
        else:
            self.final_pairs = em_graph = entity_matching_method.predict(
                comparison_cleaning_blocks if comparison_cleaning_blocks is not None \
                    else block_building_blocks,
                data,
                tqdm_disable=workflow_step_tqdm_disable,
                **self.entity_matching["exec_params"])

        self.final_step_method = entity_matching_method

        if data.ground_truth is not None:
            res = entity_matching_method.evaluate(em_graph,
                                                    export_to_dict=True,
                                                    with_classification_report=with_classification_report,
                                                    verbose=verbose)
            self._save_step(res, entity_matching_method.method_configuration())
        self._workflow_bar.update(1)
        #
        # Clustering step [optional]
        #
        if self.clustering:
            clustering_method = self.clustering['method'](**self.clustering["params"]) \
                                            if "params" in self.clustering \
                                            else self.clustering['method']()
            if "exec_params" not in self.clustering:
                self.final_pairs = components = clustering_method.process(em_graph, data)
            else:
                self.final_pairs = components = clustering_method.process(em_graph, data, **self.clustering["exec_params"])

            self.final_step_method = clustering_method
            self.clusters = components
            if data.ground_truth is not None:
                res = clustering_method.evaluate(components,
                                                export_to_dict=True,
                                                with_classification_report=with_classification_report,
                                                verbose=verbose)
                self._save_step(res, clustering_method.method_configuration())
            self.workflow_exec_time = time() - start_time
            self._workflow_bar.update(1)
        # self.runtime.append(self.workflow_exec_time)
    
    ############################################
    #  Pre-defined workflows same as JedAI     #
    ############################################

    def best_blocking_workflow_ccer(self) -> None:
        """Best CC-ER workflow.

        Returns:
            PYJEDAIWorkFlow: Best workflow
        """
        self.block_building = dict(method=StandardBlocking)
        self.block_cleaning = [dict(
            method=BlockFiltering,
            params=dict(ratio=0.9)
        )]
        self.comparison_cleaning = dict(method=WeightedEdgePruning, params=dict(weighting_scheme='EJS'))
        self.entity_matching = dict(method=EntityMatching,
                                    params=dict(metric='cosine',
                                                    tokenizer='char_tokenizer', 
                                                    vectorizer='tfidf',
                                                    qgram=3,
                                                    similarity_threshold=0.0))
        self.clustering = dict(method=UniqueMappingClustering, 
                               exec_params=dict(similarity_threshold=0.17))
        self.name="best-ccer-workflow"

    def best_blocking_workflow_der(self) -> None:
        """Best D-ER workflow.

        Returns:
            PYJEDAIWorkFlow: Best workflow
        """
        self.block_building = dict(method=StandardBlocking)
        self.block_cleaning = [
            dict(method=BlockPurging, params=dict(smoothing_factor=1.0)),
            dict(method=BlockFiltering)
        ]
        self.comparison_cleaning = dict(method=CardinalityNodePruning,
                                    params=dict(weighting_scheme='JS'))
        self.entity_matching = dict(method=EntityMatching, 
                                    params=dict(metric='cosine',
                                                similarity_threshold=0.55))
        self.clustering = dict(method=ConnectedComponentsClustering),
        self.name="best-der-workflow"

    def default_schema_clustering_workflow_der(self) -> None:
        """Default D-ER workflow.

        Returns:
            PYJEDAIWorkFlow: Best workflow
        """
        self.block_building = dict(method=StandardBlocking)
        self.block_cleaning = [
            dict(method=BlockPurging, params=dict(smoothing_factor=1.0)),
            dict(method=BlockFiltering)
        ]
        self.entity_matching = dict(method=EntityMatching, 
                                    params=dict(metric='cosine',
                                                similarity_threshold=0.35))
        self.clustering = dict(method=ConnectedComponentsClustering),
        self.name="best-schema-clustering-der-workflow"


    def default_schema_clustering_workflow_ccer(self) -> None:
        """Default CC-ER workflow.
        """
        self.block_building = dict(method=StandardBlocking)
        self.block_cleaning = [
                dict(method=BlockPurging, params=dict(smoothing_factor=1.0)),
                dict(method=BlockFiltering)
            ]
        self.entity_matching = dict(method=EntityMatching,
                                    metric='cosine',
                                    similarity_threshold=0.35)
        self.clustering = dict(method=ConnectedComponentsClustering)
        self.name="default-schema-clustering-ccer-workflow"

    def default_blocking_workflow_ccer(self) -> None:
        """Default CC-ER workflow.
        """
        self.block_building = dict(method=StandardBlocking)
        self.block_cleaning = [
                dict(method=BlockPurging, params=dict(smoothing_factor=1.0)),
                dict(method=BlockFiltering)
            ]
        self.comparison_cleaning = dict(method=CardinalityNodePruning)
        self.entity_matching = dict(method=EntityMatching,
                                    metric='cosine',
                                    similarity_threshold=0.55)
        self.clustering = dict(method=UniqueMappingClustering)
        self.name="default-ccer-workflow"

    def default_blocking_workflow_der(self) -> None:
        """Default D-ER workflow.

        Returns:
            PYJEDAIWorkFlow: Best workflow
        """
        self.block_building = dict(method=StandardBlocking)
        self.block_cleaning = [
                dict(method=BlockPurging, params=dict(smoothing_factor=1.0)),
                dict(method=BlockFiltering)
        ]
        self.comparison_cleaning = dict(method=CardinalityNodePruning, 
                                        params=dict(weighting_scheme='JS'))
        self.entity_matching = dict(method=EntityMatching,
                                    params=dict(metric='cosine', 
                                                similarity_threshold=0.55))
        self.name="default-der-workflow"
 
class EmbeddingsNNWorkFlow(PYJEDAIWorkFlow):
    """Blocking-based workflow.
    """

    def __init__(
            self,
            block_building: dict,
            clustering: dict = None,
            name: str = None
    ) -> None:
        super().__init__()
        self.block_building, self.clustering = block_building, clustering
        self.name: str = name if name else "Workflow-" + str(self._id)

    def run(self,
            data: Data,
            verbose: bool = False,
            with_classification_report: bool = False,
            workflow_step_tqdm_disable: bool = False,
            workflow_tqdm_enable: bool = False
        ) -> None:
        """Main function for creating an Entity resolution workflow.

        Args:
            data (Data): Dataset module.
            verbose (bool, optional): Print detailed report for each step. Defaults to False.
            with_classification_report (bool, optional): Print pairs counts. Defaults to False.
            workflow_step_tqdm_disable (bool, optional):  Tqdm progress bar in each step. Defaults to True.
            workflow_tqdm_enable (bool, optional): Overall progress bar. Defaults to False.
        """
        steps = [self.block_building, self.clustering]
        num_of_steps = sum(x is not None for x in steps)
        self._workflow_bar = tqdm(total=num_of_steps,
                                  desc=self.name,
                                  disable=not workflow_tqdm_enable)
        self.data = data
        self._init_experiment()
        start_time = time()
        #
        # Block building step: Only one algorithm can be performed
        #
        block_building_method = self.block_building['method'](**self.block_building["params"]) \
                                                    if "params" in self.block_building \
                                                    else self.block_building['method']()

        
        if "exec_params" not in self.block_building:
            block_building_blocks = \
                block_building_method.build_blocks(data,
                                            attributes_1=self.block_building["attributes_1"] \
                                                                if "attributes_1" in self.block_building else None,
                                                attributes_2=self.block_building["attributes_2"] \
                                                                if "attributes_2" in self.block_building else None,
                                                tqdm_disable=workflow_step_tqdm_disable)
        else:
            block_building_blocks, em_graph = \
                block_building_method.build_blocks(data, 
                                                   attributes_1=self.block_building["attributes_1"] \
                                                                if "attributes_1" in self.block_building else None,
                                                    attributes_2=self.block_building["attributes_2"] \
                                                                if "attributes_2" in self.block_building else None,
                                                tqdm_disable=workflow_step_tqdm_disable,
                                                with_entity_matching=True,
                                                **self.block_building["exec_params"])                
        self.final_step_method = block_building_method
        self.final_pairs = block_building_blocks
        # self.final_pairs = block_building_method.export_to_df(self.final_pairs)

        if data.ground_truth is not None:
            res = block_building_method.evaluate(block_building_blocks,
                                                export_to_dict=True,
                                                with_classification_report=with_classification_report,
                                                verbose=verbose)
            self._save_step(res, block_building_method.method_configuration())
        self._workflow_bar.update(1)
        #
        # Clustering step [optional]
        #
        if self.clustering:
            clustering_method = self.clustering['method'](**self.clustering["params"]) \
                                            if "params" in self.clustering \
                                            else self.clustering['method']()
            if "exec_params" not in self.clustering:
                self.final_pairs = components = clustering_method.process(em_graph, data)
            else:
                self.final_pairs = components = clustering_method.process(em_graph, data, **self.clustering["exec_params"])
            
            self.final_step_method = clustering_method
            # self.final_pairs = clustering_method.export_to_df(self.final_pairs)

            if data.ground_truth is not None:
                res = clustering_method.evaluate(components,
                                                export_to_dict=True,
                                                with_classification_report=False,
                                                verbose=verbose)
                self._save_step(res, clustering_method.method_configuration())
            self.clusters = components
            self.workflow_exec_time = time() - start_time
            self._workflow_bar.update(1)
        # self.runtime.append(self.workflow_exec_time)

# class SimilarityJoinsWorkFlow(PYJEDAIWorkFlow):
#     raise NotImplementedError("Joins workflow is not implemented yet.")

# class ProgressiveWorkFlow(PYJEDAIWorkFlow):
#     raise NotImplementedError("Progressive workflow is not implemented yet.")