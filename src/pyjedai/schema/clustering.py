import pandas as pd
import time as time

from tqdm import tqdm   
from ..datamodel import Data, PYJEDAIFeature
from ..evaluation import Evaluation
from ..workflow import PYJEDAIWorkFlow, BlockingBasedWorkFlow
from ..clustering import AbstractClustering, UniqueMappingClustering, ConnectedComponentsClustering
from ..block_building import StandardBlocking
from ..block_cleaning import BlockFiltering, BlockPurging
from ..comparison_cleaning import WeightedEdgePruning, CardinalityEdgePruning, WeightedNodePruning
from ..matching import EntityMatching

from abc import abstractmethod

from typing import Optional, List, Tuple

class AbstractSchemaClustering(AbstractClustering):
    """Abstract class for schema clustering methods
    """
        
    def __init__(self):
        super().__init__()
        self.execution_time: float = 0.0
        self.schema_clustering_execution_time: float = 0.0

    @abstractmethod
    def _configuration(self) -> dict:
        pass

    @abstractmethod
    def stats(self) -> None:
        pass
    
    def report(self) -> None:
        """Prints method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "\nRuntime (schema-clustering): {:2.4f} seconds".format(self.schema_clustering_execution_time) +
            "\nRuntime (total): {:2.4f} seconds".format(self.execution_time)
        )

class SchemaClustering(AbstractSchemaClustering):
    """Class to provide schema clustering methods
    """
    _method_name = "Schema Clustering"
    _method_info = "Performs pyjedai workflow to the names or values of the given data schema and then for each cluster performs pyjedai workflow for entity resolution"

    def __init__(self):
        super().__init__()
        self.on: str = 'names'
        self.schema_clustering_workflow: str
        self.entity_resolution_workflow: str

    def _configuration(self) -> dict:
        return {
            'on': self.on,
            'schema_clustering_workflow': self.schema_clustering_workflow,
            'entity_resolution_workflow': self.entity_resolution_workflow
        }

    def transform_mapping_to_ids(self,
                                 clusters,
                                 data: Data) -> pd.DataFrame:
        pairs_dict = dict()
        for cluster in clusters:
            lcluster = list(cluster)
            for i1 in range(0, len(lcluster)):
                for i2 in range(i1+1, len(lcluster)):
                    if lcluster[i1] < data.dataset_limit:
                        id1 = data._gt_to_ids_reversed_1[lcluster[i1]]
                        id2 = data._gt_to_ids_reversed_1[lcluster[i2]] \
                                if data.is_dirty_er \
                                else data._gt_to_ids_reversed_2[lcluster[i2]]
                    else:
                        id2 = data._gt_to_ids_reversed_2[lcluster[i1]]
                        id1 = data._gt_to_ids_reversed_1[lcluster[i2]]

                    pairs_dict[str(id1)] = str(id2)
        return pairs_dict

    def stats(self) -> None:
        print(
            "Statistics:" +
            "\n\tNew subsets shapes: " + \
            ''.join(['\n\t\tSubset {0}: {1}'.format(i, shape) for i, shape in self.new_data_stats]))
        return {
            self.new_data_stats    
        }

    def evaluate(self, 
                 prediction, 
                 export_to_df: bool = False, 
                 export_to_dict: bool = False, 
                 with_classification_report: bool = False, 
                 verbose: bool = True) -> any:

        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " + 
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        true_positives = 0
        
        for _, (id1, id2) in tqdm(self.data.ground_truth.iterrows(), desc="Evaluating"):
            if id1 in prediction and  \
                    prediction[id1] == id2:
                true_positives += 1
        eval_obj.calculate_scores(true_positives=true_positives, total_matching_pairs=len(self.data.ground_truth))
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)

    def process(self, 
                data: Data, on='names', 
                pyjedai_workflow_for_clustering: PYJEDAIWorkFlow = None,
                pyjedai_workflow_for_er: PYJEDAIWorkFlow = None,
                verbose_schema_clustering: bool = False,
                verbose_er: bool = False,
                return_clusters = False) -> Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]]:

        _start_time = time.time()
        self.data = data
        self.on = on

        if pyjedai_workflow_for_clustering == None:
            pyjedai_workflow_for_clustering = BlockingBasedWorkFlow()
            if data.is_dirty_er:
                pyjedai_workflow_for_clustering.default_schema_clustering_workflow_der()
            else:
                pyjedai_workflow_for_clustering.default_schema_clustering_workflow_ccer()

        self.schema_clustering_workflow = pyjedai_workflow_for_clustering.name
        
        print("Workflow used for schema clustering:", pyjedai_workflow_for_clustering.name)
        # D1
        entities_d1 = dict()
        for column in data.dataset_1.columns:
            if on == 'names':
                entities_d1[column] = column
            elif on == 'values':
                entities_d1[column] = ' '.join(data.dataset_1[column].astype(str))
            elif on == 'hybrid':
                entities_d1[column] = column + ' ' + ' '.join(data.dataset_1[column].astype(str))
            else:
                raise ValueError("on parameter must be one of 'names', 'values' or 'hybrid'")

        entities_d2 = None
        if not data.is_dirty_er:
            entities_d2 = dict()
            # D2
            for column in data.dataset_2.columns:
                if on == 'names':
                    entities_d2[column] = column
                elif on == 'values':
                    entities_d2[column] = ' '.join(data.dataset_2[column].astype(str))
                elif on == 'hybrid':
                    entities_d2[column] = column + ' ' + ' '.join(data.dataset_2[column].astype(str))
                else:
                    raise ValueError("on parameter must be one of 'names', 'values' or 'hybrid'")

        # Create dataframes from dictionaries
        attributes_d1 = pd.DataFrame.from_dict(entities_d1, orient='index', columns=['attribute'])
        attributes_d1['ids'] = range(0, len(attributes_d1))

        if not data.is_dirty_er:
            attributes_d2 = pd.DataFrame.from_dict(entities_d2, orient='index', columns=['attribute'])
            attributes_d2['ids'] = range(0, len(attributes_d2))

        # Clustering with pyJedAI        
        attributes_data = Data(
            dataset_1=attributes_d1,
            attributes_1=['attribute'],
            id_column_name_1='ids',
            dataset_2=attributes_d2 if not data.is_dirty_er else None,
            attributes_2=['attribute'] if not data.is_dirty_er else None,
            id_column_name_2='ids' if not data.is_dirty_er else None
        )
        
        pyjedai_workflow_for_clustering.run(attributes_data, verbose=verbose_schema_clustering)

        clusters = pyjedai_workflow_for_clustering.clusters

        # Create a new cluster of entities not in clusters
        def find_entities_not_in_clusters(ids, clusters):
            all_entities = set(ids)
            entities_in_clusters = set.union(*clusters)
            entities_not_in_clusters = all_entities - entities_in_clusters
            new_set = set(entities_not_in_clusters)
            return new_set

        all_ids = set(range(0, len(attributes_data.dataset_1) + len(attributes_data.dataset_2)))
        redundant_entities = find_entities_not_in_clusters(all_ids, clusters)
        if len(redundant_entities) > 0:
            clusters.append(redundant_entities)

        def contains_attributes_from_both(limit, cluster):
            has_entity_from_d1 = any(num < limit for num in cluster)
            has_entity_from_d2 = any(num >= limit for num in cluster)

            return has_entity_from_d1 and has_entity_from_d2
        
        # print("Clusters: ", clusters)
        
        new_datasets = []
        for i, cluster in enumerate(clusters):

            if not contains_attributes_from_both(attributes_data.dataset_limit, cluster):
                continue
            
            non_nan_indexes_d1 = set()
            non_nan_indexes_d2 = set() if not data.is_dirty_er else None
            
            for entity in cluster:
                if entity < attributes_data.dataset_limit:
                    attribute_name = attributes_data.dataset_1.iloc[entity].name
                    new_ids_d1 = set(data.dataset_1[data.dataset_1[attribute_name].notna()].index)
                    non_nan_indexes_d1.update(new_ids_d1)
                else:
                    attribute_name = attributes_data.dataset_2.iloc[entity-attributes_data.dataset_limit].name
                    non_nan_indexes_d2_attr = set(data.dataset_2[data.dataset_2[attribute_name].notna()].index)
                    new_ids_2 = set(map(lambda x: x - attributes_data.dataset_limit, non_nan_indexes_d2_attr))
                    non_nan_indexes_d2.update(new_ids_2)
            
            new_df_1 = data.dataset_1.iloc[list(non_nan_indexes_d1)]
            
            if not data.is_dirty_er:
                new_df_2 = data.dataset_2.iloc[list(non_nan_indexes_d2)]
                new_datasets.append((new_df_1, new_df_2))
        
        self.schema_clustering_execution_time = time.time() - _start_time

        if return_clusters:
            return new_datasets
        
        if pyjedai_workflow_for_er == None:
            pyjedai_workflow_for_er = BlockingBasedWorkFlow()
            if data.is_dirty_er:
                pyjedai_workflow_for_er.best_blocking_workflow_der() 
            else:
                pyjedai_workflow_for_er.best_blocking_workflow_ccer()

        self.entity_resolution_workflow = pyjedai_workflow_for_er.name

        global_pairs_dict = dict()
        self.new_data_stats = []
        for i in tqdm(range(len(new_datasets)), desc="Entity resolution for clusters"):
            d1, d2 = new_datasets[i]
            self.new_data_stats.append((i, d1.shape, d2.shape))
            
            new_data = Data(
                dataset_1=d1,
                attributes_1=data.attributes_1,
                id_column_name_1=data.id_column_name_1,
                dataset_2=d2,
                attributes_2=data.attributes_2,
                id_column_name_2=data.id_column_name_2,
                ground_truth=data.ground_truth
            )
            new_data.print_specs()
            pyjedai_workflow_for_er.run(new_data, verbose=verbose_er)
            new_clusters = pyjedai_workflow_for_er.clusters
            pairs = self.transform_mapping_to_ids(new_clusters, new_data)            
            global_pairs_dict.update(pairs)
            
        self.execution_time = time.time() - _start_time
        
        return global_pairs_dict

class RDFSchemaClustering(AbstractSchemaClustering):
    """Class to provide schema clustering methods
    """
    _method_name = "RDF Schema Clustering"
    _method_info = "Performs pyjedai dirty er workflow for the concatenation of objects per predicate. " + \
                        "For each rdf-triple containing the corresponding predicate, a new set is created. " + \
                            "For each set, a pyjedai dirty er workflow runs and finds the duplicate entities. " + \
                                "Links between entities, give us the corresponding same subjects between datasets. "

    def __init__(self):
        super().__init__()
        self.on: str = 'predicates'
        self.schema_clustering_workflow: str
        self.entity_resolution_workflow: str

    def _configuration(self) -> dict:
        return {
            'on': self.on,
            'schema_clustering_workflow': self.schema_clustering_workflow,
            'entity_resolution_workflow': self.entity_resolution_workflow
        }

    def stats(self) -> None:
        pass

    def transform_mapping_to_ids(self,
                                 clusters,
                                 data: Data) -> pd.DataFrame:
        pairs_dict = dict()
        for cluster in clusters:
            lcluster = list(cluster)
            for i1 in range(0, len(lcluster)):
                for i2 in range(i1+1, len(lcluster)):
                    if lcluster[i1] < data.dataset_limit:
                        id1 = data._gt_to_ids_reversed_1[lcluster[i1]]
                        id2 = data._gt_to_ids_reversed_1[lcluster[i2]] \
                                if data.is_dirty_er \
                                else data._gt_to_ids_reversed_2[lcluster[i2]]
                    else:
                        id2 = data._gt_to_ids_reversed_2[lcluster[i1]]
                        id1 = data._gt_to_ids_reversed_1[lcluster[i2]]

                    sid1 = data.dataset_1.iloc[int(id1)].name
                    sid2 = data.dataset_2.iloc[int(id2)].name
                    pairs_dict[sid1] = sid2
        return pairs_dict

    def evaluate(self, 
                 prediction, 
                 export_to_df: bool = False, 
                 export_to_dict: bool = False, 
                 with_classification_report: bool = False, 
                 verbose: bool = True) -> any:

        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " + 
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        true_positives = 0
        i1=0
        i2=0
        
        for _, (id1, id2) in tqdm(self.data.ground_truth.iterrows(), desc="Evaluating"):
            # print("ids: ", id1," and ", id2)
            if id1 in prediction and prediction[id1] == id2:
                true_positives += 1
            if id1 in prediction:
                i1+=1
            if id2 in set(prediction.values()):
                i2+=1
                # print(id1, ":", id2)
                # print(prediction[id1], ":", id2)
        print("Num of-1: ", i1)
        print("Num of-2: ", i2)
        print("True Positives: ", true_positives)
        print("Total Matching Pairs: ", len(self.data.ground_truth))
        eval_obj.calculate_scores(true_positives=true_positives, total_matching_pairs=len(self.data.ground_truth))
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)
    
    def process(self, 
                data: Data, 
                on='predicates', 
                pyjedai_workflow_for_clustering: PYJEDAIWorkFlow = None,
                pyjedai_workflow_for_er: PYJEDAIWorkFlow = None,
                verbose_schema_clustering: bool = False,
                verbose_er: bool = False,
                return_clusters = False) -> Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]]:

        _start_time = time.time()
        self.data = data
        self.on = on
        # self._progress_bar = tqdm(total=100, desc="RDF Schema Clustering")
        print("Starting RDF Schema Clustering")
        if pyjedai_workflow_for_clustering == None:
            pyjedai_workflow_for_clustering = self.pyjedai_workflow_for_er_on_predicates()

        self.schema_clustering_workflow = pyjedai_workflow_for_clustering.name
        
        print("Workflow used for schema clustering:", pyjedai_workflow_for_clustering.name)
        
        # Concatenate dataframe predicates and join object values
        predicates = dict()
        predicates_set_d1 = set()
        print("D1: Concatenating predicates and joining object values")
        for _, row in tqdm(data.dataset_1.iterrows()):
            if row['predicate'] not in predicates:
                predicates[row['predicate']] = str()
            predicates[row['predicate']] += ' ' + row['object']
            predicates_set_d1.add(row['predicate'])
        
        print("D2: Concatenating predicates and joining object values")
        predicates_set_d2 = None
        if not data.is_dirty_er:
            predicates_set_d2 = set()
            for _, row in data.dataset_2.iterrows():
                if row['predicate'] not in predicates:
                    predicates[row['predicate']] = str()
                predicates[row['predicate']] += ' ' + row['object']
                predicates_set_d2.add(row['predicate'])
        
        # Create dataframe from predicate dictionary
        # print("Predicates: ", predicates)
        print("Creating dataframe from predicate dictionary")
        predicates_df = pd.DataFrame.from_dict(predicates, orient='index', columns=['concatenation'])
        predicates_df['ids'] = range(0, len(predicates_df))

        print(predicates_df.head())

        # Clustering with pyJedAI
        print("Clustering with pyJedAI")
        all_predicates_data = Data(
            dataset_1=predicates_df,
            attributes_1=['concatenation'],
            id_column_name_1='ids'
        )

        all_predicates_data.print_specs()
        
        pyjedai_workflow_for_clustering.run(data=all_predicates_data, 
                                            verbose=verbose_schema_clustering,
                                            workflow_step_tqdm_disable=not verbose_schema_clustering,
                                            workflow_tqdm_enable=verbose_schema_clustering)

        clusters = pyjedai_workflow_for_clustering.clusters
        
        # print("Clusters: ", clusters)

        # Create a new cluster of entities not in clusters
        def find_entities_not_in_clusters(ids, clusters):
            all_entities = set(ids)
            entities_in_clusters = set.union(*clusters)
            entities_not_in_clusters = all_entities - entities_in_clusters
            new_set = set(entities_not_in_clusters)
            return new_set

        all_ids = set(range(0, len(all_predicates_data.dataset_1)))
        redundant_entities = find_entities_not_in_clusters(all_ids, clusters)
        if len(redundant_entities) > 0:
            clusters.append(redundant_entities)

        print("\n\n\n Clusters: ", clusters)

        def contains_items_from_both(cluster):
            cluster = list(cluster)
            has_entity_from_d1 = any((all_predicates_data.dataset_1.iloc[cluster[i]].name in predicates_set_d1) for i in range(len(cluster)))
            has_entity_from_d2 = any(all_predicates_data.dataset_1.iloc[cluster[i]].name in predicates_set_d2 for i in range(len(cluster)))
            
            return has_entity_from_d1 and has_entity_from_d2
        
        new_datasets = []
        for i, cluster in tqdm(enumerate(clusters), desc="Creating subsets"):

            if not contains_items_from_both(cluster):
                continue
            
            non_nan_indexes_d1 = set()
            non_nan_indexes_d2 = set() if not data.is_dirty_er else None
            
            for predicate_id in cluster:
                
                predicate = all_predicates_data.dataset_1.iloc[predicate_id].name
                # print(predicate_id, ":", predicate)
                if predicate in predicates_set_d1:
                    new_ids_d1 = set(data.dataset_1[data.dataset_1['predicate'] == predicate].index)
                    non_nan_indexes_d1.update(new_ids_d1)
                
                if not data.is_dirty_er and predicate in predicates_set_d2:
                    new_ids_d2 = set(data.dataset_2[data.dataset_2['predicate'] == predicate].index)
                    non_nan_indexes_d2.update(new_ids_d2)

            new_df_1 = data.dataset_1.iloc[list(non_nan_indexes_d1)].copy()
                        
            if not data.is_dirty_er:
                new_df_2 = data.dataset_2.iloc[list(non_nan_indexes_d2)].copy()
                new_datasets.append((new_df_1, new_df_2))
                # print("New datasets: ", new_df_1.shape[0], " x ", new_df_2.shape[0])
            else:
                new_datasets.append(new_df_1)
        self.schema_clustering_execution_time = time.time() - _start_time

        # free(all)
        print("New datasets: ", len(new_datasets))
        
        if return_clusters:
            return new_datasets

        print("Freeing memory")
        del all_predicates_data, predicates_df, clusters, all_ids, \
            redundant_entities, non_nan_indexes_d1, non_nan_indexes_d2

        if pyjedai_workflow_for_er == None:
            pyjedai_workflow_for_er = self.pyjedai_workflow_for_er_on_subjects()
            # if data.is_dirty_er:
            #     pyjedai_workflow_for_er.best_blocking_workflow_der()
            # else:
            #     pyjedai_workflow_for_er.best_blocking_workflow_ccer()

        self.entity_resolution_workflow = pyjedai_workflow_for_er.name
        print("Workflow used for entity resolution between subjects:", pyjedai_workflow_for_er.name)

        global_pairs_dict = dict()

        for i in tqdm(range(len(new_datasets)), desc="Entity resolution for clusters"):
            d1, d2 = new_datasets[i]
            
            print("Dataset 1: ", d1.shape)
            # print(d1.head())

            print("Dataset 2: ", d2.shape)
            # print(d2.head())
            
            if d1.shape[0] < 2 or (not data.is_dirty_er and d2.shape[0] < 2):
                continue

            subjects_d1 = dict()
            for _, row in tqdm(d1.iterrows(), desc="Creating subjects for dataset 1"):
                if row['subject'] not in subjects_d1:
                    subjects_d1[row['subject']] = str()
                subjects_d1[row['subject']] += ' ' + row['object']
            
            subjects_d2 = None
            if not data.is_dirty_er:
                subjects_d2 = dict()
                for _, row in tqdm(d2.iterrows(), desc="Creating subjects for dataset 2"):
                    if row['subject'] not in subjects_d2:
                        subjects_d2[row['subject']] = str()
                    subjects_d2[row['subject']] += ' ' + row['object']
            
            subjects_df_1 = pd.DataFrame.from_dict(subjects_d1, orient='index', columns=['concatenation'])
            subjects_df_1['ids'] = range(0, len(subjects_df_1))
            print("Subjects df 1: ", subjects_df_1.shape)
            print(subjects_df_1)

            subjects_df_2 = pd.DataFrame.from_dict(subjects_d2, orient='index', columns=['concatenation'])
            subjects_df_2['ids'] = range(0, len(subjects_df_2))
            print("Subjects df 2: ", subjects_df_2.shape)
            print(subjects_df_2)



            # Clustering with pyJedAI
            ccer_in_subjects_data = Data(
                dataset_1=subjects_df_1,
                attributes_1=['concatenation'],
                id_column_name_1='ids',
                dataset_2=subjects_df_2,
                attributes_2=['concatenation'],
                id_column_name_2='ids'
            )
            
            print("Dataset 1: ", subjects_df_1.shape)
            # print(d1.head())

            print("Dataset 2: ", subjects_df_2.shape)
            # print(d2.head())
            
            # if subjects_df_1.shape[0] < 2 or (not data.is_dirty_er and subjects_df_1.shape[0] < 2):
            #     print("Skipping cluster ", i)
            #     continue

            print("Running pyJedAI workflow for entity resolution on subjects")
            ccer_in_subjects_data.print_specs()


            
            try:
                pyjedai_workflow_for_er.run(data=ccer_in_subjects_data, 
                                            verbose=verbose_er,
                                            workflow_step_tqdm_disable=not verbose_er,
                                            workflow_tqdm_enable=verbose_er)
                new_clusters = pyjedai_workflow_for_er.clusters
            except Exception as e:
                print("Error in cluster ", i)
                print("Error: ", e)
                continue
            
            # print("New: ", new_clusters)
            # print("Transforming mapping to ids")
            pairs = self.transform_mapping_to_ids(new_clusters, ccer_in_subjects_data)
            global_pairs_dict.update(pairs)
            # global_pairs_df = pd.concat([global_pairs_df, pairs], ignore_index=True)
            # all_clusters += new_clusters
            
            print("Finished cluster: ", i)
            # Print pairs
            print("Predicted pairs: ", len(global_pairs_dict))
            # loop
            for pair in global_pairs_dict:
                print(pair, ":", global_pairs_dict[pair])
            print("\n\n\n")
            # print("Global pairs: ", global_pairs_dict)
        
        # print(all_clusters)
        # print(global_pairs_dict)
        self.execution_time = time.time() - _start_time
        return global_pairs_dict
    
    def pyjedai_workflow_for_er_on_subjects(self) -> PYJEDAIWorkFlow:
        return BlockingBasedWorkFlow(
            block_building = dict(
                method=StandardBlocking,
            ),
            block_cleaning = [
                dict(
                    method=BlockFiltering,
                    params=dict(ratio=0.2)
                )
            ],
            comparison_cleaning = dict(method=WeightedNodePruning),
            entity_matching = dict(method=EntityMatching, params=dict(metric='cosine',
                                                                      tokenizer='char_tokenizer', 
                                                                      vectorizer='tfidf', 
                                                                      qgram=3, 
                                                                      similarity_threshold=0.0)),
            clustering = dict(method=UniqueMappingClustering, 
                              exec_params=dict(similarity_threshold=0.1)),
            name="subjects-ccer"
        )
    
    def pyjedai_workflow_for_er_on_predicates(self) -> PYJEDAIWorkFlow:
        return BlockingBasedWorkFlow(
            block_building = dict(
                method=StandardBlocking,
            ),
            block_cleaning = [
                dict(method=BlockPurging, params=dict(smoothing_factor=1.0)),
                dict(method=BlockFiltering)
            ],
            comparison_cleaning = dict(method=WeightedNodePruning),
            entity_matching = dict(method=EntityMatching,
                                   params=dict(metric='cosine',
                                               similarity_threshold=0.0)),
            clustering = dict(method=ConnectedComponentsClustering),
            name="predicates-er"
        )
