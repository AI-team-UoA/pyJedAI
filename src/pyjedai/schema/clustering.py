import pandas as pd
import time as time

from tqdm import tqdm   
from ..datamodel import Data, PYJEDAIFeature
from ..evaluation import Evaluation
from ..workflow import PYJEDAIWorkFlow, BlockingBasedWorkFlow
from ..clustering import AbstractClustering
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

    def stats(self) -> None:
        pass

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
        # print(entities_d1)

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
            # print(entities_d2)

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
            dataset_2=attributes_d2,
            attributes_2=['attribute'],
            id_column_name_2='ids'
        )
        
        # attributes_data.print_specs()
        
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

        # print("\n\n\n Clusters: ", clusters)

        def contains_attributes_from_both(limit, cluster):
            has_entity_from_d1 = any(num < limit for num in cluster)
            has_entity_from_d2 = any(num >= limit for num in cluster)

            return has_entity_from_d1 and has_entity_from_d2
        
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
        # print(new_datasets)
        if return_clusters:
            return new_datasets
        
        if pyjedai_workflow_for_er == None:
            pyjedai_workflow_for_er = BlockingBasedWorkFlow()
            if data.is_dirty_er:
                pyjedai_workflow_for_er.best_blocking_workflow_der() 
            else:
                pyjedai_workflow_for_er.best_blocking_workflow_ccer()

        self.entity_resolution_workflow = pyjedai_workflow_for_er.name

        all_clusters = []
        for i in tqdm(range(len(new_datasets)), desc="Entity resolution for clusters"):
            d1, d2 = new_datasets[i]
            new_data = Data(
                dataset_1=d1,
                attributes_1=data.attributes_1,
                id_column_name_1=data.id_column_name_1,
                dataset_2=d2,
                attributes_2=data.attributes_2,
                id_column_name_2=data.id_column_name_2,
                ground_truth=data.ground_truth
            )
            pyjedai_workflow_for_er.run(new_data, verbose=verbose_er)
            new_clusters = pyjedai_workflow_for_er.clusters
            all_clusters += new_clusters

        self.execution_time = time.time() - _start_time
        return all_clusters
