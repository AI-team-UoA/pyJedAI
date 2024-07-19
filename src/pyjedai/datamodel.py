"""Datamodel of pyjedai.
"""
import pandas as pd
from pandas import DataFrame

import re
import csv

import nltk
from nltk.corpus import stopwords

from abc import ABC, abstractmethod
from collections import defaultdict
from ordered_set import OrderedSet
from tqdm import tqdm

from shapely.geometry import shape
from shapely.wkt import loads

class PYJEDAIFeature(ABC):

    _method_name: str
    _method_info: str
    _method_short_name: str

    def __init__(self) -> None:
        super().__init__()
        self._progress_bar: tqdm
        self.execution_time: float
        self.tqdm_disable: bool
        self.data: Data

    @abstractmethod
    def _configuration(self) -> dict:
        pass

    @abstractmethod
    def evaluate(self,
                 prediction=None,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        pass
    
    def method_configuration(self) -> dict:
        """Returns configuration details
        """
        return {
            "name" : self._method_name,
            "parameters" : self._configuration(),
            "runtime": self.execution_time
        }
        
    def report(self) -> None:
        """Prints Block Building method configuration
        """
        parameters = ("\n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()])) \
                        if len(self._configuration().items()) != 0 else ' None'
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            "\nParameters: " + parameters +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

    @abstractmethod
    def stats(self) -> None:
        pass

class Data:
    """The corpus of the dataset that will be processed with pyjedai. \
        Contains all the information of the dataset and will be passed to each step \
        of the ER workflow.
    """

    def __init__(
                self,
                dataset_1: DataFrame,
                id_column_name_1: str,                
                attributes_1: list = None,
                dataset_name_1: str = None,
                dataset_2: DataFrame = None,
                attributes_2: list = None,
                id_column_name_2: str = None,
                dataset_name_2: str = None,
                ground_truth: DataFrame = None,
                skip_ground_truth_processing: bool = False
    ) -> None:
        # Original Datasets as pd.DataFrame
        if isinstance(dataset_1, pd.DataFrame):
            self.dataset_1 = dataset_1
        else:
            raise AttributeError("Dataset 1 must be a pandas DataFrame")

        if dataset_2 is not None:
            if id_column_name_2 is None:
                raise AttributeError("Must provide datasets 2 id column")

            if isinstance(dataset_2, pd.DataFrame):
                self.dataset_2 = dataset_2
            else:
                raise AttributeError("Dataset 2 must be a pandas DataFrame")

        # Processed dataframes to lists (all attribute columns)
        # Tranformed to list for optimization (list)
        self.entities_d1: list
        self.entities_d2: list = None

        # D1 and D2 dataframes concatenated
        self.entities: DataFrame

        # Datasets specs
        self.is_dirty_er = dataset_2 is None
        self.dataset_limit = self.num_of_entities_1 = len(dataset_1)
        self.num_of_entities_2: int = len(dataset_2) if dataset_2 is not None else 0
        self.num_of_entities: int = self.num_of_entities_1 + self.num_of_entities_2

        self.id_column_name_1 = id_column_name_1
        self.id_column_name_2 = id_column_name_2
        
        self.dataset_name_1 = dataset_name_1
        self.dataset_name_2 = dataset_name_2
        
        # Fill NaN values with empty string
        self.dataset_1.fillna("", inplace=True)
        self.dataset_1 = self.dataset_1.astype(str)
        if not self.is_dirty_er:
            self.dataset_2.fillna("", inplace=True)
            self.dataset_2 = self.dataset_2.astype(str)
            
        # Attributes
        if attributes_1 is None:
            if dataset_1.columns.values.tolist():
                self.attributes_1 = dataset_1.columns.values.tolist()
                if self.id_column_name_1 in self.attributes_1:
                    self.attributes_1.remove(self.id_column_name_1)
            else:
                raise AttributeError(
                    "Dataset 1 must contain column names if attributes_1 is empty.")
        else:
            self.attributes_1: list = attributes_1

        if dataset_2 is not None:
            if attributes_2 is None:
                if dataset_2.columns.values.tolist():
                    self.attributes_2 = dataset_2.columns.values.tolist()
                    if self.id_column_name_2 in self.attributes_2:
                        self.attributes_2.remove(self.id_column_name_2)
                else:
                    raise AttributeError("Dataset 2 must contain column names if attributes_2 is empty.")
            else:
                self.attributes_2: list = attributes_2

        # Ground truth data
        self.skip_ground_truth_processing = skip_ground_truth_processing
        if ground_truth is not None and not skip_ground_truth_processing:
            self.ground_truth = ground_truth.astype(str)
            self.ground_truth.drop_duplicates(inplace=True)
            self._ids_mapping_1: dict
            self._gt_to_ids_reversed_1: dict
            self._ids_mapping_2: dict
            self._gt_to_ids_reversed_2: dict
        else:
            self.ground_truth = None
        
        self.entities = self.dataset_1 = self.dataset_1.astype(str)
        
        # Concatenated columns into new dataframe
        self.entities_d1 = self.dataset_1[self.attributes_1]
        
        if not self.is_dirty_er:
            self.dataset_2 = self.dataset_2.astype(str)
            self.entities_d2 = self.dataset_2[self.attributes_2]
            self.entities = pd.concat([self.dataset_1, self.dataset_2],
                                      ignore_index=True)
        # if not skip_ground_truth_processing:
        self._create_gt_mapping()
        if ground_truth is not None:
            if skip_ground_truth_processing:
                self.ground_truth = ground_truth
            else:
                self._store_pairs()
        else:
            self.ground_truth = None
    
    def _store_pairs(self) -> None:
        """Creates a mapping:
            - pairs_of : ids of first dataset to ids of true matches from second dataset"""
        self.duplicate_of = defaultdict(set)
        
        for _, row in self.ground_truth.iterrows():
            id1, id2 = (row.iloc[0], row.iloc[1])
            if id1 in self.duplicate_of: self.duplicate_of[id1].add(id2)
            else: self.duplicate_of[id1] = {id2}
            
    def _are_true_positives(self, id1 : int, id2 : int):
        """Checks if given pair of identifiers represents a duplicate.
           Identifiers must be inorder, first one belonging to the first and the second to the second dataset

        Args:
            id1 (int, optional): Identifier from the first dataframe. 
            id2 (int, optional): Identifier from the second dataframe.

        Returns:
            _type_: _description_
        """
        return id1 in self.duplicate_of and id2 in self.duplicate_of[id1]
    
    def _create_gt_mapping(self) -> None:
        """Creates two mappings:
            - _ids_mapping_X: ids from initial dataset to index
            - _gt_to_ids_reversed_X (inversed _ids_mapping_X): index number \
                            from range to initial dataset id
        """
        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.astype(str)
        # else:
        #     return
        self._ids_mapping_1 = dict(
            zip(
                self.dataset_1[self.id_column_name_1].tolist(),
                range(0, self.num_of_entities_1)
            )
        )

        self._gt_to_ids_reversed_1 = dict(
            zip(
                self._ids_mapping_1.values(),
                self._ids_mapping_1.keys()
            )
        )

        if not self.is_dirty_er:
            self._ids_mapping_2 = dict(
                zip(
                    self.dataset_2[self.id_column_name_2].tolist(),
                    range(self.num_of_entities_1, self.num_of_entities_1+self.num_of_entities_2)
                )
            )

            self._gt_to_ids_reversed_2 = dict(
                zip(
                    self._ids_mapping_2.values(),
                    self._ids_mapping_2.keys()
                )
            )
            
    def get_pyjedai_id_of(self, dataset_id: any) -> int:
        pass
    
    def get_real_id_of(self, pyjedai_id: int) -> any:
        if pyjedai_id < self.dataset_limit:
            return self._gt_to_ids_reversed_1[pyjedai_id]
        else:
            return self._gt_to_ids_reversed_2[pyjedai_id]
        
    def print_specs(self) -> None:
        """Dataset report.
        """
        def calculate_memory_usage_of_pandas(dataframe: pd.DataFrame) -> float:
            memory_usage = dataframe.memory_usage(deep=True).sum()
            if memory_usage > 1024**4:
                memory_usage /= (1024**4)
                unit = "TB"
            elif memory_usage > 1024**3:
                memory_usage /= (1024**3)
                unit = "GB"
            elif memory_usage > 1024**2:
                memory_usage /= (1024**2)
                unit = "MB"
            elif memory_usage > 1024:
                memory_usage /= (1024)
                unit = "KB"
            else:
                unit = "B"
            
            return memory_usage, unit

        print('*' * 123)
        print(' ' * 50, 'Data Report')
        print('*' * 123)
        print("Type of Entity Resolution: ", "Dirty" if self.is_dirty_er else "Clean-Clean" )
        name1 = self.dataset_name_1 if self.dataset_name_1 is not None else "D1"
        print("Dataset 1 (" + name1 + "):")
        print("\tNumber of entities: ", self.num_of_entities_1)
        print("\tNumber of NaN values: ", self.dataset_1.isnull().sum().sum())
        memory_usage, unit = calculate_memory_usage_of_pandas(self.dataset_1)
        print("\tMemory usage [" + unit + "]: ", "{:.2f}".format(memory_usage))
        print("\tAttributes:")
        for attr in self.attributes_1:
            print("\t\t", attr)
        if not self.is_dirty_er:
            name2 = self.dataset_name_2 if self.dataset_name_2 is not None else "D2"
            print("Dataset 2 (" + name2 + "):")
            print("\tNumber of entities: ", self.num_of_entities_2)
            print("\tNumber of NaN values: ", self.dataset_2.isnull().sum().sum())
            memory_usage, unit = calculate_memory_usage_of_pandas(self.dataset_2)
            print("\tMemory usage [" + unit + "]: ", "{:.2f}".format(memory_usage))
            print("\tAttributes:")
            for attr in self.attributes_2:
                print("\t\t", attr)
        print("\nTotal number of entities: ", self.num_of_entities)
        if self.ground_truth is not None:
            print("Number of matching pairs in ground-truth: ", len(self.ground_truth))
        print(u'\u2500' * 123)
    
    # Functions that removes stopwords, punctuation, uni-codes, numbers from the dataset
    def clean_dataset(self, 
                      remove_stopwords: bool = True, 
                      remove_punctuation: bool = True, 
                      remove_numbers:bool = True,
                      remove_unicodes: bool = True) -> None:
        """Removes stopwords, punctuation, uni-codes, numbers from the dataset.
        """
        nltk.download('stopwords')

        # Make self.dataset_1 and self.dataset_2 lowercase
        self.dataset_1 = self.dataset_1.applymap(lambda x: x.lower())
        if not self.is_dirty_er:
            self.dataset_2 = self.dataset_2.applymap(lambda x: x.lower())
            
        if remove_numbers:
            self.dataset_1 = self.dataset_1.applymap(lambda x: re.sub(r'\d+', '', x))
            if not self.is_dirty_er:
                self.dataset_2 = self.dataset_2.applymap(lambda x: re.sub(r'\d+', '', x))    
                
        if remove_unicodes:
            self.dataset_1 = self.dataset_1.applymap(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
            if not self.is_dirty_er:
                self.dataset_2 = self.dataset_2.applymap(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
            
        if remove_punctuation:
            self.dataset_1  = self.dataset_1.applymap(lambda x: re.sub(r'[^\w\s]','',x))
            if not self.is_dirty_er:
                self.dataset_2 = self.dataset_2.applymap(lambda x: re.sub(r'[^\w\s]','',x))
        
        if remove_stopwords:
            self.dataset_1 = self.dataset_1.applymap(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))
            if not self.is_dirty_er:
                self.dataset_2 = self.dataset_2.applymap(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))    

        self.entities = self.dataset_1 = self.dataset_1.astype(str)
        
        # Concatenated columns into new dataframe
        self.entities_d1 = self.dataset_1[self.attributes_1]

        if not self.is_dirty_er:
            self.dataset_2 = self.dataset_2.astype(str)
            self.entities_d2 = self.dataset_2[self.attributes_2]
            self.entities = pd.concat([self.dataset_1, self.dataset_2],
                                      ignore_index=True)

    def stats_about_data(self) -> None:
        
        stats_df = pd.DataFrame(columns=['word_count_1', 'word_count_2'])
        
        # Calculate the average number of words per line
        stats_df['word_count_1'] = self.dataset_1.apply(lambda row: len(row.str.split()), axis=1)
        print(stats_df['word_count_1'])
        average_words_per_line_1 = stats_df['word_count_1'].mean()
        print(average_words_per_line_1)
        
        if not self.is_dirty_er:
            stats_df['word_count_2'] = self.dataset_2.apply(lambda row: len(row.str.split()), axis=1)
            average_words_per_line_2 = stats_df['word_count_2'].mean()
            print(average_words_per_line_2)
            
        return stats_df

class SpatialData:
    def __init__(
                self,
                source_reader: csv.reader,
                source_delimiter: str,
                target_reader: csv.reader,
                target_delimiter: str,
                skip_header: bool=False
    ) -> None:
        self.source_geometriesSize = 0
        self.source_reader = source_reader
        self.source_delimiter = source_delimiter

        self.targetGeometriesSize = 0
        self.target_reader = target_reader
        self.target_delimiter = target_delimiter

        self.skip_header = skip_header
        self.source_geometries = []
        self.targetGeometries = []

        self.readSourceGeometries()
        self.readTargetGeometries()
        return
    
    def readSourceGeometries(self) -> list:
        geometries_loaded = 0
        geometries_failed = 0
        geoCollections = 0

        if(self.skip_header == True):
            next(self.source_reader)

        for geometry in self.source_reader:
            try:
                geometry, *information = [s.split(self.source_delimiter)[0] for s in geometry]
                geometry = shape(loads(geometry))
            except:
                geometries_failed += 1
                continue

            if geometry.geom_type == "GeometryCollection":
                geoCollections += 1
            else:
                self.source_geometries.append(geometry)
                geometries_loaded += 1

        # print("SpatialData initialized:","\n Geometries loaded:", geometries_loaded, "\n Geometries failed:", geometries_failed, "\n GeoCollections found:", geoCollections,"\n")
        self.source_geometries_size = geometries_loaded
        return

    def readTargetGeometries(self) -> list:
        geometries_loaded = 0
        geometries_failed = 0
        geoCollections = 0

        if(self.skip_header == True):
            next(self.target_reader)

        for geometry in self.target_reader:
            try:
                geometry, *information = [s.split(self.target_delimiter)[0] for s in geometry]
                geometry = shape(loads(geometry))
            except:
                geometries_failed += 1
                continue

            if geometry.geom_type == "GeometryCollection":
                geoCollections += 1
            else:
                self.targetGeometries.append(geometry)
                geometries_loaded += 1

        # print("SpatialData initialized:","\n Geometries loaded: ", geometries_loaded, "\n Geometries failed: ", geometries_failed, "\n GeoCollections found: ", geoCollections)
        self.targetGeometriesSize = geometries_loaded
        return

class SchemaData:
    """Data module for schema matching tasks. Valentine-based structure.
    """

    def __init__(
                self,
                dataset_1: DataFrame,
                attributes_1: list,
                dataset_2: DataFrame,
                attributes_2: list,
                dataset_name_1: str = None,
                dataset_name_2: str = None,
                ground_truth: DataFrame = None,
    ) -> None:
        # Original Datasets as pd.DataFrame
        if isinstance(dataset_1, pd.DataFrame):
            self.dataset_1 = dataset_1

        else:
            raise AttributeError("Dataset 1 must be a pandas DataFrame")

        if dataset_2 is not None:
            if isinstance(dataset_2, pd.DataFrame):
                self.dataset_2 = dataset_2
            else:
                raise AttributeError("Dataset 2 must be a pandas DataFrame")
        
        if ground_truth is not None:
            self.ground_truth = ground_truth.to_records(index=False).tolist()

class Block:
    """The main module used for storing entities in the blocking steps of pyjedai module. \
        Consists of 2 sets of profile entities 1 for Dirty ER and 2 for Clean-Clean ER.
    """
    def __init__(self) -> None:
        self.entities_D1: set = OrderedSet()
        self.entities_D2: set = OrderedSet()

    def get_cardinality(self, is_dirty_er) -> int:
        """Returns block cardinality.

        Args:
            is_dirty_er (bool): Dirty or Clean-Clean ER.

        Returns:
            int: Cardinality
        """
        if is_dirty_er:
            return len(self.entities_D1)*(len(self.entities_D1)-1)/2
        return len(self.entities_D1) * len(self.entities_D2)

    def get_size(self) -> int:
        """Returns block size.

        Returns:
            int: Block size
        """
        return len(self.entities_D1) + len(self.entities_D2)

    def verbose(self, key: any, is_dirty_er: bool) -> None:
        """Prints contents of a block

        Args:
            key (any): Block key
            is_dirty_er (bool): Dirty or Clean-Clean ER.
        """
        print("\nBlock ", "\033[1;32m"+key+"\033[0m", " has cardinality ", str(self.get_cardinality(is_dirty_er)) ," and contains entities with ids: ")
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

