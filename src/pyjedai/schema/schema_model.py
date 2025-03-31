import pandas as pd
from pandas import DataFrame
from ..datamodel import Data
from .utils import vectorFromTFIDF

class Schema(Data): 
    
    
    """
        The corpus of the dataset that will be processed with pyjedai. \
        Contains all the information of the dataset and will be passed to each step \
        of the ER workflow. 
    """

    def __init__(
                self,
                dataset_1: DataFrame,
                dataset_2: DataFrame,
                dataset_name_1: str = None,
                dataset_name_2: str = None,
                ground_truth: DataFrame = None,
                skip_ground_truth_processing: bool = False,
                matching_type: str = None,
    ) -> None:
        
        # Original Datasets as pd.DataFrame
        if not isinstance(dataset_1, pd.DataFrame):
            raise AttributeError("Dataset 1 must be a pandas DataFrame")

        if not isinstance(dataset_2, pd.DataFrame):
            raise AttributeError("Dataset 2 must be a pandas DataFrame")
            
        
        if matching_type == 'CONTENT':
            dataset_1, dataset_2, ground_truth = self.load_content(dataset_1, dataset_2, ground_truth, skip_ground_truth_processing)
        elif matching_type == 'COMPOSITE': 
            dataset_1, dataset_2, ground_truth = self.load_composite(dataset_1, dataset_2)
        else:
            dataset_1, dataset_2, ground_truth = self.load_schema(dataset_1, dataset_2, ground_truth) 
            
        super().__init__(dataset_1 = dataset_1, 
                    id_column_name_1 = 'id',
                    dataset_name_1 = dataset_name_1,
                    dataset_2 = dataset_2, 
                    id_column_name_2 = 'id',
                    dataset_name_2 = dataset_name_2,
                    ground_truth = ground_truth,
                    skip_ground_truth_processing = skip_ground_truth_processing)
        


    def load_content(self,
                dataset_1: DataFrame, 
                dataset_2: DataFrame,
                ground_truth: DataFrame = None,
                skip_ground_truth_processing: bool = False) -> tuple:
        

        dataset_1 = dataset_1.astype(str)
        dataset_2 = dataset_2.astype(str)

        source_attributes = dataset_1.columns
        target_attributes = dataset_2.columns

        source_index = range(len(source_attributes)) 
        source_data = [vectorFromTFIDF(dataset_1, col) for col in source_attributes]

        target_index = range(len(target_attributes)) 
        target_data = [vectorFromTFIDF(dataset_2, col) for col in target_attributes]


        dataset_1 = pd.DataFrame({
            'id' : source_index,
            'data': source_data
        })

        dataset_2 = pd.DataFrame({
            'id' : target_index,
            'data': target_data
        })


        dataset_1_columns = pd.DataFrame({
            'source': source_attributes,
            "source_index": source_index
        })

        dataset_2_columns = pd.DataFrame({ 
            "target" : target_attributes,
            "target_index": target_index
        })


        if ground_truth is not None and not skip_ground_truth_processing:
            ground_truth.columns = ['source', 'target']
            ground_truth = pd.merge(ground_truth, dataset_1_columns, on="source", how='left')
            ground_truth = pd.merge(ground_truth, dataset_2_columns, on='target', how='left')
            ground_truth = ground_truth.drop(columns=['source', 'target'])

        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.ground_truth = ground_truth      
        return dataset_1, dataset_2, ground_truth
    

    def load_composite(
            dataset_1: DataFrame, 
            dataset_2: DataFrame,
            ground_truth: DataFrame = None,
            skip_ground_truth_processing: bool = False
    ) -> tuple:
        dataset_1 = dataset_1.astype(str)
        dataset_2 = dataset_2.astype(str)

        source_attributes = dataset_1.columns
        target_attributes = dataset_2.columns

        source_index = range(len(source_attributes)) 
        source_data = [vectorFromTFIDF(dataset_1, col) for col in source_attributes]

        target_index = range(len(target_attributes)) 
        target_data = [vectorFromTFIDF(dataset_2, col) for col in target_attributes]


        dataset_1 = pd.DataFrame({
            'id' : source_index,
            'attributes': source_attributes,
            'data': source_data
        })

        dataset_2 = pd.DataFrame({
            'id' : target_index,
            'attributes': target_attributes,
            'data': target_data
        })


        dataset_1_columns = pd.DataFrame({
            'source': source_attributes,
            "source_index": source_index
        })

        dataset_2_columns = pd.DataFrame({ 
            "target" : target_attributes,
            "target_index": target_index
        })


        if ground_truth is not None and not skip_ground_truth_processing:
            ground_truth.columns = ['source', 'target']
            ground_truth = pd.merge(ground_truth, dataset_1_columns, on="source", how='left')
            ground_truth = pd.merge(ground_truth, dataset_2_columns, on='target', how='left')
            ground_truth = ground_truth.drop(columns=['source', 'target'])

        return dataset_1, dataset_2, ground_truth
    
        

    def load_schema(
            dataset_1: DataFrame, 
            dataset_2: DataFrame,
            ground_truth: DataFrame = None,
            skip_ground_truth_processing: bool = False
    ) -> tuple:
        dataset_1 = dataset_1.astype(str)
        dataset_2 = dataset_2.astype(str)

        source_attributes = dataset_1.columns
        target_attributes = dataset_2.columns

        source_index = range(len(source_attributes)) 

        target_index = range(len(target_attributes)) 


        dataset_1 = pd.DataFrame({
            'id' : source_index,
            'attributes': source_attributes,
        })

        dataset_2 = pd.DataFrame({
            'id' : target_index,
            'attributes': target_attributes,
        })


        dataset_1_columns = pd.DataFrame({
            'source': source_attributes,
            "source_index": source_index
        })

        dataset_2_columns = pd.DataFrame({ 
            "target" : target_attributes,
            "target_index": target_index
        })


        if ground_truth is not None and not skip_ground_truth_processing:
            ground_truth.columns = ['source', 'target']
            ground_truth = pd.merge(ground_truth, dataset_1_columns, on="source", how='left')
            ground_truth = pd.merge(ground_truth, dataset_2_columns, on='target', how='left')
            ground_truth = ground_truth.drop(columns=['source', 'target'])

        return dataset_1, dataset_2, ground_truth