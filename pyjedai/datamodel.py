import pandas as pd
from pandas import DataFrame, concat

class Data:
    """Class that is the contains the corpus of the dataset that will be processed with pyjedai. \
        Contains all the information of the dataset and will be passed to each step \
        of the ER workflow.
    """

    def __init__(
            self,
            dataset_1: DataFrame,
            attributes_1: list,
            id_column_name_1: str,
            dataset_2: DataFrame = None,
            attributes_2: list = None,
            id_column_name_2: str = None,
            ground_truth: DataFrame = None
    ) -> None:
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        if dataset_2 is not None and (id_column_name_2 is None or attributes_2 is None):
            raise AttributeError("Must provide datasets 2 id column")
        self.entities_d1: DataFrame
        self.entities_d2: DataFrame = None
        self.ground_truth = ground_truth.astype(str)
        self.is_dirty_er = True if dataset_2 is None else False
        self.dataset_limit = self.num_of_entities_1 = len(dataset_1)
        self.num_of_entities_2: int = len(dataset_2) if dataset_2 is not None else 0
        self.num_of_entities: int = self.num_of_entities_1 + self.num_of_entities_2
        self.attributes_1: list = attributes_1
        if dataset_2 is not None: self.attributes_2: list = attributes_2
        self.entities: DataFrame
        self.id_column_name_1 = id_column_name_1
        self.id_column_name_2 = id_column_name_2
        self._ids_mapping_1: dict
        self._gt_to_ids_reversed_1: dict
        self._ids_mapping_2: dict
        self._gt_to_ids_reversed_2: dict

    def process(self) -> None:
        """Creates the appropriate dataframes and the mapping \
            if ground-truth file has been provided.
        """
        self.entities = self.dataset_1 = self.dataset_1.astype(str)
        self.entities_d1 = self.dataset_1[self.attributes_1].apply(" ".join, axis=1)
        if not self.is_dirty_er:
            self.dataset_2 = self.dataset_2.astype(str)
            self.entities_d2 = self.dataset_2[self.attributes_2].apply(" ".join, axis=1)
            self.entities = concat([self.dataset_1, self.dataset_2])
        self._create_gt_mapping()

    def _create_gt_mapping(self) -> None:
        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.astype(str)
        else:
            return

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

    def print_specs(self):
        """Dataset report.
        """
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
    """The main module used for storing entities in the blocking steps of pyjedai module. \
        Consists of 2 sets of profile entities 1 for Dirty ER and 2 for Clean-Clean ER.
    """
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
