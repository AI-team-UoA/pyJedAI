"""Schema Matching methods
"""
import pandas as pd
import valentine
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.algorithms.coma.coma import Coma
from valentine.algorithms.cupid.cupid_model import Cupid
from valentine.algorithms.distribution_based.distribution_based import DistributionBased
from valentine.algorithms.jaccard_distance.jaccard_distance import JaccardDistanceMatcher
from valentine.algorithms.similarity_flooding.similarity_flooding import SimilarityFlooding
import valentine.metrics as valentine_metrics
from pandas import DataFrame, concat

from ..datamodel import Block, SchemaData, PYJEDAIFeature
from ..evaluation import Evaluation
from abc import abstractmethod

class AbstractSchemaMatching(PYJEDAIFeature):
    """Abstract class for schema matching methods
    """

    @abstractmethod
    def evaluate(self,
                 prediction=None,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        pass

    @abstractmethod
    def _configuration(self) -> dict:
        pass

    @abstractmethod
    def stats(self) -> None:
        pass

    @abstractmethod
    def process(self,
                data: SchemaData,
                ) -> list:
        pass

    @abstractmethod
    def process_sm_weighted(self,
                            data: SchemaData):
        pass

    def __init__(self):
        super().__init__()


class ValentineMethodBuilder(PYJEDAIFeature):
    """Class to provide valentine matching methods
    """

    def evaluate(self,
                 prediction=None,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        pass

    def _configuration(self) -> dict:
        pass

    def __init__(self):
        super().__init__()

    @staticmethod
    def coma_matcher(max_n: int = 0,
                     strategy: str = "COMA_OPT"
                     ) -> Coma:
        return Coma(max_n, strategy)

    @staticmethod
    def cupid_matcher(w_struct: float = 0.2,
                      leaf_w_struct: float = 0.2,
                      th_accept: float = 0.7
                      ) -> Cupid:
        return Cupid(w_struct, leaf_w_struct, th_accept)

    @staticmethod
    def distribution_based_matcher(threshold1: float = 0.15,
                                   threshold2: float = 0.15
                                   ) -> DistributionBased:
        return DistributionBased(threshold1, threshold2)

    @staticmethod
    def jaccard_distance_matcher(threshold_leven: float = 0.8) -> JaccardDistanceMatcher:
        return JaccardDistanceMatcher(threshold_leven)

    @staticmethod
    def similarity_flooding_mathcer(coeff_policy: str = "inverse_average",
                                    formula: str = "formula_c") -> SimilarityFlooding:
        return SimilarityFlooding(coeff_policy, formula)

class ValentineSchemaMatching(AbstractSchemaMatching):
    """Class for schema matching methods provided by Valentine
    """

    def __init__(self, matcher: BaseMatcher):
        super().__init__()
        self.data: SchemaData = None
        self.matcher: BaseMatcher = matcher
        self.matches = None
        self.top_columns: list = []

    def process(self,
                data: SchemaData,
                ) -> list:
        self.data = data
        df1 = self.data.dataset_1
        df2 = self.data.dataset_2
        self.matches = valentine.valentine_match(df1, df2, self.matcher)
        self.top_columns = [[x[0][1] for x in self.matches.keys()], [x[1][1] for x in self.matches.keys()]]
        return self.top_columns

    def process_sm_weighted(self, data: SchemaData):
        pass

    def print_matches(self):
        for match, sim in self.matches.items():
            print(match, " - ", sim)
        
    def get_matches(self) -> dict:
        return self.matches

    def evaluate(self,
                 prediction=None,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:

        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " +
                                 "Data object has not been initialized with the ground-truth file")

        return valentine_metrics.all_metrics(self.matches, self.data.ground_truth)

    def _configuration(self) -> dict:
        pass

    def stats(self) -> None:
        pass

