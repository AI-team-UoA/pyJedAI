import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from pyjedai.datamodel import Data


def test_clean_dataset_keeps_identifier_columns_and_cached_mappings_consistent():
    data = Data(
        dataset_1=pd.DataFrame({"id": ["A-1"], "name": ["Alice-1"]}),
        id_column_name_1="id",
        ground_truth=pd.DataFrame([["A-1", "A-1"]]),
    )

    data.clean_dataset(
        remove_numbers=True,
        remove_punctuation=True,
        remove_stopwords=False,
        remove_unicodes=False,
    )

    assert data.dataset_1.loc[0, "id"] == "A-1"
    assert data.dataset_1.loc[0, "name"] == "alice"
    assert list(data._ids_mapping_1.keys()) == ["A-1"]
    assert list(data.duplicate_of.keys()) == ["A-1"]
    assert data._ids_mapping_1["A-1"] == 0
    assert data._are_true_positives("A-1", "A-1")
