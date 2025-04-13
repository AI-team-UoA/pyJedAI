import pandas as pd
import pytest

from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding


@pytest.fixture(scope="module")
def data():
    # Mock data object
    return Data(
        dataset_1=pd.DataFrame({"attribute_1": ["entity1", "entity2"], "id": [1, 2]}),
        dataset_2=pd.DataFrame({"attribute_1": ["entity3", "entity4"], "id": [3, 4]}),
        attributes_1=["attribute_1"],
        attributes_2=["attribute_1"],
        id_column_name_1="id",
        id_column_name_2="id",
    )

def test_custom_vectorizer_standard_model_raises_error(data):
    instance = EmbeddingsNNBlockBuilding(vectorizer="word2vec")

    with pytest.raises(AttributeError):
        instance.build_blocks(data, custom_pretrained_model="word")

def test_custom_vectorizer_word_embeddings(data, mocker):
    instance = EmbeddingsNNBlockBuilding(vectorizer='custom')
    # mock the tokenizer and model loading
    assert instance.vectorizer == 'custom'
    mock_tokenizer = mocker.patch("pyjedai.vector_based_blocking.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("pyjedai.vector_based_blocking.AutoModel.from_pretrained")
    instance.build_blocks(data, custom_pretrained_model="word")
    # Check if the tokenizer and model were called once
    assert mock_tokenizer.call_count == 1
    assert mock_model.call_count == 1


def test_custom_vectorizer_sentence_embeddings(data, mocker):
    instance = EmbeddingsNNBlockBuilding(vectorizer='custom')
    assert instance.vectorizer == 'custom'
    mock_transformer = mocker.patch("pyjedai.vector_based_blocking.SentenceTransformer")

    # check if the transformer was called with the correct model
    instance.build_blocks(data, custom_pretrained_model="sentence")
    assert mock_transformer.call_count == 1
    assert mock_transformer.call_args[0][0] == "custom"