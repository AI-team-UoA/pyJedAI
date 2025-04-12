import pytest
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.datamodel import Data
import pandas as pd

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

    # Create an instance of the class with vectorizer 'custom'
    instance = EmbeddingsNNBlockBuilding(vectorizer='custom')

    # Check if the instance is created successfully
    assert instance.vectorizer == 'custom'
    mock_tokenizer = mocker.patch("transformers.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("transformers.AutoModel.from_pretrained")
    # Attempt to call build_blocks and expect an AttributeError
    instance.build_blocks(data, custom_pretrained_model="word")
    assert mock_tokenizer.call_count == 1
    assert mock_model.call_count == 1


def test_custom_vectorizer_sentence_embeddings(data):

    # Create an instance of the class with vectorizer 'custom'
    instance = EmbeddingsNNBlockBuilding(vectorizer='custom')

    # Check if the instance is created successfully
    assert instance.vectorizer == 'custom'

    # Attempt to call build_blocks and expect an AttributeError
    instance.build_blocks(data, custom_pretrained_model="sentence")