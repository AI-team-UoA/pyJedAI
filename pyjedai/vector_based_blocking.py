import sys
import warnings
from time import time
from typing import List

import faiss
import gensim.downloader as api
import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm
from transformers import (AlbertModel, AlbertTokenizer, BertModel,
                          BertTokenizer, DistilBertModel, DistilBertTokenizer,
                          RobertaModel, RobertaTokenizer, XLNetModel,
                          XLNetTokenizer)

transformers.logging.set_verbosity_error()

from .block_building import StandardBlocking
from .datamodel import Data

LINUX_ENV=False
try:
    if 'linux' in sys.platform:
        import falconn
        import scann
        LINUX_ENV=True
except:
    warnings.warn(ImportWarning, "Can't use FALCONN/SCANN in windows environment")

class EmbeddingsNNBlockBuilding(StandardBlocking):
    """Block building via creation of embeddings and a Nearest Neighbor Approach.
    """

    _method_name = "Embeddings-NN Block Building"
    _method_info = "Creates a set of candidate pais for every entity id " + \
        "based on Embeddings creariot and Similarity search among the vectors."

    _gensim_mapping_download = {
        'fasttext' : 'fasttext-wiki-news-subwords-300',
        'glove' : 'glove-wiki-gigaword-300',
        'word2vec' : 'word2vec-google-news-300'
    }
    _sentence_transformer_mapping = {
        'smpnet' : 'all-mpnet-base-v2',
        'st5' : 'gtr-t5-large',
        'sdistilroberta' : 'all-distilroberta-v1',
        'sminilm' : 'all-MiniLM-L12-v2',
        'glove' : 'average_word_embeddings_glove.6B.300d'
    }

    def __init__(
            self,
            vectorizer: str,
            similarity_search: str
    ) -> None:
        self.vectorizer, self.similarity_search = vectorizer, similarity_search
        self.embeddings: np.array
        self.vectors_1: np.array
        self.vectors_2: np.array = None
        self.vector_size: int
        self.num_of_clusters: int
        self.top_k: int

    def build_blocks(
            self,
            data: Data,
            vector_size: int = 300,
            num_of_clusters: int = 5,
            top_k: int = 30,
            attributes_1: list = None,
            attributes_2: list = None,
            tqdm_disable: bool = False
    ) -> dict:
        """Main method of the vector based approach. Contains two steps. First an embedding method. \
            And afterwards a similarity search upon the vectors created in the previous step.
            Pre-trained schemes are used for the embedding process.

        Args:
            data (Data): dataset from datamodel
            vector_size (int, optional): For the Gensim vectorizers. Defaults to 300. \
                Qaution for the hugging face embeddings has no effect.
            num_of_clusters (int, optional): Number of clusters for FAISS. Defaults to 5.
            top_k (int, optional): Top K similar candidates. Defaults to 30.
            attributes_1 (list, optional): Vectorization of specific attributes for D1. Defaults to None.
            attributes_2 (list, optional): Vectorization of specific attributes for D2. Defaults to None.
            tqdm_disable (bool, optional): Disable progress bar. For experiment purposes. Defaults to False.

        Raises:
            AttributeError: Vectorizer check
            AttributeError: Similarity Search method check.

        Returns:
            dict: Entity ids to sets of top-K candidate ids.
        """
        _start_time = time()
        self.blocks = dict()
        self.data, self.attributes_1, self.attributes_2, self.vector_size, self.num_of_clusters, self.top_k \
            = data, attributes_1, attributes_2, vector_size, num_of_clusters, top_k
        self._progress_bar = tqdm(
            total=data.num_of_entities, desc=self._method_name, disable=tqdm_disable
        )
        if attributes_1:
            isolated_attr_dataset_1 = data.dataset_1[attributes_1].apply(" ".join, axis=1)
        if attributes_2:
            isolated_attr_dataset_2 = data.dataset_2[attributes_1].apply(" ".join, axis=1)

        vectors_1 = []
        if self.vectorizer in ['word2vec', 'fasttext', 'doc2vec', 'glove']:
            # ------------------- #
            # Gensim Embeddings
            # More: https://github.com/RaRe-Technologies/gensim-data
            # ------------------- #
            vocabulary = api.load(self._gensim_mapping_download[self.vectorizer])
            for i in range(0, data.num_of_entities_1, 1):
                record = isolated_attr_dataset_1.iloc[i] if attributes_1 \
                            else data.entities_d1.iloc[i]
                vectors_1.append(
                    self._create_vector(self._tokenize_entity(record), vocabulary)
                )
                self._progress_bar.update(1)
            self.vectors_1 = np.array(vectors_1).astype('float32')
            if not data.is_dirty_er:
                vectors_2 = []
                for i in range(0, data.num_of_entities_2, 1):
                    record = isolated_attr_dataset_2.iloc[i] if attributes_2 \
                                else data.entities_d2.iloc[i]
                    vectors_2.append(
                        self._create_vector(self._tokenize_entity(record), vocabulary)
                    )
                    self._progress_bar.update(1)
                self.vectors_2 = np.array(vectors_2).astype('float32')
        elif self.vectorizer in ['bert', 'distilbert', 'roberta', 'xlnet', 'albert']:
            # ---------------------------- #
            # Pre-trained Word Embeddings
            # ---------------------------- #
            if self.vectorizer == 'bert':
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model = BertModel.from_pretrained("bert-base-uncased")
            elif self.vectorizer == 'distilbert':
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            elif self.vectorizer == 'roberta':
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                model = RobertaModel.from_pretrained('roberta-base')
            elif self.vectorizer == 'xlnet':
                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
                model = XLNetModel.from_pretrained('xlnet-base-cased')
            elif self.vectorizer == 'albert':
                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                model = AlbertModel.from_pretrained("albert-base-v2")
            for i in range(0, data.num_of_entities_1, 1):
                record = isolated_attr_dataset_1.iloc[i] if attributes_1 \
                            else data.entities_d1.iloc[i]
                encoded_input = tokenizer(
                    record,
                    return_tensors='pt',
                    truncation=True,
                    max_length=100,
                    padding='max_length'
                )
                output = model(**encoded_input)
                vector = output.last_hidden_state[:, 0, :]
                vector = vector.detach().numpy()
                vectors_1.append(vector.reshape(-1))
                self._progress_bar.update(1)
            self.vector_size = vectors_1[0].shape[0]
            self.vectors_1 = np.array(vectors_1).astype('float32')
            if not data.is_dirty_er:
                vectors_2 = []
                for i in range(0, data.num_of_entities_2, 1):
                    record = isolated_attr_dataset_2.iloc[i] if attributes_2 \
                                else data.entities_d2.iloc[i]
                    encoded_input = tokenizer(
                        record,
                        return_tensors='pt',
                        truncation=True,
                        max_length=100,
                        padding='max_length'
                    )
                    output = model(**encoded_input)
                    vector = output.last_hidden_state[:, 0, :]
                    vector = vector.detach().numpy()
                    vectors_2.append(vector.reshape(-1))
                    self._progress_bar.update(1)
                self.vectors_2 = np.array(vectors_2).astype('float32')
        elif self.vectorizer in ['smpnet', 'st5', 'glove', 'sdistilroberta', 'sminilm']:
            # ---------------------------- #
            # Pre-trained Sentence Embeddings
            # ---------------------------- #
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SentenceTransformer(self._sentence_transformer_mapping[self.vectorizer], device=device)
            for i in range(0, data.num_of_entities_1, 1):
                record = isolated_attr_dataset_1.iloc[i] if attributes_1 \
                            else data.entities_d1.iloc[i]
                vector = model.encode(record)
                vectors_1.append(vector)
                self._progress_bar.update(1)
            self.vector_size = len(vectors_1[0])
            self.vectors_1 = np.array(vectors_1).astype('float32')
            if not data.is_dirty_er:
                vectors_2 = []
                for i in range(0, data.num_of_entities_2, 1):
                    record = isolated_attr_dataset_2.iloc[i] if attributes_2 \
                                else data.entities_d2.iloc[i]
                    vector = model.encode(record)
                    vectors_2.append(vector)
                    self._progress_bar.update(1)
                self.vector_size = vectors_2[0].shape[1]
                self.vectors_2 = np.array(vectors_2).astype('float32')
        else:
            raise AttributeError("Not available vectorizer")

        if self.similarity_search == 'faiss':
            quantiser = faiss.IndexFlatL2(self.vectors_1.shape[1])
            index = faiss.IndexIVFFlat(
                quantiser,
                self.vectors_1.shape[1],
                self.num_of_clusters,
                faiss.METRIC_L2
            )
            index.train(self.vectors_1)  # train on the vectors of dataset 1
            index.add(self.vectors_1)   # add the vectors and update the index
            _, indices = index.search(
                self.vectors_1 if self.data.is_dirty_er else self.vectors_2, 
                self.top_k
            )
            if self.data.is_dirty_er:
                self.blocks = {
                    i : set(x for x in indices[i] if x not in [-1, i]) \
                            for i in range(0, indices.shape[0])
                }
            else:
                self.blocks = {
                    i+self.data.dataset_limit : set(x for x in indices[i] if x != -1) \
                            for i in range(0, indices.shape[0])
                }
        elif self.similarity_search == 'falconn':
            if not LINUX_ENV:
                raise ImportError("Can't use FALCONN in windows environment. Use FAISS instead.")
            # TODO FALCONN
        elif self.similarity_search == 'scann'  and LINUX_ENV:
            if not LINUX_ENV:
                raise ImportError("Can't use SCANN in windows environment. Use FAISS instead.")

            searcher = scann.scann_ops_pybind.builder(self.vectors_1, num_neighbors=self.top_k, distance_measure="dot_product") \
                            .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000) \
                            .score_ah(2, anisotropic_quantization_threshold=0.2) \
                            .reorder(100) \
                            .build()

            neighbors, distances = searcher.search_batched(
                self.vectors_1 if self.data.is_dirty_er else self.vectors_2,
                final_num_neighbors=self.top_k
            )
            print(neighbors.shape, distances.shape)
            if self.data.is_dirty_er:
                self.blocks = {
                    i : set(x for x in neighbors[i] if x not in [-1, i]) \
                            for i in range(0, neighbors.shape[0])
                }
            else:
                self.blocks = {
                    i+self.data.dataset_limit : set(x for x in neighbors[i] if x != -1) \
                            for i in range(0, neighbors.shape[0])
                }
        else:
            raise AttributeError("Not available method")
        self._progress_bar.close()
        self.execution_time = time() - _start_time
        return self.blocks

    def _create_vector(self, tokens: List[str], vocabulary) -> np.array:
        num_of_tokens = 0
        vector = np.zeros(self.vector_size)
        for token in tokens:
            if token in vocabulary:
                vector += vocabulary[token]
                num_of_tokens += 1
        if num_of_tokens > 0:
            vector /= num_of_tokens
        return vector

    def _configuration(self) -> dict:
        return {
            "Vectorizer" : self.vectorizer,
            "Similarity-Search" : self.similarity_search,
            "Top-K" : self.top_k,
            "Vector size": self.vector_size
        }
