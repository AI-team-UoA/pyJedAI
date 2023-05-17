"""Entity Matching Module
"""
import statistics
from time import time

import matplotlib.pyplot as plt
import numpy as np
from networkx import Graph
from py_stringmatching.similarity_measure.affine import Affine
from py_stringmatching.similarity_measure.bag_distance import BagDistance
from py_stringmatching.similarity_measure.cosine import Cosine
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.similarity_measure.editex import Editex
from py_stringmatching.similarity_measure.generalized_jaccard import \
    GeneralizedJaccard
from py_stringmatching.similarity_measure.hamming_distance import \
    HammingDistance
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.jaro import Jaro
from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
from py_stringmatching.similarity_measure.needleman_wunsch import \
    NeedlemanWunsch
from py_stringmatching.similarity_measure.overlap_coefficient import \
    OverlapCoefficient
from py_stringmatching.similarity_measure.partial_ratio import PartialRatio
from py_stringmatching.similarity_measure.partial_token_sort import \
    PartialTokenSort
from py_stringmatching.similarity_measure.ratio import Ratio
from py_stringmatching.similarity_measure.smith_waterman import SmithWaterman
from py_stringmatching.similarity_measure.soundex import Soundex
from py_stringmatching.similarity_measure.tfidf import TfIdf
from py_stringmatching.similarity_measure.token_sort import TokenSort
from py_stringmatching.similarity_measure.tversky_index import TverskyIndex
from py_stringmatching.tokenizer.alphabetic_tokenizer import \
    AlphabeticTokenizer
from py_stringmatching.tokenizer.alphanumeric_tokenizer import \
    AlphanumericTokenizer
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.whitespace_tokenizer import \
    WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.autonotebook import tqdm

from .datamodel import Data, PYJEDAIFeature
from .evaluation import Evaluation
from .utils import WordQgrammsTokenizer

# Package import from https://anhaidgroup.github.io/py_stringmatching/v0.4.2/index.html

def cosine(x, y):
    """Cosine similarity between two vectors
    """
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]

available_tokenizers = [
    'white_space_tokenizer', 'qgram_tokenizer', 'delimiter_tokenizer',
    'alphabetic_tokenizer', 'alphanumeric_tokenizer'
]

metrics_mapping = {
    'levenshtein' : Levenshtein(),
    'edit_distance': Levenshtein(),
    'jaro_winkler' : JaroWinkler(),
    'bag_distance' : BagDistance(),
    'editex' : Editex(),
    'cosine' : Cosine(),
    'jaro' : Jaro(),
    'soundex' : Soundex(),
    'tfidf' : TfIdf(),
    'tversky_index':TverskyIndex(),
    'ratio' : Ratio(),
    'partial_token_sort' : PartialTokenSort(),
    'partial_ratio' : PartialRatio(),
    'hamming_distance' : HammingDistance(),
    'jaccard' : Jaccard(),
    'generalized_jaccard' : GeneralizedJaccard(),
    'dice': Dice(),
    'overlap_coefficient' : OverlapCoefficient(),
    'token_sort': TokenSort(),
    'cosine_vector_similarity': cosine
}

string_metrics = [
    'bag_distance', 'editex', 'hamming_distance', 'jaro', 'jaro_winkler', 'levenshtein',
    'edit_distance', 'partial_ratio', 'partial_token_sort', 'ratio', 'soundex', 'token_sort'
]

set_metrics = [
    'cosine', 'dice', 'generalized_jaccard', 'jaccard', 'overlap_coefficient', 'tversky_index'
]

bag_metrics = [
    'tf-idf'
]

vector_metrics = [
    'cosine_vector_similarity'
]

available_metrics = string_metrics + set_metrics + bag_metrics + vector_metrics


class EntityMatching(PYJEDAIFeature):
    """Calculates similarity from 0.0 to 1.0 for all blocks
    """

    _method_name: str = "Entity Matching"
    _method_info: str = "Calculates similarity from 0. to 1. for all blocks"

    def __init__(
            self,
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = False, # unique values or not,
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:
        self.pairs: Graph
        self.metric = metric
        self.qgram: int = qgram
        self.attributes: list = attributes
        self.similarity_threshold = similarity_threshold
        self.vectors_d1 = None
        self.vectors_d2 = None
        self.tokenizer = tokenizer
        self.execution_time = 0
        #
        # Selecting tokenizer
        #
        if metric not in available_metrics:
            raise AttributeError(
                'Metric ({}) does not exist. Please select one of the available. ({})'.format(
                    metric, available_metrics
                )
            )
        else:
            self._metric = metric

        if metric in set_metrics:
            self.tokenizer_return_set = True
        else:
            self.tokenizer_return_set = tokenizer_return_set

        if tokenizer == 'white_space_tokenizer':
            self._tokenizer = WhitespaceTokenizer(return_set=self.tokenizer_return_set)
        elif tokenizer == 'char_qgram_tokenizer':
            self._tokenizer = QgramTokenizer(qval=self.qgram,
                                             return_set=self.tokenizer_return_set,
                                             padding=padding,
                                             suffix_pad=suffix_pad,
                                             prefix_pad=prefix_pad)
        elif tokenizer == 'word_qgram_tokenizer':
            self._tokenizer = WhitespaceTokenizer(return_set=self.tokenizer_return_set)
        elif tokenizer == 'delimiter_tokenizer':
            self._tokenizer = DelimiterTokenizer(return_set=self.tokenizer_return_set,
                                                 delim_set=delim_set)
        elif tokenizer == 'alphabetic_tokenizer':
            self._tokenizer = AlphabeticTokenizer(return_set=self.tokenizer_return_set)
        elif tokenizer == 'alphanumeric_tokenizer':
            self._tokenizer = AlphanumericTokenizer(return_set=self.tokenizer_return_set)
        else:
            raise AttributeError(
                'Tokenizer ({}) does not exist. Please select one of the available. ({})'.format(
                    tokenizer, available_tokenizers
                )
            )
        
    def predict(self,
                blocks: dict,
                data: Data,
                tqdm_disable: bool = False,
                vectors_d1: np.array = None,
                vectors_d2: np.array = None) -> Graph:
        """Main method of entity matching. Inputs a set of blocks and outputs a graph \
            that contains of the entity ids (nodes) and the similarity scores between them (edges).

            Args:
                blocks (dict): blocks of entities
                data (Data): dataset module
                tqdm_disable (bool, optional): Disables progress bar. Defaults to False.

            Returns:
                networkx.Graph: entity ids (nodes) and similarity scores between them (edges)
        """
        start_time = time()
        self.tqdm_disable = tqdm_disable
        self.vectors_d1 = vectors_d1
        self.vectors_d2 = vectors_d2
        
        if self.metric in vector_metrics:
            if(vectors_d2 is not None and vectors_d1 is None):
                raise ValueError("Embeddings of the first dataset not given")

            if(vectors_d1 is not None):
                self.vectors = vectors_d1
                if(not data.is_dirty_er):
                    if(vectors_d2 is None):
                        raise ValueError("Embeddings of the second dataset not given")
                    self.vectors = np.concatenate((vectors_d1,vectors_d2), axis=0)

        if not blocks:
            raise ValueError("Empty blocks structure")
        self.data = data
        self.pairs = Graph()
        all_blocks = list(blocks.values())
        self._progress_bar = tqdm(total=len(blocks),
                                  desc=self._method_name+" ("+self.metric+")",
                                  disable=self.tqdm_disable)
        
        if self.metric == 'tf-idf':
            self._calculate_tfidf()
        
        if 'Block' in str(type(all_blocks[0])):
            self._predict_raw_blocks(blocks)
        elif isinstance(all_blocks[0], set):
            self._predict_prunned_blocks(blocks)
        else:
            raise AttributeError("Wrong type of Blocks")
        self.execution_time = time() - start_time
        self._progress_bar.close()

        return self.pairs

    def _predict_raw_blocks(self, blocks: dict) -> None:
        """Method for similarity evaluation blocks after Block building

        Args:
            blocks (dict): Block building blocks
        """
        if self.data.is_dirty_er:
            for _, block in blocks.items():
                entities_array = list(block.entities_D1)
                for index_1 in range(0, len(entities_array), 1):
                    for index_2 in range(index_1+1, len(entities_array), 1):
                        similarity = self._similarity(entities_array[index_1],
                                                      entities_array[index_2])
                        self._insert_to_graph(entities_array[index_1],
                                              entities_array[index_2],
                                              similarity)
                self._progress_bar.update(1)
        else:
            for _, block in blocks.items():
                for entity_id1 in block.entities_D1:
                    for entity_id2 in block.entities_D2:
                        similarity = self._similarity(entity_id1, entity_id2)
                        self._insert_to_graph(entity_id1, entity_id2, similarity)
                self._progress_bar.update(1)

    def _predict_prunned_blocks(self, blocks: dict) -> None:
        """Similarity evaluation after comparison cleaning.

        Args:
            blocks (dict): Comparison cleaning blocks.
        """
        for entity_id, candidates in blocks.items():
            for candidate_id in candidates:
                similarity = self._similarity(entity_id, candidate_id)
                self._insert_to_graph(entity_id, candidate_id, similarity)
            self._progress_bar.update(1)

    def _insert_to_graph(self, entity_id1, entity_id2, similarity):
        if self.similarity_threshold is None or \
            (self.similarity_threshold is not None and similarity > self.similarity_threshold):
            self.pairs.add_edge(entity_id1, entity_id2, weight=similarity)

    def _calculate_vector_similarity(self, entity_id1: int, entity_id2: int) -> float:
        if self.metric in vector_metrics:
            return metrics_mapping[self._metric](self.vectors[entity_id1],
                                                 self.vectors[entity_id2])
        else:
            raise AttributeError("Please select one vector similarity metric from the given: " + ','.join(vector_metrics))

    def _calculate_tfidf(self) -> None:
        
        analyzer = 'char' if self.tokenizer == 'char_qgram_tokenizer' else 'word'
        vectorizer = TfidfVectorizer(analyzer='') if self.qgram is None else TfidfVectorizer(analyzer=analyzer, ngram_range=(self.qgram, self.qgram))
        
        d1 = self.data.dataset_1[self.attributes] if self.attributes else self.data.dataset_1
        self._entities_d1 = d1 \
                    .apply(" ".join, axis=1) \
                    .apply(lambda x: x.lower()) \
                    .values.tolist()
        
        d2 = self.data.dataset_2[self.attributes] if self.attributes and not self.data.is_dirty_er else self.data.dataset_2
        self._entities_d2 = d2 \
                    .apply(" ".join, axis=1) \
                    .apply(lambda x: x.lower()) \
                    .values.tolist() if not self.data.is_dirty_er else None
                    
        if self.data.is_dirty_er:
            pass
        else:
            self.corpus = self._entities_d1 + self._entities_d2
            self.tfidf_vectorizer = vectorizer.fit(self.corpus)
            self.tfidf_matrix = vectorizer.transform(self.corpus)
            self.tfidf_similarity_matrix = cosine_similarity(self.tfidf_matrix)
            # feature_names = self.tfidf_vectorizer.get_feature_names()
            # tfidf_df = pd.DataFrame(self.tfidf_matrix.toarray(), columns=feature_names)
            # self.tfidf_df = tfidf_df

    def _calculate_tfidf_similarity(self, entity_id1: int, entity_id2: int) -> float:
        return self.tfidf_similarity_matrix[entity_id1][entity_id2]

    def _similarity(self, entity_id1: int, entity_id2: int) -> float:

        similarity: float = 0.0
        if self.vectors_d1 is not None and self.metric in vector_metrics:
            return self._calculate_vector_similarity(entity_id1, entity_id2)
        elif self.metric == 'tf-idf':
            return self._calculate_tfidf_similarity(entity_id1, entity_id2)

        if isinstance(self.attributes, dict):
            for attribute, weight in self.attributes.items():
                e1 = self.data.entities.iloc[entity_id1][attribute].lower()
                e2 = self.data.entities.iloc[entity_id2][attribute].lower()

                similarity += weight*metrics_mapping[self._metric].get_sim_score(
                    self._tokenizer.tokenize(e1) if self._metric in set_metrics else e1,
                    self._tokenizer.tokenize(e2) if self._metric in set_metrics else e2
                )
        if isinstance(self.attributes, list):
            for attribute in self.attributes:
                e1 = self.data.entities.iloc[entity_id1][attribute].lower()
                e2 = self.data.entities.iloc[entity_id2][attribute].lower()
                similarity += metrics_mapping[self._metric].get_sim_score(
                    self._tokenizer.tokenize(e1) if self._metric in set_metrics else e1,
                    self._tokenizer.tokenize(e2) if self._metric in set_metrics else e2
                )
                similarity /= len(self.attributes)
        else:
            # concatenated row string
            e1 = self.data.entities.iloc[entity_id1].str.cat(sep=' ').lower()
            e2 = self.data.entities.iloc[entity_id2].str.cat(sep=' ').lower()
            te1 = self._tokenizer.tokenize(e1) if self._metric in set_metrics else e1
            te2 = self._tokenizer.tokenize(e2) if self._metric in set_metrics else e2
            similarity = metrics_mapping[self._metric].get_sim_score(te1, te2)

        return similarity        

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes:\n\t" + ', '.join(c for c in (self.attributes if self.attributes is not None \
                else self.data.dataset_1.columns)) +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

    def _configuration(self) -> dict:
        return {
            "Metric" : self.metric,
            "Attributes" : self.attributes,
            "Similarity threshold" : self.similarity_threshold
        }

    def get_weights_avg(self) -> float:
        return sum([w for _, _, w in self.pairs.edges(data='weight')])/len(self.pairs.edges(data='weight'))

    def get_weights_median(self) -> float:
        return [w for _, _, w in sorted(self.pairs.edges(data='weight'))][int(len(self.pairs.edges(data='weight'))/2)]    
    
    def get_weights_standard_deviation(self) -> float:
        return statistics.stdev([w for _, _, w in self.pairs.edges(data='weight')])
    
    def plot_distribution_of_all_weights(self) -> None:
        title = "Distribution of scores with " + self.metric + " metric in graph from entity matching"
        plt.figure(figsize=(10, 6))
        all_weights = [w for _, _, w in self.pairs.edges(data='weight')]
        sorted_weights = sorted(all_weights, reverse=True)
        
        plt.hist(sorted_weights)
        plt.xlim(0, 1)
        # only one line may be specified; full height
        plt.axvline(x = self.get_weights_avg(), color = 'blue', label = 'Average weight')
        plt.axvline(x = self.get_weights_median(), color = 'black', label = 'Median weight')
        plt.axvline(x = self.get_weights_avg()+self.get_weights_standard_deviation(), color = 'green', label = 'Average + SD weight')
        plt.legend()
        plt.show()

    
    def plot_distribution_of_all_weights_2d(self) -> None:
        title = "Distribution of scores with " + self.metric + " metric in graph from entity matching"
        plt.figure(figsize=(10, 6))
        all_weights = [w for _, _, w in self.pairs.edges(data='weight')]
        sorted_weights = sorted(all_weights, reverse=True)
        
        fig, ax = plt.subplots(tight_layout=True)
        hist = ax.hist2d(sorted_weights, sorted_weights)
        # plt.hist(sorted_weights)
        # plt.xlim(0, 1)
        # only one line may be specified; full height
        plt.axvline(x = self.get_weights_avg(), color = 'blue', label = 'Average weight')
        plt.axvline(x = self.get_weights_median(), color = 'black', label = 'Median weight')
        plt.axvline(x = self.get_weights_avg()+self.get_weights_standard_deviation(), color = 'green', label = 'Average + SD weight')
        plt.legend()
        plt.show()

    def plot_distribution_of_scores(self) -> None:
        title = "Distribution of scores with " + self.metric + " metric in graph from entity matching"
        def weight_distribution(G):
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            distribution = [0] * (len(bins)-1)
            for u, v, w in G.edges(data='weight'):
                for i in range(len(bins) - 1):
                    if bins[i] <= w < bins[i + 1]:
                        distribution[i] += 1
                        break
            return distribution, len(G.edges(data='weight'))

        labels = [f'{(i)/10:.1f} - {(i+1)/10:.1f}' for i in range(0, 10)]

        distribution, num_of_pairs = weight_distribution(self.pairs)
        width = 0.5
        x = np.arange(len(labels))  # the label locations
        distribution = list(map(lambda x: (x/num_of_pairs)*100, distribution))
        print("Distribution-% of predicted scores: ", distribution)

        fig, ax = plt.subplots(figsize=(10,6))
        r1 = ax.bar(x, distribution, width, align='center', color='red')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Percentage of pairs in each range to all (%)')
        ax.set_title(title)
        ax.set_xlabel('Similarity score range')
        fig.tight_layout()
        
        # only one line may be specified; full height
        plt.axvline(x = self.get_weights_avg()*10, color = 'blue', label = 'Average weight')
        plt.axvline(x = self.get_weights_median()*10, color = 'black', label = 'Median weight')
        plt.axvline(x = self.get_weights_avg()*10+self.get_weights_standard_deviation()*10, color = 'green', label = 'Average + SD weight')
        plt.legend()
        plt.show()

    def plot_gt_distribution_of_scores(self) -> None:
        title = "Distribution of scores with " + self.metric + " metric on ground truth pairs"
        def weight_distribution():
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            distribution = [0] * (len(bins)-1)
            for _, (id1, id2) in self.data.ground_truth.iterrows():
                id1 = self.data._ids_mapping_1[id1]
                id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er else self.data._ids_mapping_2[id2]
                w = self._calculate_vector_similarity(id1, id2)

                for i in range(len(bins) - 1):
                    if bins[i] <= w < bins[i + 1]:
                        distribution[i] += 1
                        break
            return distribution, len(self.data.ground_truth)

        labels = [f'{(i)/10:.1f} - {(i+1)/10:.1f}' for i in range(0, 10)]

        distribution, num_of_pairs = weight_distribution()
        width = 0.5
        x = np.arange(len(labels))  # the label locations
        distribution = list(map(lambda x: (x/num_of_pairs)*100, distribution))
        print("Distribution-% of predicted scores: ", distribution)

        fig, ax = plt.subplots(figsize=(10,6))
        r1 = ax.bar(x, distribution, width, align='center', color='blue')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Percentage of pairs in each range to all (%)')
        ax.set_title(title)
        ax.set_xlabel('Similarity score range')
        fig.tight_layout()
        plt.show()


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
        total_matching_pairs = prediction.number_of_edges()
        for _, (id1, id2) in self.data.ground_truth.iterrows():
            id1 = self.data._ids_mapping_1[id1]
            id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er \
                                                else self.data._ids_mapping_2[id2]
            if (id1 in prediction and id2 in prediction[id1]) or   \
                 (id2 in prediction and id1 in prediction[id2]):
                true_positives += 1

        eval_obj.calculate_scores(true_positives=true_positives,
                                  total_matching_pairs=total_matching_pairs)
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)

    def _configuration(self) -> dict:
        return {
            "Tokenizer" : self.tokenizer,
            "Metric" : self.metric,
            "Similarity Threshold" : self.similarity_threshold
        }
        
    def stats(self) -> None:
        pass