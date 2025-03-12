"""Entity Matching Module
"""
import statistics
import pandas as pd
from time import time

import matplotlib.pyplot as plt
import numpy as np
from networkx import Graph
from py_stringmatching.similarity_measure.cosine import Cosine
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.similarity_measure.generalized_jaccard import \
    GeneralizedJaccard
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.jaro import Jaro
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.overlap_coefficient import \
    OverlapCoefficient
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.whitespace_tokenizer import \
    WhitespaceTokenizer
from tqdm.autonotebook import tqdm

from .datamodel import Data, PYJEDAIFeature
from .evaluation import Evaluation
from .utils import WordQgramTokenizer, cosine, get_qgram_from_tokenizer_name, FrequencyEvaluator


metrics_mapping = {
    'edit_distance': Levenshtein(),
    'cosine' : Cosine(),
    'jaro' : Jaro(),
    'jaccard' : Jaccard(),
    'generalized_jaccard' : GeneralizedJaccard(),
    'dice': Dice(),
    'overlap_coefficient' : OverlapCoefficient(),
}

vector_metrics_mapping = {
    'cosine': cosine
}

string_metrics = [
    'jaro', 'edit_distance'
]

set_metrics = [
    'cosine', 'dice', 'generalized_jaccard', 'jaccard', 'overlap_coefficient'
]

vector_metrics = [ 
    'cosine', 'dice', 'jaccard', 'sqeuclidean'
]

whoosh_index_metrics = [
    'TF-IDF', 'Frequency', 'PL2', 'BM25F'
] 

faiss_metrics = [
    'cosine', 'euclidean'
]

magellan_metrics = string_metrics + set_metrics
available_metrics = magellan_metrics + vector_metrics + whoosh_index_metrics + faiss_metrics

#
# Tokenizers
#
# char_qgram_tokenizers = { 'char_'+ str(i) + 'gram':i for i in range(1, 7) }
# word_qgram_tokenizers = { 'word_'+ str(i) + 'gram':i for i in range(1, 7) }
char_qgram_tokenizers = ['char_tokenizer']
word_qgram_tokenizers = ['word_tokenizer']
magellan_tokenizers = ['white_space_tokenizer']
joins_tokenizers = ["qgrams", "standard", "standard_multiset", "qgrams_multiset"]

# tfidf_tokenizers = [ 'tfidf_' + cq for cq in char_qgram_tokenizers.keys() ] + \
#                     [ 'tfidf_' + wq for wq in word_qgram_tokenizers.keys() ]

# tf_tokenizers = [ 'tf_' + cq for cq in char_qgram_tokenizers.keys() ] + \
#                     [ 'tf_' + wq for wq in word_qgram_tokenizers.keys() ]
                        
# boolean_tokenizers = [ 'boolean_' + cq for cq in char_qgram_tokenizers.keys() ] + \
#                         [ 'boolean_' + wq for wq in word_qgram_tokenizers.keys() ]

# vector_tokenizers = tfidf_tokenizers + tf_tokenizers + boolean_tokenizers

# available_tokenizers = [key for key in char_qgram_tokenizers] + [key for key in word_qgram_tokenizers] + magellan_tokenizers + vector_tokenizers
available_tokenizers = char_qgram_tokenizers + word_qgram_tokenizers + magellan_tokenizers + joins_tokenizers
available_vectorizers = ['tfidf', 'tf', 'boolean']

class AbstractEntityMatching(PYJEDAIFeature):
    """Calculates similarity from 0.0 to 1.0
    """

    _method_name: str = "Abstract Entity Matching"
    _method_info: str = "Calculates similarity from 0. to 1."

    def __init__(
            self,
            metric: str = 'dice',
            similarity_threshold: float = 0.5,
        ) -> None:
        self.pairs: Graph
        self.metric = metric
        self.similarity_threshold = similarity_threshold
        self.execution_time = 0
        self.qgram=None
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

    def _predict_candidate_pairs(self, blocks: dict) -> None:
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

    def get_weights_avg(self) -> float:
        return sum([w for _, _, w in self.pairs.edges(data='weight')])/len(self.pairs.edges(data='weight'))

    def get_weights_median(self) -> float:
        return [w for _, _, w in sorted(self.pairs.edges(data='weight'))][int(len(self.pairs.edges(data='weight'))/2)]    
    
    def get_weights_standard_deviation(self) -> float:
        return statistics.stdev([w for _, _, w in self.pairs.edges(data='weight')])
    
    def plot_distribution_of_all_weights(self, save_figure_path=None) -> None:
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
        if save_figure_path:
            plt.savefig(save_figure_path)
        plt.show()

    def plot_distribution_of_all_weights_2d(self, save_figure_path=None) -> None:
        title = "Distribution of scores with " + self.metric + " metric in graph from entity matching"
        plt.figure(figsize=(10, 6))
        all_weights = [w for _, _, w in self.pairs.edges(data='weight')]
        sorted_weights = sorted(all_weights, reverse=True)
        
        fig, ax = plt.subplots(tight_layout=True)
        hist = ax.hist2d(sorted_weights, sorted_weights)
        plt.axvline(x = self.get_weights_avg(), color = 'blue', label = 'Average weight')
        plt.axvline(x = self.get_weights_median(), color = 'black', label = 'Median weight')
        plt.axvline(x = self.get_weights_avg()+self.get_weights_standard_deviation(), color = 'green', label = 'Average + SD weight')
        plt.legend()
        if save_figure_path:
            plt.savefig(save_figure_path)
        plt.show()

    def plot_distribution_of_scores(self, save_figure_path=None) -> None:
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
        if save_figure_path:
            plt.savefig(save_figure_path)
        plt.show()

    def plot_gt_distribution_of_scores(self, save_figure_path=None) -> None:
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
        if save_figure_path:
            plt.savefig(save_figure_path)
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
        
    def stats(self) -> None:
        pass
        
    def export_to_df(self, prediction: Graph, tqdm_enable=False) -> pd.DataFrame:
        """Creates a dataframe with the predicted pairs.

        Args:
            prediction (Graph): Predicted graph
            tqdm_enable (bool): Whether to enable tqdm progress bar

        Returns:
            pd.DataFrame: Dataframe with the predicted pairs
        """
        pairs_list = []

        is_dirty_er = self.data.is_dirty_er
        dataset_limit = self.data.dataset_limit
        gt_to_ids_reversed_1 = self.data._gt_to_ids_reversed_1
        if not is_dirty_er:
            gt_to_ids_reversed_2 = self.data._gt_to_ids_reversed_2

        for edge in tqdm(prediction.edges, disable=not tqdm_enable, desc="Exporting to DataFrame"):
            node1, node2 = edge

            if not is_dirty_er:
                if node1 < dataset_limit:
                    id1 = gt_to_ids_reversed_1[node1]
                    id2 = gt_to_ids_reversed_2[node2]
                else:
                    id1 = gt_to_ids_reversed_1[node2]
                    id2 = gt_to_ids_reversed_2[node1]
            else:
                id1 = gt_to_ids_reversed_1[node1]
                id2 = gt_to_ids_reversed_1[node2]

            pairs_list.append((id1, id2))

        pairs_df = pd.DataFrame(pairs_list, columns=['id1', 'id2'])

        return pairs_df

class EntityMatching(AbstractEntityMatching):
    """Calculates similarity from 0.0 to 1.0 for all blocks
    """

    _method_name: str = "Entity Matching"
    _method_info: str = "Calculates similarity from 0. to 1. for all blocks"

    def __init__(
            self,
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            vectorizer : str = None,
            qgram : int = 1,
            similarity_threshold: float = 0.0,
            tokenizer_return_unique_values = False, # unique values or not,
            attributes: any = None,
        ) -> None:
        super().__init__()
        self.pairs: Graph
        self.metric = metric
        self.attributes: list = attributes
        self.similarity_threshold = similarity_threshold
        self.tokenizer = tokenizer
        self.execution_time = 0
        self.vectorizer = vectorizer
        self.qgram: int = -1
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

        self.tokenizer_return_set = (metric in set_metrics) or tokenizer_return_unique_values    
        self.qgram : int = qgram
        
        if(vectorizer is not None):
            if self.vectorizer not in available_vectorizers:
                raise AttributeError(
                    'Weighting Scheme ({}) does not exist. Please select one of the available. ({})'.format(
                        vectorizer, available_vectorizers
                    )
                )
        elif(tokenizer is not None):
            if tokenizer == 'white_space_tokenizer':
                self._tokenizer = WhitespaceTokenizer(return_set=self.tokenizer_return_set)
            elif tokenizer == 'char_tokenizer':
                self._tokenizer = QgramTokenizer(qval=self.qgram,
                                                return_set=self.tokenizer_return_set)
            elif tokenizer == 'word_tokenizer':
                self._tokenizer = WordQgramTokenizer(q=self.qgram)
            elif tokenizer not in available_tokenizers:
                raise AttributeError(
                    'Tokenizer ({}) does not exist. Please select one of the available. ({})'.format(
                        tokenizer, available_tokenizers
                    )
                )
        
    def predict(self,
                blocks: dict,
                data: Data,
                tqdm_disable: bool = False) -> Graph:
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
        
        if not blocks:
            raise ValueError("Empty blocks structure")
        self.data = data
        self.pairs = Graph()
        all_blocks = list(blocks.values())
        self._progress_bar = tqdm(total=len(blocks),
                                  desc=self._method_name+" ("+self.metric+ ", " + str(self.tokenizer) + ")",
                                  disable=self.tqdm_disable)

        if self.vectorizer is not None:
            self.initialize_vectorizer()

        if 'Block' in str(type(all_blocks[0])):
            self._predict_raw_blocks(blocks)
        elif isinstance(all_blocks[0], set):
            self._predict_candidate_pairs(blocks)
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

    def initialize_vectorizer(self) -> None:
        self.frequency_evaluator : FrequencyEvaluator = FrequencyEvaluator(vectorizer=self.vectorizer,
                                                                            tokenizer=self.tokenizer,
                                                                            qgram=self.qgram)
        d1 = self.data.dataset_1[self.attributes] if self.attributes else self.data.dataset_1
        self._entities_d1 = d1 \
                    .apply(" ".join, axis=1) \
                    .apply(lambda x: x.lower()) \
                    .values.tolist()
        
        d2 = None
        if(not self.data.is_dirty_er):
            d2 = self.data.dataset_2
            if self.attributes:
                d2 = d2[self.attributes]

        self._entities_d2 = d2 \
                    .apply(" ".join, axis=1) \
                    .apply(lambda x: x.lower()) \
                    .values.tolist() if not self.data.is_dirty_er else self._entities_d1 
        
        
        _dataset_identifier : str = ('_'.join([self.data.dataset_name_1, self.data.dataset_name_2])) if(self.data.dataset_name_1 is not None and self.data.dataset_name_2 is not None) else ("dataset") 
        self.frequency_evaluator.fit(metric=self.metric,
                                    dataset_identifier=_dataset_identifier,
                                    indexing='inorder',
                                    d1_entities=self._entities_d1,
                                    d2_entities=self._entities_d2)

    def _similarity(self, entity_id1: int, entity_id2: int) -> float:

        similarity: float = 0.0
        if self.vectorizer is not None:
            return self.frequency_evaluator.predict(id1=entity_id1, id2=entity_id2)
        elif isinstance(self.attributes, dict):
            for attribute, weight in self.attributes.items():
                e1 = self.data.entities.iloc[entity_id1][attribute].lower()
                e2 = self.data.entities.iloc[entity_id2][attribute].lower()

                similarity += weight*metrics_mapping[self._metric].get_sim_score(
                    self._tokenizer.tokenize(e1) if self._metric in set_metrics else e1,
                    self._tokenizer.tokenize(e2) if self._metric in set_metrics else e2
                )
        elif isinstance(self.attributes, list):
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

    def _configuration(self) -> dict:
        return  {
            "Metric" : self.metric,
            "Attributes" : self.attributes,
            "Similarity threshold" : self.similarity_threshold,
            "Tokenizer" : self.tokenizer,
            "Vectorizer" : self.vectorizer if self.vectorizer is not None else "None",
            "Qgrams" : self.qgram
        }

class VectorBasedMatching(AbstractEntityMatching):

    _method_name: str = "Vector Based Matching"
    _method_info: str = "Calculates similarity from 0. to 1. for vectors"

    def __init__(
            self,
            metric: str = 'cosine',
            similarity_threshold: float = 0.5,
        ) -> None:
        self.pairs: Graph
        self.metric = metric
        self.similarity_threshold = similarity_threshold
        self.vectors_d1 = None
        self.vectors_d2 = None
        self.execution_time = 0

        #
        # Selecting tokenizer
        #
        if metric not in vector_metrics:
            raise AttributeError(
                'Metric ({}) does not exist. Please select one of the available. ({})'.format(
                    metric, available_metrics
                )
            )
        else:
            self._metric = metric
            
    def predict(self,
                blocks: dict,
                data: Data,
                vectors_d1: np.array,
                vectors_d2: np.array = None,
                tqdm_disable: bool = False,
        ) -> Graph:
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
        # self.vectors_d2 = vectors_d2
        

        if(vectors_d1 is None):
            raise ValueError("Embeddings of the first dataset not given")
        else:
            self.vectors = vectors_d1
            if(not data.is_dirty_er):
                if(vectors_d2 is None):
                    raise ValueError("Embeddings of the second dataset not given")
                self.vectors = np.concatenate((vectors_d1,vectors_d2), axis=0)
        self.data = data
        self.pairs = Graph()
        self._progress_bar = tqdm(total=len(blocks),
                                  desc=self._method_name+" ("+self.metric+ ", " + str(self.tokenizer) + ")",
                                  disable=self.tqdm_disable)
        self._predict_candidate_pairs(blocks)
        self.execution_time = time() - start_time
        self._progress_bar.close()

        return self.pairs

    def _similarity(self, entity_id1: int, entity_id2: int) -> float:
        return vector_metrics_mapping[self._metric](self.vectors[entity_id1], self.vectors[entity_id2])

    def _configuration(self) -> dict:
        conf =  {
            "Metric" : self.metric,
            "Similarity threshold" : self.similarity_threshold
        }
        
        return conf
