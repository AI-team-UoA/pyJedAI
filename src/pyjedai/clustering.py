from queue import PriorityQueue
from time import time

from networkx import Graph, connected_components
from tqdm.autonotebook import tqdm

from .datamodel import Data, PYJEDAIFeature
from .evaluation import Evaluation
from .utils import are_matching


class AbstractClustering(PYJEDAIFeature):
    
    def __init__(self) -> None:
        super().__init__()
        self.data: Data
        self.similarity_threshold: float = 0.1
        
    def evaluate(self,
                 prediction,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:

        if prediction is None:
            if self.blocks is None:
                raise AttributeError("Can not proceed to evaluation without build_blocks.")
            else:
                eval_blocks = self.blocks
        else:
            eval_blocks = prediction
            
        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " + 
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        true_positives = 0
        entity_index = eval_obj._create_entity_index_from_clusters(eval_blocks)
        for _, (id1, id2) in self.data.ground_truth.iterrows():
            id1 = self.data._ids_mapping_1[id1]
            id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er else self.data._ids_mapping_2[id2]
            if id1 in entity_index and    \
                id2 in entity_index and are_matching(entity_index, id1, id2):
                true_positives += 1
        # print(entity_index)
        eval_obj.calculate_scores(true_positives=true_positives)
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)
    
    def stats(self) -> None:
        pass

class ConnectedComponentsClustering(AbstractClustering):
    """Creates the connected components of the graph. \
        Applied to graph created from entity matching. \
        Input graph consists of the entity ids (nodes) and the similarity scores (edges).
    """

    _method_name: str = "Connected Components Clustering"
    _method_short_name: str = "CCC"
    _method_info: str = "Gets equivalence clusters from the " + \
                    "transitive closure of the similarity graph."

    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold: float

    def process(self, graph: Graph, data: Data, similarity_threshold: float = None) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """
        start_time = time()
        self.data = data
        self.similarity_threshold: float = similarity_threshold
        graph_copy = graph.copy()
        if self.similarity_threshold is not None:
            for x in graph.edges(data=True):
                if x[2]['weight'] < self.similarity_threshold:
                    graph_copy.remove_edge(x[0], x[1])
        clusters = list(connected_components(graph_copy))
        # print(clusters)
        # print("Number of clusters: ", len(clusters))
        resulting_clusters = list(filter(lambda x: len(x) == 2, clusters)) \
                                if not data.is_dirty_er else clusters
        # print("Number of clusters after filtering: ", len(resulting_clusters))
        self.execution_time = time() - start_time
        return resulting_clusters

    def _configuration(self) -> dict:
        return {}

class UniqueMappingClustering(AbstractClustering):
    """Prunes all edges with a weight lower than t, sorts the remaining ones in
        decreasing weight/similarity and iteratively forms a partition for
        the top-weighted pair as long as none of its entities has already
        been matched to some other.
    """

    _method_name: str = "Unique Mapping Clustering"
    _method_short_name: str = "UMC"
    _method_info: str = "Prunes all edges with a weight lower than t, sorts the remaining ones in" + \
                        "decreasing weight/similarity and iteratively forms a partition for" + \
                        "the top-weighted pair as long as none of its entities has already" + \
                        "been matched to some other."

    def __init__(self) -> None:
        """Unique Mapping Clustering Constructor

        Args:
            similarity_threshold (float, optional): Prunes all edges with a weight
                lower than this. Defaults to 0.1.
            data (Data): Dataset module.
        """
        super().__init__()
        self.similarity_threshold: float

    def process(self, graph: Graph, data: Data, similarity_threshold: float = 0.1) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """
        if data.is_dirty_er:
            raise AttributeError("Unique Mapping Clustering can only be performed in Clean-Clean Entity Resolution.")
        self.similarity_threshold: float = similarity_threshold
        
        start_time = time()
        matched_entities = set()
        self.data = data
        new_graph = Graph()
        priority_queue = PriorityQueue(maxsize = graph.number_of_edges()*2)
        for x in graph.edges(data=True):
            if x[2]['weight'] > self.similarity_threshold:
                priority_queue.put_nowait((1- x[2]['weight'], x[0], x[1]))

        while not priority_queue.empty():
            sim, entity_1, entity_2 = priority_queue.get()
            if entity_1 in matched_entities or entity_2 in matched_entities:
                continue
            new_graph.add_edge(entity_1, entity_2, weight=sim)
            matched_entities.add(entity_1)
            matched_entities.add(entity_2)

        clusters = ConnectedComponentsClustering().process(new_graph, data, similarity_threshold=None)
        self.execution_time = time() - start_time
        return clusters

    def _configuration(self) -> dict:
        return {}

class ExactClustering(AbstractClustering):
    """Implements an adapted, simplified version of the Exact THRESHOLD algorithm,
        introduced in "Similarity Flooding: A Versatile Graph Matching Algorithm and Its Application to Schema Matching",
        also referred in "BIGMAT: A Distributed Affinity-Preserving Random Walk Strategy for Instance Matching on Knowledge Graphs".
        In essence, it keeps the top-1 candidate per entity, as long as the candidate also considers this node as its top candidate.
    """

    _method_name: str = "Exact Clustering"
    _method_short_name: str = "EC"
    _method_info: str = "Ιmplements an adapted, simplified version of the Exact THRESHOLD algorithm," + \
        "In essence, it keeps the top-1 candidate per entity, as long as the candidate also considers this node as its top candidate."

    def __init__(self, similarity_threshold: float = 0.1) -> None:
        """"""
        super().__init__()
        self.similarity_threshold = similarity_threshold

    def process(self, graph: Graph, data: Data) -> list:
        """
        """
        raise NotImplementedError("Exact Clustering is not implemented yet.")

    def _configuration(self) -> dict:
        return {}
