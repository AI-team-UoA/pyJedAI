from queue import PriorityQueue
from time import time

import pandas as pd
from networkx import Graph, connected_components, gomory_hu_tree
from tqdm.autonotebook import tqdm
from ordered_set import OrderedSet
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .datamodel import Data, PYJEDAIFeature
from .evaluation import Evaluation
from .utils import are_matching
from collections import defaultdict
import random
from ordered_set import OrderedSet
import math

RANDOM_SEED = 42


class EquivalenceCluster(PYJEDAIFeature):
    
    def __init__(self, data : Data) -> None:
        super().__init__()
        self.data : Data = data
        self.d1_entities = OrderedSet()
        self.d2_entities = OrderedSet()
        
    def __init__(self, data : Data, flattened_cluster : list) -> None:
        super().__init__()
        self.data : Data = data
        self.d1_entities = set()
        self.d2_entities = set()
        self.add_entities(flattened_cluster)
    
    def get_entity_dataset(self, entity : int) -> set:
        return self.d1_entities \
                if(entity < self.data.dataset_limit) \
                else self.d2_entities
    
    def add_entity(self, entity : int) -> None:
        target_dataset_entities = self.get_entity_dataset(entity) 
        target_dataset_entities.add(entity)
    
    def add_entities(self, entities : list) -> None:
        for entity in entities:
            self.add_entity(entity)
    
    def get_entities(self) -> list:
        return list((self.get_D1_entities() | self.get_D2_entities()))
           
    def get_D1_entities(self) -> set:
        return self.d1_entities
    
    def get_D2_entities(self) -> set:
        return self.d2_entities 
    
    def has_entities(self) -> bool:
        return self.has_D1_entities() or self.has_D2_entities()
    
    def has_D1_entities(self) -> bool:
        return (len(self.d1_entities) > 0)
    
    def has_D2_entities(self) -> bool:
        return (len(self.d1_entities) > 0)
    
    def has_entity(self, entity : int) -> bool:
        target_dataset_entities = self.get_entity_dataset(entity)
        return (entity in target_dataset_entities)
    
    def remove_entity(self, entity: int) -> None:
        target_dataset_entities = self.get_entity_dataset(entity)
        target_dataset_entities.remove(entity)
        
    def remove_entities(self, entities: list) -> None:
        for entity in entities:
            self.remove_entity(entity)        
    
    def flatten(self) -> list:
        flattened_cluster : list = []
        
        for d1_entity in self.d1_entities:
            flattened_cluster.append(d1_entity) 
        for d2_entity in self.d2_entities:
            flattened_cluster.append(d2_entity)
            
        return flattened_cluster 
    
    def evaluate(self,
                 prediction=None,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        pass

    def _configuration(self) -> dict:
        pass

    def stats(self) -> None:
        pass

class ExtendedSimilarityEdge(PYJEDAIFeature):
    def __init__(self,
                 left_node : int,
                 right_node : int,
                 similarity : float,
                 active : bool = True) -> None:
        super().__init__()
        self.set_left_node(left_node=left_node)
        self.set_right_node(right_node=right_node)
        self.set_similarity(similarity=similarity)
        self.set_active(active=active)
        
    def set_left_node(self, left_node : int):
        self.left_node : int = left_node
        
    def set_right_node(self, right_node : int):
        self.right_node : int = right_node

    def set_similarity(self, similarity : float):
        self.similarity : float = similarity
        
    def set_active(self, active : bool):
        self.active : bool = active
        
    def is_active(self):
        return self.active
        
    def __lt__(self, other):
        return self.similarity < other.similarity

    def __le__(self, other):
        return self.similarity <= other.similarity

    def __eq__(self, other):
        return self.similarity == other.similarity

    def __ne__(self, other):
        return self.similarity != other.similarity

    def __gt__(self, other):
        return self.similarity > other.similarity

    def __ge__(self, other):
        return self.similarity >= other.similarity

    def evaluate(self,
                 prediction=None,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        pass

    def _configuration(self) -> dict:
        pass

    def stats(self) -> None:
        pass
    
class Vertex(PYJEDAIFeature):
    def __init__(self,
                 identifier : int,
                 edges : list = None) -> None:
        super().__init__()
        self.set_identifier(identifier=identifier)
        self.set_attached_edges(attached_edges=0)
        self.set_weight_sum(weight_sum=0)
        self.set_edges(edges={})
        if(edges is not None): self.insert_edges(edges=edges)
        
    def set_identifier(self, identifier : int) -> None:
        self.identifier : int = identifier
        
    def set_attached_edges(self, attached_edges : int) -> None:
        self.attached_edges : int = attached_edges
        
    def set_weight_sum(self, weight_sum : float) -> None:
        self.weight_sum : float = weight_sum
        
    def set_edges(self, edges : dict) -> None:
        self.edges : dict = edges

    def set_average_weight(self, average_weight : float) -> None:
        self.average_weight : float = average_weight
    
    def insert_edges(self, edges : list) -> None:
        for edge in edges:
            self.insert_edge(edge=edge)
        
    def insert_edge(self, edge : tuple) -> None:
        vertex, weight = edge
        self.update_weight_sum_by(update_value=weight)
        self.update_attached_edges_by(update_value=1)
        self.edges[vertex] = weight
        self.update_average_weight()
        
    def remove_edges(self, edges : list) -> None:
        for edge in edges:
            self.remove_edge(edge=edge)
                
    def remove_edge(self, edge : int) -> None:
        weight = self.edges.pop(edge, None)
        if(weight is not None):
            self.update_attached_edges_by(update_value=-1)
            self.update_weight_sum_by(update_value=-weight)
            self.update_average_weight()
            
    def get_attached_edges(self) -> int:
        return self.attached_edges
    
    def get_weight_sum(self) -> float:
        return self.weight_sum
    
    def get_edges(self) -> list:
        return self.edges
    
    def get_identifier(self) -> int:
        return self.identifier 
    
    def get_similarity_with(self, entity : int) -> float:
        return self.edges[entity] if entity in self.edges else 0.0
    
    def update_weight_sum_by(self, update_value : float) -> None:
        self.set_weight_sum(self.get_weight_sum() + update_value)
        
    def update_attached_edges_by(self, update_value : float) -> None:
        self.set_attached_edges(self.get_attached_edges() + update_value)
        
    def update_average_weight(self, negative = True) -> None:
        _average_weight : float = (self.get_weight_sum() / self.get_attached_edges())
        _average_weight = -_average_weight if negative else _average_weight 
        self.set_average_weight(average_weight=_average_weight)
        
    def has_edges(self):
        return (self.get_attached_edges() > 0)
        
    def __lt__(self, other):
        return self.average_weight < other.average_weight

    def __le__(self, other):
        return self.average_weight <= other.average_weight

    def __eq__(self, other):
        return self.average_weight == other.average_weight

    def __ne__(self, other):
        return self.average_weight != other.average_weight

    def __gt__(self, other):
        return self.average_weight > other.average_weight

    def __ge__(self, other):
        return self.average_weight >= other.average_weight
    
    def evaluate(self,
                 prediction=None,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        pass

    def _configuration(self) -> dict:
        pass

    def stats(self) -> None:
        pass
    
class RicochetCluster(PYJEDAIFeature):
    def __init__(self,
                 center : int,
                 members : []) -> None:
        super().__init__()
        self.set_center(center=center)
        self.set_members(members=set())
        self.add_members(new_members=members)
    
    def set_center(self, center : int) -> None:
        self.center : int = center
    
    def set_members(self, members : set) -> None:
        self.members : set = members
    
    def add_members(self, new_members : list) -> None:
        for new_member in new_members:
            self.add_member(new_member)
    
    def add_member(self, new_member: int) -> None:
        self.members.add(new_member)
        
    def remove_member(self, member : int) -> None:
        self.members.remove(member)
        
    def get_members(self) -> list:
        return self.members
    
    def get_center(self) -> int:
        return self.center
    
    def change_center(self, new_center : int):
        self.remove_member(member=self.get_center())
        self.add_member(new_member=new_center)
        self.set_center(center=new_center)
        
    def evaluate(self,
                 prediction=None,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        pass

    def _configuration(self) -> dict:
        pass

    def stats(self) -> None:
        pass
        
class AbstractClustering(PYJEDAIFeature):
    
    _method_name: str = "Abstract Clustering"
    _method_short_name: str = "AC"
    _method_info: str = "Abstract Clustering Method"
    
    def __init__(self) -> None:
        super().__init__()
        self.data: Data
        self.similarity_threshold: float = 0.1
        self.execution_time: float = 0.0
        
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

    def _configuration(self) -> dict:
        pass

    import pandas as pd

    def export_to_df(self, prediction: list, tqdm_enable:bool = False) -> pd.DataFrame:
        """Creates a dataframe for the evaluation report.

        Args:
            prediction (list): Predicted clusters.

        Returns:
            pd.DataFrame: Dataframe containing evaluation scores and stats.
        """
        pairs_list = []

        dataset_limit = self.data.dataset_limit
        is_dirty_er = self.data.is_dirty_er
        gt_to_ids_reversed_1 = self.data._gt_to_ids_reversed_1
        if not is_dirty_er:
            gt_to_ids_reversed_2 = self.data._gt_to_ids_reversed_2

        for cluster in tqdm(prediction, desc="Exporting to DataFrame", disable=not tqdm_enable):
            lcluster = list(cluster)

            for i1 in range(len(lcluster)):
                for i2 in range(i1 + 1, len(lcluster)):
                    node1 = lcluster[i1]
                    node2 = lcluster[i2]

                    if node1 < dataset_limit:
                        id1 = gt_to_ids_reversed_1[node1]
                        id2 = gt_to_ids_reversed_1[node2] if is_dirty_er else gt_to_ids_reversed_2[node2]
                    else:
                        id2 = gt_to_ids_reversed_2[node1]
                        id1 = gt_to_ids_reversed_1[node2]

                    pairs_list.append((id1, id2))

        pairs_df = pd.DataFrame(pairs_list, columns=['id1', 'id2'])

        return pairs_df

    
    def sorted_indicators(self, first_indicator : int, second_indicator : int):
        return (first_indicator, second_indicator) if (first_indicator < second_indicator) else (second_indicator, first_indicator)

    def id_to_index(self, identifier : int):
        return identifier \
            if identifier < self.data.dataset_limit \
            else (identifier - self.data.dataset_limit)
        
    def index_to_id(self, index : int, left_dataset : True):
        return index if left_dataset else index + self.data.dataset_limit 

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
        return {
            "Similarity Threshold": self.similarity_threshold
        }

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
        matched_entities = OrderedSet()
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
        return {
            "Similarity Threshold": self.similarity_threshold
        }

class ExactClustering(AbstractClustering):
    """Implements an adapted, simplified version of the Exact THRESHOLD algorithm,
        introduced in "Similarity Flooding: A Versatile Graph Matching Algorithm and Its Application to Schema Matching",
        also referred in "BIGMAT: A Distributed Affinity-Preserving Random Walk Strategy for Instance Matching on Knowledge Graphs".
        In essence, it keeps the top-1 candidate per entity, as long as the candidate also considers this node as its top candidate.
    """

    _method_name: str = "Exact Clustering"
    _method_short_name: str = "EC"
    _method_info: str = "Implements an adapted, simplified version of the Exact THRESHOLD algorithm," + \
        "In essence, it keeps the top-1 candidate per entity, as long as the candidate also considers this node as its top candidate."

    def __init__(self) -> None:
        """"""
        super().__init__()

    def process(self, graph: Graph, data: Data, similarity_threshold: float = 0.1) -> list:
        """
        """
        self.similarity_threshold = similarity_threshold
        raise NotImplementedError("Exact Clustering is not implemented yet.")

    def _configuration(self) -> dict:
        return {
            "Similarity Threshold": self.similarity_threshold
        }

class CenterClustering(AbstractClustering):
    """Implements the Center Clustering algorithm. Input comparisons (graph edges) are sorted in descending order of similarity.
       Pairs of entities connected by these edges form the basis of the updated graph. Entities are evaluated to determine if they will serve
       as a center of a future cluster or as its member. This evaluation is based on a comparison of their cumulative edge weights in the graph,
       normalized by the number of edges in which they are involved. Finally, the algorithm identifies connected components within the graph,
       using the previously defined centers as the focal points for forming clusters.
    """


    _method_name: str = "Center Clustering"
    _method_short_name: str = "CC"
    _method_info: str = "Implements the Center Clustering algorithm," + \
        "In essence, it keeps it defines if a node within an edge constitutes a center or member of future clusters" + \
        " by normalized over the graph weight sum comparison"
    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold: float

    def process(self, graph: Graph, data: Data, similarity_threshold: float = 0.5) -> list:

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.data = data
        edges_weight = defaultdict(float)
        edges_attached = defaultdict(int)
        comparisons = PriorityQueue(maxsize = graph.number_of_edges()*2)

        for (v1, v2, data) in graph.edges(data=True):
            similarity_score = data.get('weight', 0)
            if similarity_score > self.similarity_threshold:
                comparisons.put((-similarity_score, v1, v2))
                edges_weight[v1] = edges_weight[v1] + similarity_score
                edges_weight[v2] = edges_weight[v2] + similarity_score

                edges_attached[v1] = edges_attached[v1] + 1
                edges_attached[v2] = edges_attached[v2] + 1

        new_graph = Graph()
        cluster_centers = set()
        cluster_members = set()

        while not comparisons.empty():
            similarity_score, v1, v2 = comparisons.get()
            v1_is_center : bool = v1 in cluster_centers
            v2_is_center : bool = v2 in cluster_centers
            v1_is_member : bool = v1 in cluster_members
            v2_is_member : bool = v2 in cluster_members
            
            if(not(v1_is_center or v2_is_center or v1_is_member or v2_is_member)):
                w1 = edges_weight[v1] / edges_attached[v1]
                w2 = edges_weight[v2] / edges_attached[v2]

                cluster_centers.add(v1 if w1 > w2 else v2)
                cluster_members.add(v1 if w1 <= w2 else v2)
                new_graph.add_edge(v1, v2, weight=-similarity_score)
            elif ((v1_is_center and v2_is_center) or (v1_is_member and v2_is_member)):
                continue
            elif (v1_is_center and not v2_is_member):
                cluster_members.add(v2)
                new_graph.add_edge(v1, v2, weight=-similarity_score)
            elif (v2_is_center and not v1_is_member):
                cluster_members.add(v1)
                new_graph.add_edge(v1, v2, weight=-similarity_score)

        clusters = list(connected_components(new_graph))
        self.execution_time = time() - start_time
        return clusters

    def _configuration(self) -> dict:
        return {
            "Similarity Threshold": self.similarity_threshold
        }

class BestMatchClustering(AbstractClustering):
    """Implements the Best Match Clustering algorithm. Based on supplied order, it either traverse the entities of the left (inorder)
       or right (reverse) dataset. For each entity, it retrieves all of its candidate pairs, stores them in descending similarity order.
       For each source entity, only the best candidate is kept (only highest similarity edge is kept in the new graph).
    """

    _method_name: str = "Best Match Clustering"
    _method_short_name: str = "BMC"
    _method_info: str = "Implements the Best Match Clustering algorithm," + \
        "In essence, it keeps the best candidate for each entity of the source dataset (defined through ordering)"
    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold: float

    def process(self, graph: Graph, data: Data, similarity_threshold: float = 0.5, order : str = "inorder") -> list:

        start_time = time()
        self.data = data
        self.similarity_threshold : float = similarity_threshold
        self.order : str = order
        
        if(self.order != "inorder" and self.order != "reverse"):
             raise ValueError(f"Best Match Clustering doesn't support {self.order} ordering - Use inorder/reverse.")
        
        number_of_comparisons = len(graph.edges(data=True))
        matched_entities = set()
        new_graph = Graph()
        candidates_of = {} 
        clusters = []
        
        if(number_of_comparisons == 0):
            return clusters

        if self.data.is_dirty_er:
            raise ValueError(f"Best Match Clustering doesn't support Dirty ER.")

        source_entities_num = self.data.num_of_entities_1 \
                              if(self.order == "inorder") else \
                              self.data.num_of_entities_2

        candidates_of = [PriorityQueue() for _ in range(source_entities_num)]

        for (v1, v2, data) in graph.edges(data=True):
            similarity_score = data.get('weight', 0)
            original_d1_entity, original_d2_entity = (v1, v2) if (v1 < v2) else (v2, v1)
            
            source_entity, target_entity = (original_d1_entity, original_d2_entity) \
                                           if(self.order == "inorder") else \
                                           (original_d2_entity, original_d1_entity)
                                           
            source_index = source_entity \
                           if(self.order == "inorder") else \
                           source_entity - self.data.dataset_limit
            
            if similarity_score > self.similarity_threshold:
                candidates_of[source_index].put((-similarity_score, target_entity))

        for source_index, source_candidates in enumerate(candidates_of):
            while not source_candidates.empty():
                similarity, target_entity = source_candidates.get()
                
                if target_entity in matched_entities:
                    continue
                
                source_entity = source_index \
                                if(self.order == "inorder") else \
                                source_index + self.data.dataset_limit 

                e1, e2 = (source_entity, target_entity) \
                         if(self.order == "inorder") else \
                         (target_entity, source_entity)
                new_graph.add_edge(e1, e2, weight=-similarity)
                matched_entities.add(source_entity)
                matched_entities.add(target_entity)
                break

        clusters = list(connected_components(new_graph))
        self.execution_time = time() - start_time
        return clusters

    def _configuration(self) -> dict:
        return {
            "Similarity Threshold": self.similarity_threshold
        }
    
    def set_order(self, order : str) -> None:
        self.order : str = order
        
        
class MergeCenterClustering(AbstractClustering):
    """Implements the Merge Center Clustering algorithm. It is a simplified version of the Center Clustering algorithm,
       where the pair entities are not chosen as cluster center and member respectively based on their cumulative, normalized
       weight in the original graph. Rather, entities of the left dataset are set as centers and their right dataset candidates
       are set as member of the corresponding clusters.  
    """


    _method_name: str = "Merge Center Clustering"
    _method_short_name: str = "MCC"
    _method_info: str = "Implements the Merge Center Clustering algorithm," + \
        "In essence, it implements Center Clustering without the cumulative, " + \
        "normalized weight calculation. Left dataset entities are set as candidate cluster centers."
    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold: float

    def process(self, graph: Graph, data: Data, similarity_threshold: float = 0.5) -> list:

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.data = data
        comparisons = PriorityQueue(maxsize = graph.number_of_edges()*2)

        for (v1, v2, data) in graph.edges(data=True):
            similarity_score = data.get('weight', 0)
            d1_id, d2_id = self.sorted_indicators(v1, v2)
            if similarity_score > self.similarity_threshold:
                comparisons.put((-similarity_score, d1_id, d2_id))

        new_graph = Graph()
        cluster_centers = set()
        cluster_members = set()

        while not comparisons.empty():
            similarity_score, v1, v2 = comparisons.get()
            v1_is_center : bool = v1 in cluster_centers
            v2_is_center : bool = v2 in cluster_centers
            v1_is_member : bool = v1 in cluster_members
            v2_is_member : bool = v2 in cluster_members
            
            if(not(v1_is_center or v2_is_center or v1_is_member or v2_is_member)):
                cluster_centers.add(v1)
                cluster_members.add(v2)
                new_graph.add_edge(v1, v2, weight=-similarity_score)
            elif ((v1_is_center and v2_is_center) or (v1_is_member and v2_is_member)):
                continue
            elif (v1_is_center):
                cluster_members.add(v2)
                new_graph.add_edge(v1, v2, weight=-similarity_score)
            elif (v2_is_center):
                cluster_members.add(v1)
                new_graph.add_edge(v1, v2, weight=-similarity_score)

        clusters = list(connected_components(new_graph))
        self.execution_time = time() - start_time
        return clusters

    def _configuration(self) -> dict:
        return {
            "Similarity Threshold": self.similarity_threshold
        }
    
class CorrelationClustering(AbstractClustering):
    """Implements the Correlation Clustering algorithm. Candidate pairs are mapped into a graph, whose connected components
       act as our initial clusters. We iteratively choose one of the 3 possible moves (change, merge, break up cluster) and
       we apply them on randomly chosen entities. We decide whether we should conduct the move or not, based on an objective function,
       which quantifies the quality of our clusters (contain similar entities, seperate disimilar ones)   
    """


    _method_name: str = "Correlation Clustering"
    _method_short_name: str = "CC"
    _method_info: str = "Implements the Correlation Clustering algorithm," + \
        "In essence, it implements iterative clustering, " + \
        "reassigning clusters to randomly chosen entities based on the reassignment's effect on our objective function " + \
        "that evaluates the quality of the newly defined clusters." 
    
    def __init__(self) -> None:
        super().__init__()
        self.initial_threshold : float
        self.similarity_threshold : float
        self.non_similarity_threshold : float
        self.move_limit : int
        self.lsi_iterations: int

    def process(self,
                graph: Graph,
                data: Data,
                initial_threshold: float = 0.5,
                similarity_threshold: float = 0.8,
                non_similarity_threshold: float = 0.2,
                move_limit: int = 3,
                lsi_iterations: int = 100) -> list:

        start_time = time()
        self.data : Data = data
        self.initial_threshold : float = initial_threshold
        self.similarity_threshold : float = similarity_threshold
        self.non_similarity_threshold : float = non_similarity_threshold
        self.move_limit : int = move_limit
        self.lsi_iterations: int = lsi_iterations
        
        self.num_of_source_entities = self.data.num_of_entities_1
        self.num_of_target_entities = self.data.num_of_entities_1 if self.data.is_dirty_er \
                                      else self.data.num_of_entities_2
        
        self.similarity = lil_matrix((self.num_of_source_entities, self.num_of_target_entities), dtype=float)
        new_graph = graph.copy()

        for (v1, v2, data) in graph.edges(data=True):
            d1_id, d2_id = self.sorted_indicators(v1, v2)
            d1_index, d2_index = (self.id_to_index(d1_id), self.id_to_index(d2_id))
            similarity_score = data['weight']
            self.similarity[d1_index, d2_index] = similarity_score 
            if similarity_score < self.initial_threshold:
                new_graph.remove_edge(v1, v2)

        initial_clusters = [list(connected_component) for connected_component in connected_components(new_graph)]
        
        self.clusters = [EquivalenceCluster(data=self.data, flattened_cluster=cluster) for cluster in initial_clusters]
        self.initial_clusters_num = len(initial_clusters) 
        self.max_clusters_num = self.initial_clusters_num + 10
        self.entity_cluster_index = [0] * self.data.num_of_entities
        self.valid_entities = set()
        
        for cluster_index, cluster in enumerate(self.clusters):
            for entity in range(self.data.num_of_entities):
                if(cluster.has_entity(entity=entity)):
                    self.valid_entities.add(entity)
                    self.entity_cluster_index[entity] = cluster_index
        self.valid_entities = list(self.valid_entities)
                              
        self.similar = lil_matrix((self.num_of_source_entities, self.num_of_target_entities), dtype=bool)
        self.not_similar = lil_matrix((self.num_of_source_entities, self.num_of_target_entities), dtype=bool)

        for d1_index in range(self.num_of_source_entities):
            for d2_index in range(d1_index, self.num_of_target_entities):
                self.not_similar[d1_index, d2_index] = self.similarity[d1_index, d2_index] < self.non_similarity_threshold
                self.similar[d1_index, d2_index] = self.similarity[d1_index, d2_index] > self.similarity_threshold
        
        random.seed(RANDOM_SEED)
        previous_OF : int = self.calculate_OF()
        
        for iteration in range(self.lsi_iterations):
            move_index : int = self.choose_move()
            current_OF : int = self.move(move_index, previous_OF)
            previous_OF = current_OF

        final_clusters : list = []
        for cluster in self.clusters:
            if(cluster.has_entities()):
                final_clusters.append(set(cluster.flatten()))
        self.execution_time = time() - start_time
        return final_clusters
    
    def calculate_OF(self) -> int:
        OF : int = 0
        
        for d1_index in range(self.num_of_source_entities):
            for d2_index in range(d1_index, self.num_of_target_entities):
                d1_entity = self.index_to_id(index=d1_index, left_dataset=True)
                d2_entity = self.index_to_id(index=d2_index, left_dataset=self.data.is_dirty_er)
                
                similar_and_cluster_match = self.similar[d1_index, d2_index] and \
                (self.entity_cluster_index[d1_entity] == self.entity_cluster_index[d2_entity])
                dissimilar_and_cluster_missmatch = self.not_similar[d1_index, d2_index] and \
                (self.entity_cluster_index[d1_entity] != self.entity_cluster_index[d2_entity])
                
                if(similar_and_cluster_match or dissimilar_and_cluster_missmatch):
                    OF += 1
                    
        return OF   
     
    def choose_move(self) -> int:
        move = random.randint(0, self.move_limit - 1)
        while(move == 1 and len(self.clusters) == 1):
            move = random.randint(0, self.move_limit - 1)
        return move
            
    def move(self, move_index : int, previous_OF : int):

        if(move_index == 0):
            random_entity = random.choice(self.valid_entities)
            random_cluster = random.randint(0, self.initial_clusters_num - 1)
            while(not self.clusters[random_cluster].has_entities()):
                random_cluster = random.randint(0, self.initial_clusters_num - 1)
            return self.change_entity_cluster(previous_OF, random_entity, random_cluster)
        elif(move_index == 1):
            previous_cluster = random.randint(0, self.initial_clusters_num - 1)
            while(not self.clusters[previous_cluster].has_entities()):
                previous_cluster = random.randint(0, self.initial_clusters_num - 1)
                
            new_cluster = random.randint(0, self.initial_clusters_num - 1)
            while((previous_cluster == new_cluster) or (not self.clusters[new_cluster].has_entities())):
                new_cluster = random.randint(0, self.initial_clusters_num - 1)
                
            return self.unify_clusters(previous_OF, previous_cluster, new_cluster)
        
        elif(move_index == 2):
            previous_cluster = random.randint(0, self.initial_clusters_num - 1)
            while(not self.clusters[previous_cluster].has_entities()):
                previous_cluster = random.randint(0, self.initial_clusters_num - 1)
            return self.seperate_clusters(previous_OF, previous_cluster)
        else:
            raise ValueError(f"Invalid Move Index \"{move_index}\": Choose 0->2")
        
        
    def change_entity_cluster(self, previous_OF : int, entity : int, new_cluster : int):
        previous_cluster = self.entity_cluster_index[entity]
        self.entity_cluster_index[entity] = new_cluster
        
        new_OF = self.calculate_OF()

        if(new_OF > previous_OF):
            self.clusters[previous_cluster].remove_entity(entity)
            self.clusters[new_cluster].add_entity(entity)
            return new_OF
        else:
            self.entity_cluster_index[entity] = previous_cluster
            return previous_OF
        
    def unify_clusters(self, previous_OF : int, previous_cluster_index : int, new_cluster_index : int):
        previous_cluster = self.clusters[previous_cluster_index]
        new_cluster = self.clusters[new_cluster_index]
        to_be_removed_entities = []    
        previous_cluster_entities = previous_cluster.get_entities()
    
        for entity in previous_cluster_entities:
            to_be_removed_entities.append(entity)
            self.entity_cluster_index[entity] = new_cluster_index
        
        new_OF : int = self.calculate_OF()

        if(new_OF > previous_OF):
            previous_cluster.remove_entities(previous_cluster_entities)
            new_cluster.add_entities(previous_cluster_entities)
            return new_OF

        for to_be_removed_entity in to_be_removed_entities:
            self.entity_cluster_index[to_be_removed_entity] = previous_cluster_index
            
        return previous_OF
    
    def seperate_clusters(self, previous_OF, previous_cluster_index):
        previous_cluster = self.clusters[previous_cluster_index]
        previous_cluster_entities = previous_cluster.get_entities()
        to_be_removed_entities = []
        new_cluster_index = self.initial_clusters_num
        
        for index in range(0, len(previous_cluster_entities), 2):
            to_be_removed_entity = previous_cluster_entities[index]
            to_be_removed_entities.append(to_be_removed_entity)
            self.entity_cluster_index[to_be_removed_entity] = new_cluster_index
        
        new_OF : int = self.calculate_OF()
        # print(previous_OF, new_OF)
        if(new_OF > previous_OF):
            self.clusters.append(EquivalenceCluster(data=self.data, flattened_cluster=to_be_removed_entities))
            self.initial_clusters_num += 1
            previous_cluster.remove_entities(to_be_removed_entities)    
            return new_OF
        
        for to_be_removed_entity in to_be_removed_entities:
            self.entity_cluster_index[to_be_removed_entity] = previous_cluster_index
              
        return previous_OF

    def _configuration(self) -> dict:
        return {
            "Initial Threshold" : self.initial_threshold,
            "Similarity Threshold" : self.similarity_threshold,
            "Non-Similarity Threshold" : self.non_similarity_threshold,
            "Move limit" : self.move_limit,
            "LSI Iterations" : self.lsi_iterations
        }
    
class CutClustering(AbstractClustering):
    """Implements the Cut Clustering algorithm. Retains the candidate pairs whose similarity is over the specified threshold.
       Those pairs are mapped into graph edges. Using the newly defined graph, we retrieve its Gomory Hu Tree representation
       using the Edmonds Karp flow function, while edges' capacity is considered to be infinite. We return the connected components
       of the resulting minimum s-t cuts for the pairs in the original, trimmed graph.    
    """

    _method_name: str = "Cut Clustering"
    _method_short_name: str = "CTC"
    _method_info: str = "Implements the Cut Clustering algorithm," + \
        "In essence, it calculates the Gomory Hu Tree of the graph resulting from input similarity pairs. " + \
        "We retain the connected components of this tree."
    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold: float

    def process(self, graph: Graph, data: Data, similarity_threshold: float = 0.5, alpha: float = 0.2) -> list:

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.data = data
        threshold_trimmed_graph : Graph = Graph()

        for (v1, v2, data) in graph.edges(data=True):
            similarity_score = data.get('weight', 0)
            d1_id, d2_id = self.sorted_indicators(v1, v2)
            if similarity_score > self.similarity_threshold:
                threshold_trimmed_graph.add_edge(d1_id, d2_id, weight=similarity_score)
        
        sink_node : int = self.data.num_of_entities  
        threshold_trimmed_graph.add_node(sink_node)
        for node in graph.nodes():
            if node != sink_node:
                threshold_trimmed_graph.add_edge(sink_node, node, weight=alpha)

        final_gomory_hu_tree = gomory_hu_tree(G=threshold_trimmed_graph, capacity='weight')
        final_gomory_hu_tree.remove_node(sink_node)
        clusters = list(connected_components(final_gomory_hu_tree))
        
        # print(len(clusters))
        self.execution_time = time() - start_time
        return clusters

    def _configuration(self) -> dict:
        return {
            "Similarity Threshold": self.similarity_threshold
        }
    
class MarkovClustering(AbstractClustering):
    """Implements the Markov Clustering algorithm. It simulates random walks on a (n x n) matrix as the adjacency matrix
       of a weighted, similarity graph. It alternates an expansion step and an inflation step until an equilibrium state is reached.
       Entries with similarity above threhold, are inserted into final graph, whose CCs we retain.    
    """

    _method_name: str = "Markov Clustering"
    _method_short_name: str = "MCL"
    _method_info: str = "Ιmplements the Markov Clustering algorithm," + \
        "In essence, it simulates random walks on a (n x n) matrix as the adjacency " + \
        "matrix of a graph. It alternates an expansion step and an inflation step " + \
        "until an equilibrium state is reached. We retain the connected components " + \
        "of the graph resulting from final similarity matrix entries valued over threshold."
    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold : float
        self.cluster_threshold : float
        self.matrix_similarity_threshold : float
        self.similarity_checks_limit : int
        
    def process(self, graph: Graph,
                data: Data, 
                similarity_threshold: float = 0.5,
                cluster_threshold: float = 0.001,
                matrix_similarity_threshold: float = 0.00001,
                similarity_checks_limit : int = 10) -> list:

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.cluster_threshold : float = cluster_threshold
        self.matrix_similarity_threshold : float = matrix_similarity_threshold
        self.similarity_checks_limit : int = similarity_checks_limit
        self.data = data
        self.current_similarity = lil_matrix((self.data.num_of_entities, data.num_of_entities), dtype=float)
        new_graph : Graph = Graph()
        
        
        for (v1, v2, data) in graph.edges(data=True):
            d1_id, d2_id = self.sorted_indicators(v1, v2)
            similarity_score = data.get('weight', 0)
            
            if(similarity_score > self.similarity_threshold):
                self.current_similarity[d1_id, d2_id] = similarity_score 
                self.current_similarity[d2_id, d1_id] = similarity_score 
        
        self.set_node_loop(similarity = 1.0)
        self.normalize()
        
        for check in range(self.similarity_checks_limit):
            self.previous_similarity = self.current_similarity.copy()
            self.inflate()
            self.normalize()
            self.expand()
            self.normalize()
            # print(check+1)
            if(self.equilibrium()):
                break
        
        edges_populated = self.get_existing_indices(matrix=self.current_similarity)    
        for edge in edges_populated:
            row, column = edge
            new_similarity = self.current_similarity[row, column]
            final_row, final_column = self.sorted_indicators(row, column)
            
            if(new_graph.has_edge(final_row, final_column)):
                existing_similarity = new_graph[final_row][final_column]["weight"]
                if(new_similarity > existing_similarity):
                    new_graph[final_row][final_column]["weight"] = new_similarity
            elif(new_similarity > self.cluster_threshold):
                new_graph.add_edge(final_row, final_column, weight=new_similarity)   
        
        clusters = list(connected_components(new_graph))
        self.execution_time = time() - start_time
        return clusters
    
    def set_node_loop(self, similarity : float = 1.0) -> None:
        rows : int = self.current_similarity.shape[0]
        # print(rows)
        for row in range(rows):
            self.current_similarity[row, row] = similarity
            
    def normalize(self) -> None:
        column_sums = self.current_similarity.sum(axis=0)
        column_sums[column_sums == 0] = 1
        self.current_similarity = self.current_similarity.multiply(1. / column_sums)
        
    def expand(self) -> None:
        self.current_similarity = self.current_similarity.power(2)
        
    def inflate(self) -> None:
        self.current_similarity = self.current_similarity.dot(self.current_similarity)
        
    def equilibrium(self) -> None:
        self.current_similarity = self.current_similarity.tocsr()
        self.previous_similarity = self.previous_similarity.tocsr()
        
        current_indices = self.get_existing_indices(matrix=self.current_similarity)
        previous_indices = self.get_existing_indices(matrix=self.previous_similarity)
        shared_indices = current_indices & previous_indices
        
        for indices in shared_indices:
            row, column = indices
            if(abs(self.current_similarity[row, column] - self.previous_similarity[row, column]) > self.matrix_similarity_threshold):
                return False
            
        return True  
    
    def get_existing_indices(self, matrix):
        return set([indices for indices in zip(*matrix.nonzero())])
    
    def _configuration(self) -> dict:
        return {
            "Similarity Threshold" : self.similarity_threshold,
            "Cluster Threshold" : self.cluster_threshold,
            "Matrix Similarity Threshold" : self.matrix_similarity_threshold,
            "Similarity Checks Limit" : self.similarity_checks_limit
        }
    
class KiralyMSMApproximateClustering(AbstractClustering):
    """Implements the Kiraly MSM Approximate Clustering algorithm. Implements the so-called "New Algorithm"
       by Zoltan Kiraly 2013, which is a 3/2-approximation to the Maximum Stable Marriage (MSM) problem.
       The pairs resulting from the approximation of the stable relationships are translated into a graph,
       whose connected components we retain.    
    """

    _method_name: str = "Kiraly MSM Approximate Clustering"
    _method_short_name: str = "KMAC"
    _method_info: str = "Ιmplements the Kiraly MSM Approximate Clustering algorithm," + \
        "In essence, it is a 3/2-approximation to the Maximum Stable Marriage (MSM) problem."
    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold : float
        
    def process(self, 
                graph: Graph,
                data: Data, 
                similarity_threshold: float = 0.1) -> list:

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.data = data
        number_of_comparisons : int = len(graph.edges(data=True))
        clusters : list = []
        
        if(number_of_comparisons == 0):
            return clusters

        if self.data.is_dirty_er:
            raise ValueError(f"Kiraly MSM Approximate Clustering doesn't support Dirty ER.")

        new_graph : Graph = Graph()
        men : set = set()
        self.men_candidates : dict = defaultdict(list)
        self.women_candidates : dict = defaultdict(list)
        
        for (v1, v2, data) in graph.edges(data=True):
            man, woman = self.sorted_indicators(v1, v2)
            similarity = data.get('weight', 0) 
            if similarity > self.similarity_threshold:    
                self.men_candidates[man].append(ExtendedSimilarityEdge(left_node=man,
                                                                right_node=woman,
                                                                similarity=similarity))   
                self.women_candidates[woman].append(ExtendedSimilarityEdge(left_node=woman,
                                                                    right_node=man,
                                                                    similarity=similarity))
            men.add(man)
            
        for man, candidates in self.men_candidates.items():
            self.men_candidates[man] = sorted(candidates, reverse=True)
        for woman, candidates in self.women_candidates.items():
            self.women_candidates[woman] = sorted(candidates, reverse=True)
            
        self.is_bachelor : list = [False] * self.data.num_of_entities_1
        self.is_uncertain : list = [False] * self.data.num_of_entities_1
        self.fiances : list = [-1] * self.data.num_of_entities_2
        self.current_matches : dict = {}
        self.free_men : list = list(men)
        
        while(len(self.free_men) > 0):
            man = self.free_men.pop(0)
            woman = self.get_first_active_candidate(entity=man)
            
            if(woman == -1):
                if(not self.is_bachelor[man]):
                    self.is_bachelor[man] = True
                    if(not self.has_candidates(entity=man)):
                        self.free_men.append(man)
                    self.activate_candidates_of(entity=man)
                else:
                    continue
            else:
                fiance = self.get_woman_fiance(woman=woman)
                if(fiance == -1):
                    self.add_match(man=man, woman=woman, similarity=0.0)
                    self.set_woman_fiance(woman=woman, fiance=man)
                else:
                    if(self.accepts_proposal(woman=woman,
                                             man=man)):
                        self.remove_match(man=fiance, woman=woman)
                        self.add_match(man=man, woman=woman, similarity=0.0)
                        self.set_woman_fiance(woman=woman, fiance=man)
                        if(not self.is_uncertain[fiance]):
                            self.deactivate_candidate(entity=fiance, candidate=woman)
                    else:
                        self.deactivate_candidate(entity=man, candidate=woman)
                    
        for _, edges in self.current_matches.items():
            for edge in edges:
                man, woman, similarity = edge.left_node, edge.right_node, edge.similarity
                new_graph.add_edge(man, woman, weight=similarity)
                
        clusters = list(connected_components(new_graph))
        self.execution_time = time() - start_time
        return clusters
    
    def is_male(self, entity: int) -> bool:
        return entity < self.data.dataset_limit
    
    def get_entity_candidates(self, entity : int) -> PriorityQueue:
        candidates = self.men_candidates if self.is_male(entity) else self.women_candidates
        return candidates[entity] 
    
    def has_candidates(self, entity : int) -> bool:
        return len(self.get_entity_candidates(entity=entity)) > 0
    
    def activate_candidates_of(self, entity : int) -> None:
        candidates = self.get_entity_candidates(entity=entity)
        for candidate in candidates:
            candidate.set_active(active=True)     
            
    def get_first_active_candidate(self, entity : int) -> int:
        candidates = self.get_entity_candidates(entity=entity)
        for candidate in candidates:
            if(candidate.is_active()):
                return candidate.right_node 
        return -1
    
    def add_match(self, man : int, woman : int, similarity : float) -> None:
        if man not in self.current_matches:
            self.current_matches[man] = []
        self.current_matches[man].append(ExtendedSimilarityEdge(left_node=man,
                                                                right_node=woman,
                                                                similarity=similarity))
    def remove_match(self, man : int, woman : int) -> None:
        self.current_matches[man] = [match for match in self.current_matches[man] \
                                    if (match.left_node != man or match.right_node != woman)]
        
    def get_woman_fiance(self, woman : int) -> int:
        return self.fiances[woman - self.data.dataset_limit]
    
    def set_woman_fiance(self, woman : int, fiance : int) -> None:
        self.fiances[woman - self.data.dataset_limit] = fiance
        
    def deactivate_candidate(self, entity : int, candidate : int) -> bool:
        entity_candidates = self.get_entity_candidates(entity=entity)
        for entity_candidate in entity_candidates:
            if(entity_candidate.right_node == candidate):
                entity_candidate.set_active(active=False)
                return True
        return False
    
    def accepts_proposal(self, woman : int, man : int):
        current_fiance : int = self.get_woman_fiance(woman=woman)
        
        if(current_fiance == -1):
            return True
        if(self.is_uncertain[current_fiance]):
            return True
        
        man_score : float = 0.0
        current_fiance_score : float = 0.0
        
        woman_candidates : list = self.get_entity_candidates(entity=woman)
        
        for comparison in woman_candidates:
            candidate : int = comparison.right_node
            if(candidate == man):
                man_score = comparison.similarity
            elif(candidate == current_fiance):
                current_fiance_score = comparison.similarity
        
        return (man_score > current_fiance_score)
    
    def _configuration(self) -> dict:
        return {
            "Similarity Threshold": self.similarity_threshold
        }
    
class RicochetSRClustering(AbstractClustering):
    """Implements the Ricochet SR Clustering algorithm. Implements the so-called "New Algorithm"
       by Zoltan Kiraly 2013, which is a 3/2-approximation to the Maximum Stable Marriage (MSM) problem.
       The pairs resulting from the approximation of the stable relationships are translated into a graph,
       whose connected components we retain.    
    """

    _method_name: str = "Ricochet SR Clustering"
    _method_short_name: str = "RSRC"
    _method_info: str = "Implements the Ricochet SR Clustering algorithm," + \
        "In essence, it is a 3/2-approximation to the Maximum Stable Marriage (MSM) problem."
    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold : float
        
    def process(self,
                graph: Graph,
                data: Data, 
                similarity_threshold: float = 0.5) -> list:

        if self.data.is_dirty_er:
            raise ValueError(f"RicochetSRClustering doesn't support Dirty ER.")

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.data = data
        clusters : list = []
        self.vertices : dict = {}
        self.sorted_vertices = PriorityQueue(maxsize = self.data.num_of_entities)
        
        for (v1, v2, data) in graph.edges(data=True):
            d1_id, d2_id = self.sorted_indicators(v1, v2)
            similarity = data.get('weight', 0)
            if similarity > self.similarity_threshold:
                if d1_id not in self.vertices: self.vertices[d1_id] = Vertex(identifier=d1_id)
                if d2_id not in self.vertices: self.vertices[d2_id] = Vertex(identifier=d2_id)
                self.vertices[d1_id].insert_edge(edge=(d2_id, similarity))
                self.vertices[d2_id].insert_edge(edge=(d1_id, similarity))
        
        for _, vertex in self.vertices.items():
            if(vertex.has_edges()):
                self.sorted_vertices.put(vertex)

        if(self.sorted_vertices.empty()):
            return clusters

        self.centers : set = set()
        self.members : set = set()
        self.center_of : dict = {}
        self.similarity_with_center : dict = defaultdict(float)
        self.current_clusters : dict = defaultdict(set)
        
        top_vertex : Vertex = self.sorted_vertices.get()
        vertex_id : int = top_vertex.get_identifier()
        self.centers.add(vertex_id)
        self.center_of[vertex_id] = vertex_id
        self.current_clusters[vertex_id].add(vertex_id)
        self.similarity_with_center[vertex_id] = 1.0
        
        top_vertex_neighbor = list(top_vertex.edges.keys())[0]
        self.members.add(top_vertex_neighbor)
        self.center_of[top_vertex_neighbor] = vertex_id
        self.current_clusters[vertex_id].add(top_vertex_neighbor)
        self.similarity_with_center[top_vertex_neighbor] = top_vertex.get_similarity_with(top_vertex_neighbor)
         
        while(not self.sorted_vertices.empty()):
            vertex = self.sorted_vertices.get()
            vertex_id = vertex.get_identifier() 
            to_reassign : set = set()
            centers_to_reassign : set = set()
            
            for neighbor, similarity in vertex.edges.items():
                if(neighbor in self.centers):
                    continue
                previous_similarity = self.similarity_with_center[neighbor]
                if(previous_similarity >= similarity):
                    continue
                to_reassign.add(neighbor)
                break
            
            if(to_reassign):
                if(vertex_id in self.members):
                    self.members.remove(vertex_id)
                    previous_center = self.center_of[vertex_id]
                    self.current_clusters[previous_center].remove(vertex_id)
                    if(len(self.current_clusters[previous_center]) < 2):
                        centers_to_reassign.add(previous_center)
                to_reassign.add(vertex_id)
                for assignee in to_reassign:
                    self.current_clusters[vertex_id].add(assignee)
                self.centers.add(vertex_id)
                
            for reassign in to_reassign:
                if(reassign != vertex_id):
                    if(reassign in self.members):
                        reassign_previous_center = self.center_of[reassign]
                        self.current_clusters[reassign_previous_center].remove(reassign)
                        
                        if(len(self.current_clusters[reassign_previous_center]) < 2):
                            centers_to_reassign.add(reassign_previous_center)
                    self.members.add(reassign)
                    self.center_of[reassign] = vertex_id
                    self.similarity_with_center[reassign] = vertex.get_similarity_with(reassign)
                    
            for center_to_reassign in centers_to_reassign:
                if(len(self.current_clusters[center_to_reassign]) > 1):
                    continue
                self.centers.remove(center_to_reassign)
                _ = self.current_clusters.pop(center_to_reassign, None)
                
                max_similarity : float = 0.0
                new_center : int = vertex_id
                
                for center in self.centers:
                    new_similarity : float = self.vertices[center].get_similarity_with(center_to_reassign)
                    if(new_similarity > 0.0):
                        if(len(self.current_clusters[center]) > 1):
                            continue
                        if(new_similarity > max_similarity):
                            max_similarity = new_similarity
                            new_center = center
                if(len(self.current_clusters[new_center]) > 1):
                    continue
                self.current_clusters[new_center].add(center_to_reassign)
                self.members.add(center_to_reassign)
                self.center_of[center_to_reassign]= new_center
                self.similarity_with_center[center_to_reassign] = max_similarity
                
        for entity in range(self.data.num_of_entities):
            if(entity not in self.members and entity not in self.centers):
                self.centers.add(entity)
                self.center_of[entity] = entity
                self.current_clusters[entity].add(entity)
                self.similarity_with_center[entity] = 1.0
                    
        clusters = []
        for center, members in self.current_clusters.items():
            center_equivalence_cluster = EquivalenceCluster(data=self.data,
                                                            flattened_cluster=list(members)) 
            clusters.append(set(center_equivalence_cluster.flatten()))
        
        self.execution_time = time() - start_time
        return clusters
    
    def _configuration(self) -> dict:
        return {
            "Similarity Threshold" : self.similarity_threshold
        }
    
    
class RowColumnClustering(AbstractClustering):
    """Implements the Row Column Clustering algorithm. For each row and column find their equivalent
       column and row respectively corresponding to the smallest similarity. Subsequently, chooses
       either rows or columns dependent on which one has the highest out of the lowest similariities 
       on average.        
    """

    _method_name: str = "Row Column Clustering"
    _method_short_name: str = "RCC"
    _method_info: str = "Implements the Row Column Clustering algorithm," + \
        "In essence, it is a 3/2-approximation to the Maximum Stable Marriage (MSM) problem."
    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold : float
        
    def process(self, 
                graph: Graph,
                data: Data, 
                similarity_threshold: float = 0.5) -> list:

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.data = data
        number_of_comparisons : int = len(graph.edges(data=True))
        self.similarity = lil_matrix((self.data.num_of_entities_1, self.data.num_of_entities_2), dtype=float)
        matched_ids = set()
        new_graph : Graph = Graph()
        clusters : list = []
        
        if(number_of_comparisons == 0):
            return clusters
        
        if self.data.is_dirty_er:
            raise ValueError(f"Kiraly MSM Approximate Clustering doesn't support Dirty ER.")
        
        for (v1, v2, data) in graph.edges(data=True):
            d1_id, d2_id = self.sorted_indicators(v1, v2)
            d1_index, d2_index = (self.id_to_index(d1_id), self.id_to_index(d2_id))
            similarity_score = data.get('weight', 0)
            
            if(similarity_score > self.similarity_threshold):
                self.similarity[d1_index, d2_index] = similarity_score 
        
        self.initialize(self.get_negative(self.similarity))
        self.solution_proxy = self.get_solution()
        
        for entry in range(len(self.solution_proxy)):
            d1_index = entry
            d2_index = self.solution_proxy[entry]
            _similarity = self.similarity[d1_index, d2_index]
            if(_similarity < self.similarity_threshold):
                continue
            d2_index += self.data.dataset_limit
            
            if(d1_index in matched_ids or d2_index in matched_ids):
                continue
                
            matched_ids.add(d1_index)
            matched_ids.add(d2_index)
            new_graph.add_edge(d1_index, d2_index, weight=_similarity)
        
        
        clusters = list(connected_components(new_graph))    
        self.execution_time = time() - start_time
        return clusters
    
    def get_min_row(self, column):
        position = -1
        minimum = math.inf
        
        for row in range(self.similarity.shape[0]):
            if(self.row_covered[row]): continue
            if(self.similarity[row, column] < minimum):
                position = row
                minimum = self.similarity[row, column]
                
        return position
    
    def get_min_column(self, row):
        position = -1
        minimum = math.inf
    
        for column in range(self.similarity.shape[1]):
            if(self.column_covered[column]): continue
            if(self.similarity[row, column] < minimum):
                position = column
                minimum = self.similarity[row, column]
                
        return position
    
    def get_row_assignment(self):
        self.row_scan_cost = 0.0
        
        for row in range(self.similarity.shape[0]):
            self.selected_column[row] = self.get_min_column(row)
            _selected_column = self.selected_column[row]
            if(_selected_column == -1): break
            self.column_covered[_selected_column] = True
            self.row_scan_cost += self.similarity[row, _selected_column]        
        
    def get_column_assignment(self):
        self.column_scan_cost = 0.0
        
        for column in range(self.similarity.shape[1]):
            self.selected_row[column] = self.get_min_row(column)
            _selected_row = self.selected_row[column]
            if(_selected_row == -1): break
            self.columns_from_selected_row[_selected_row] = column
            self.row_covered[_selected_row] = True
            self.column_scan_cost += self.similarity[_selected_row, column] 
    
    def get_solution(self):
        self.get_row_assignment()
        self.get_column_assignment()
        
        if(self.row_scan_cost < self.column_scan_cost):
            return self.selected_column
        else:
            return self.columns_from_selected_row
    
    def get_negative(self, similarity_matrix) -> np.array:
        self.negative_similarity = lil_matrix((self.data.num_of_entities_1, self.data.num_of_entities_2), dtype=float)
        
        for row in range(similarity_matrix.shape[0]):
            for column in range(similarity_matrix.shape[1]):
                self.negative_similarity[row, column] = 1.0 - similarity_matrix[row, column]
                
        return self.negative_similarity
    
    def initialize(self, similarity_matrix) -> None:
        self.similarity = similarity_matrix
        self.selected_column = [0] * similarity_matrix.shape[0]
        self.selected_row = [0] * similarity_matrix.shape[1]
        self.row_covered = [False] * similarity_matrix.shape[0]
        self.column_covered = [False] * similarity_matrix.shape[1]
    
        self.columns_from_selected_row = [0] * similarity_matrix.shape[0]    
    
    def _configuration(self) -> dict:
        return {
            "Similarity Threshold" : self.similarity_threshold
        }