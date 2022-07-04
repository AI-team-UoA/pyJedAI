import tqdm
from tqdm import tqdm
import pandas as pd
import networkx as nx
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datamodel import Data


class ConnectedComponentsClustering:

    _method_name: str = "Connected Components Clustering"
    _method_info: str = ": it gets equivalence clsuters from the transitive closure of the similarity graph."

    def __init__(self) -> None:
        self.similarity_threshold: float
        self._progress_bar: tqdm

    def process(self, graph: nx.Graph) -> pd.DataFrame:
        '''
        TODO: comment
        '''
        # connected_components = nx.connected_components(graph)
        # self._progress_bar = tqdm(total=len(list(connected_components)), desc=self._method_name)
        # pairs_df = pd.DataFrame(columns=["id1", "id2"])
        # num_of_pairs = 1
        # # for cc in connected_components:
        #     sorted_component = sorted(cc)
        #     for id1_index in range(0, len(sorted_component), 1):
        #         for id2_index in range(id1_index+1, len(sorted_component), 1):
        #             pairs_df.loc[num_of_pairs] = [sorted_component[id1_index], sorted_component[id2_index]]
        #             num_of_pairs += 1
        #     self._progress_bar.update(1)
        
        return list(nx.connected_components(graph))