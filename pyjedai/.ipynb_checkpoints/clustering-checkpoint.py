import pandas as pd
import networkx as nx
import os
import sys
import time
from .datamodel import Data


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
        start_time = time.time()
        g = list(nx.connected_components(graph))
        self.execution_time = time.time() - start_time
        return g