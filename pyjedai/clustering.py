from networkx import connected_components, Graph
from time import time

class ConnectedComponentsClustering:
    """Creates the connected components of the graph. \
        Applied to graph created from entity matching. \
        Input graph consists of the entity ids (nodes) and the similarity scores (edges).
    """

    _method_name: str = "Connected Components Clustering"
    _method_info: str = "Gets equivalence clusters from the " + \
                    "transitive closure of the similarity graph."

    def __init__(self) -> None:
        self.execution_time: float

    def process(self, graph: Graph) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """
        start_time = time()
        clusters = list(connected_components(graph))
        self.execution_time = time() - start_time
        return clusters

    def method_configuration(self) -> dict:
        """Returns configuration details
        """
        return {
            "name" : self._method_name,
            "parameters" : self._configuration(),
            "runtime": self.execution_time
        }

    def _configuration(self) -> dict:
        return {}

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )
