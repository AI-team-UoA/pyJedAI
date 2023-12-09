
from tqdm.auto import tqdm
from pyjedai.datamodel import SpatialData, PYJEDAIFeature
 
from shapely.geometry import multipolygon
from collections import defaultdict
import math

class AbstractSpatialFiltering(PYJEDAIFeature):
    """Abstract class for the block building method
    """
    def __init__(self):
        super().__init__()

        return
    
    def evaluate(self,
                 prediction,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True,
                 with_stats: bool = False
                ) -> any:
        # TODO itile

        return

    def stats(self, blocks: dict) -> None:
        # TODO itile

        return

class StandardSpatialFiltering(AbstractSpatialFiltering):
    """Abstract class for the block building method
    """
    def __init__(self):
        super().__init__()
        self.spatial_index = defaultdict(lambda: defaultdict(list))

        return

    def process(self, spatial_data:SpatialData)-> dict:
        self.geometries = spatial_data.source_geometries

        self.setThetas()
        self.indexSource()

        # return {"spatial_index":self.spatial_index,"self.theta_x":self.theta_x,"self.theta_y":self.theta_y}
        return (self.spatial_index, self.theta_x, self.theta_y)

    def setThetas(self):
        self.theta_x, self.theta_y = 0, 0
        for sEntity in self.geometries:
            if not isinstance(sEntity, multipolygon.MultiPolygon):
                print("Warning non geometry oject in filtering")
                continue

            envelope = sEntity.envelope.bounds
            self.theta_x += envelope[2] - envelope[0]
            self.theta_y += envelope[3] - envelope[1]

        if(len(self.geometries) != 0):
            self.theta_x /= len(self.geometries)
            self.theta_y /= len(self.geometries)
            # print("\nDimensions of Equigrid", self.theta_x,"and", self.theta_y)
        else:
            print("Error in setThetas(), division by zero")

    def indexSource(self) :
        geometryId = 0
        for sEntity in self.geometries:
            self.addToIndex(geometryId, sEntity.bounds)
            geometryId += 1

    def addToIndex(self, geometryId, envelope) :
        maxX = math.ceil(envelope[2] / self.theta_x)
        maxY = math.ceil(envelope[3] / self.theta_y)
        minX = math.floor(envelope[0] / self.theta_x)
        minY = math.floor(envelope[1] / self.theta_y)

        for latIndex in range(minX, maxX):
            for longIndex in range(minY, maxY):
                self.spatial_index[latIndex][longIndex].append(geometryId)

    def _configuration(self) -> dict:
        """No configuration"""
        return {}