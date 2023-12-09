import math

from pyjedai.datamodel import SpatialData, PYJEDAIFeature
from queue import PriorityQueue
from tqdm.auto import tqdm


class AbstractSpatialInitialization(PYJEDAIFeature):
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

class StandardSpatialInitialization(AbstractSpatialInitialization):
    def __init__(self,  budget:int, wScheme:str):
        super().__init__()
        self.wScheme = wScheme
        self.budget = budget

    def process(self, spatial_data:SpatialData, spatial_index:list, theta_x:float, theta_y:float): # reads target geometries on the fly
        self.source_geometries = spatial_data.source_geometries
        self.targetGeometries = spatial_data.targetGeometries
        self.spatial_index = spatial_index
        self.theta_x = theta_x
        self.theta_y = theta_y

        self.flag = [-1] * spatial_data.source_geometries_size
        self.freq = [-1] * spatial_data.source_geometries_size
        self.topKPairs = PriorityQueue(maxsize = self.budget + 1)

        minimumWeight = -1
        targetId = 0

        # TODO itile spatial.size2
        
        for targetGeom in self.targetGeometries:
            candidates = self.getCandidates(targetId, targetGeom)
            for candidateMatchId in candidates:
                if (self.validCandidate(candidateMatchId, targetGeom.envelope)):
                    weight = self.getWeight(candidateMatchId, targetGeom)
                    if (minimumWeight <= weight):
                        self.topKPairs.put((weight, candidateMatchId, targetId, targetGeom))
                        if (self.budget < self.topKPairs.qsize()):
                            minimumWeight = self.topKPairs.get()[0]
            targetId += 1

        # print("Total target geometries", targetId)
        return(self.topKPairs)

    def getCandidates(self, targetId, tEntity):
        candidates = set()

        envelope = tEntity.envelope.bounds
        maxX = math.ceil(envelope[2] / self.theta_x)
        maxY = math.ceil(envelope[3] / self.theta_y)
        minX = math.floor(envelope[0] / self.theta_x)
        minY = math.floor(envelope[1] / self.theta_y)

        for latIndex in range(minX, maxX):
            for longIndex in range(minY,maxY):
                for sourceId in self.spatial_index[latIndex][longIndex]:
                    if (self.flag[sourceId] == -1):
                        self.flag[sourceId] = targetId
                        self.freq[sourceId] = 0
                    self.freq[sourceId] += 1
                    candidates.add(sourceId)

        return candidates

    def validCandidate(self, candidateId, targetEnv):
        return self.source_geometries[candidateId].envelope.intersects(targetEnv)

    def getWeight(self,sourceId, tEntity) :
        commonBlocks = self.freq[sourceId]
        if self.wScheme == 'CF':
            return commonBlocks
        elif self.wScheme == 'JS_APPROX':
            return commonBlocks / (self.getNoOfBlocks(self.source_geometries[sourceId].envelope) + self.getNoOfBlocks(tEntity.envelope) - commonBlocks)
        elif self.wScheme == 'MBR':
            srcEnv = self.source_geometries[sourceId].envelope
            trgEnv = tEntity.envelope
            mbrIntersection = srcEnv.intersection(trgEnv)
            denominator = srcEnv.area + trgEnv.area - mbrIntersection.area
            if denominator == 0 :
                return 0
            return mbrIntersection.area/denominator
        return 1.0

    def getNoOfBlocks(self,envelope) :
        maxX = math.ceil(envelope[2] / self.theta_x)
        maxY = math.ceil(envelope[3] / self.theta_y)
        minX = math.floor(envelope[0] / self.theta_x)
        minY = math.floor(envelope[1] / self.theta_y)
        return (maxX - minX + 1) * (maxY - minY + 1)

    def _configuration(self) -> dict:
        """No configuration"""
        return {}