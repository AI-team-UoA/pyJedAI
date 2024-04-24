from pyjedai.datamodel import SpatialData, PYJEDAIFeature
from queue import PriorityQueue
from functools import reduce
from shapely import relate
from tqdm.auto import tqdm

DIMS = {
    'F': frozenset('F'),
    'T': frozenset('012'),
    '*': frozenset('F012'),
    '0': frozenset('0'),
    '1': frozenset('1'),
    '2': frozenset('2'),
    }

def pattern(pattern_string):
    return Pattern(pattern_string)

class Pattern(object):
    def __init__(self, pattern_string):
        self.pattern = tuple(pattern_string.upper())
    def __str__(self):
        return ''.join(self.pattern)
    def __repr__(self):
        return "DE-9IM pattern: '%s'" % str(self)
    def matches(self, matrix_string):
        matrix = tuple(matrix_string.upper())
        def onematch(p, m):
            return m in DIMS[p]
        return bool(
            reduce(lambda x, y: x * onematch(*y), zip(self.pattern, matrix), 1)
            )

class AntiPattern(object):
    def __init__(self, anti_pattern_string):
        self.anti_pattern = tuple(anti_pattern_string.upper())
    def __str__(self):
        return '!' + ''.join(self.anti_pattern)
    def __repr__(self):
        return "DE-9IM anti-pattern: '%s'" % str(self)
    def matches(self, matrix_string):
        matrix = tuple(matrix_string.upper())
        def onematch(p, m):
            return m in DIMS[p]
        return not (
            reduce(lambda x, y: x * onematch(*y),
                   zip(self.anti_pattern, matrix),
                   1)
            )

class NOrPattern(object):
    def __init__(self, pattern_strings):
        self.patterns = [tuple(s.upper()) for s in pattern_strings]
    def __str__(self):
        return '||'.join([''.join(list(s)) for s in self.patterns])
    def __repr__(self):
        return "DE-9IM or-pattern: '%s'" % str(self)
    def matches(self, matrix_string):
        matrix = tuple(matrix_string.upper())
        def onematch(p, m):
            return m in DIMS[p]
        for pattern in self.patterns:
            val = bool(
                reduce(lambda x, y: x * onematch(*y), zip(pattern, matrix), 1))
            if val is True:
                break
        return val

# Familiar names for patterns or patterns grouped in logical expression
# ---------------------------------------------------------------------
contains = Pattern('T*****FF*')
crosses_lines = Pattern('0********')   # cross_lines is only valid for pairs of lines and/or multi-lines
crosses_1 = Pattern('T*T******')       # valid for mixed types (dim(a) < dim(b))
crosses_2 = Pattern('T*****T**')       # valid for mixed types (dim(a) > dim(b))
disjoint = Pattern('FF*FF****')
equal = Pattern('T*F**FFF*')
intersects = AntiPattern('FF*FF****')
overlaps1 = Pattern('T*T***T**')   # points and regions share an overlap pattern
overlaps2 = Pattern('1*T***T**')   # valid for lines only
touches = NOrPattern(['FT*******', 'F**T*****', 'F***T****'])
within = Pattern('T*F**F***')
covered_by = NOrPattern(['T*F**F***','*TF**F***','**FT*F***','**F*TF***'])
covers =  NOrPattern(['T*****FF*',	'*T****FF*',	'***T**FF*',	'****T*FF*'])




class RelatedGeometries :
        def __init__(self, qualifyingPairs) :
            self.pgr = 0
            self.exceptions = 0
            self.detectedLinks = 0
            self.verifiedPairs = 0
            self.qualifyingPairs = qualifyingPairs
            self.interlinkedGeometries = 0
            self.continuous_unrelated_Pairs = 0
            self.violations = 0
            self.containsD1 = []
            self.containsD2 = []
            self.coveredByD1 = []
            self.coveredByD2 = []
            self.coversD1 = []
            self.coversD2 = []
            self.crossesD1 = []
            self.crossesD2 = []
            self.equalsD1 = []
            self.equalsD2 = []
            self.intersectsD1 = []
            self.intersectsD2 = []
            self.overlapsD1 = []
            self.overlapsD2 = []
            self.touchesD1 = []
            self.touchesD2 = []
            self.withinD1 = []
            self.withinD2 = []

        def addContains(self, gId1,  gId2) :
          self.containsD1.append(gId1)
          self.containsD2.append(gId2)
        def addCoveredBy(self, gId1,  gId2):
           self.coveredByD1.append(gId1)
           self.coveredByD2.append(gId2)
        def addCovers(self, gId1,  gId2):
           self.coversD1.append(gId1)
           self.coversD2.append(gId2)
        def addCrosses(self, gId1,  gId2) :
          self.crossesD1.append(gId1)
          self.crossesD2.append(gId2)
        def addEquals(self, gId1,  gId2) :
          self.equalsD1.append(gId1)
          self.equalsD2.append(gId2)
        def addIntersects(self, gId1,  gId2) :
          self.intersectsD1.append(gId1)
          self.intersectsD2.append(gId2)
        def addOverlaps(self, gId1,  gId2) :
          self.overlapsD1.append(gId1)
          self.overlapsD2.append(gId2)
        def addTouches(self, gId1,  gId2) :
          self.touchesD1.append(gId1)
          self.touchesD2.append(gId2)
        def addWithin(self, gId1,  gId2) :
          self.withinD1.append(gId1)
          self.withinD2.append(gId2)

        def  getInterlinkedPairs(self) :
            return self.interlinkedGeometries
        def  getNoOfContains(self) :
            return len(self.containsD1)
        def  getNoOfCoveredBy(self) :
            return len(self.coveredByD1)
        def  getNoOfCovers(self) :
            return len(self.coversD1)
        def  getNoOfCrosses(self) :
            return len(self.crossesD1)
        def  getNoOfEquals(self) :
            return len(self.equalsD1)
        def  getNoOfIntersects(self) :
            return len(self.intersectsD1)
        def  getNoOfOverlaps(self) :
            return len(self.overlapsD1)
        def  getNoOfTouches(self) :
            return len(self.touchesD1)
        def  getNoOfWithin(self) :
            return len(self.withinD1)
        def  getVerifiedPairs(self) :
            return self.verifiedPairs

        def print(self) :
            print("Qualifying pairs:\t", str(self.qualifyingPairs))
            print("Exceptions:\t", str(self.exceptions))
            print("Detected Links:\t", str(self.detectedLinks))
            print("Interlinked geometries:\t", str(self.interlinkedGeometries))
            print("No of contains:\t", str(self.getNoOfContains()))
            print("No of covered-by:\t" + str(self.getNoOfCoveredBy()))
            print("No of covers:\t", str(self.getNoOfCovers()))
            print("No of crosses:\t", str(self.getNoOfCrosses()))
            print("No of equals:\t", str(self.getNoOfEquals()))
            print("No of intersects:\t" + str(self.getNoOfIntersects()))
            print("No of overlaps:\t", str(self.getNoOfOverlaps()))
            print("No of touches:\t", str(self.getNoOfTouches()))
            print("No of within:\t", str(self.getNoOfWithin()))

        def  verifyRelations(self, geomId1,  geomId2,  sourceGeom,  targetGeom) :
            related = False
            array = relate(sourceGeom, targetGeom)
            self.verifiedPairs += 1

            if intersects.matches(array):
                related = True
                self.detectedLinks += 1
                self.addIntersects(geomId1, geomId2)
            if within.matches(array):
                related = True
                self.detectedLinks += 1
                self.addWithin(geomId1, geomId2)
            if covered_by.matches(array):
                related = True
                self.detectedLinks += 1
                self.addCoveredBy(geomId1, geomId2)
            if crosses_lines.matches(array) or crosses_1.matches(array) or crosses_2.matches(array):
                related = True
                self.detectedLinks += 1
                self.addCrosses(geomId1, geomId2)
            if overlaps1.matches(array) or overlaps2.matches(array):
                related = True
                self.detectedLinks += 1
                self.addOverlaps(geomId1, geomId2)
            if  equal.matches(array):
                related = True
                self.detectedLinks += 1
                self.addEquals(geomId1, geomId2)
            if  touches.matches(array):
                related = True
                self.detectedLinks += 1
                self.addTouches(geomId1, geomId2)
            if  contains.matches(array):
                related = True
                self.detectedLinks += 1
                self.addContains(geomId1, geomId2)
            if covers.matches(array):
                related = True
                self.detectedLinks += 1
                self.addCovers(geomId1, geomId2)

            if (related) :
                self.interlinkedGeometries += 1
                self.pgr += self.interlinkedGeometries
                self.continuous_unrelated_Pairs = 0
            else:
                self.continuous_unrelated_Pairs += 1


            return related





class AbstractSpatialVerification(PYJEDAIFeature):
    """Abstract class for the block building method
    """
    def __init__(self, q_pairs):
        super().__init__()
        self.relations = RelatedGeometries(q_pairs)
        return
    
    def evaluate(self,
                 verbose: bool = True,
                ) -> any:
        if self.relations.qualifyingPairs != 0:
            self.recall_score = self.relations.interlinkedGeometries / float(self.relations.qualifyingPairs)
            self.precision_score = self.relations.interlinkedGeometries / self.relations.verifiedPairs
            self.f1_score = (2 * self.precision_score * self.recall_score) / (self.precision_score + self.recall_score)
            self.progressive_geometry_recall = self.relations.pgr / self.relations.qualifyingPairs / self.relations.verifiedPairs

            if(verbose):
                print(u'\u2500' * 123)
                print("Performance:\n\tPrecision: {:9.2f}% \n\tRecall:    {:9.2f}%\n\tF1-score:  {:9.2f}%\n\tProgressive Geometry Recall: {:9.2f}%".format(self.precision_score, self.recall_score, self.f1_score, self.progressive_geometry_recall))
                print(u'\u2500' * 123)

        return{"Precision:":self.precision_score,
               "Recall:":self.recall_score,
               "F1-score:":self.f1_score,
               "Progressivr geometry recall:":self.progressive_geometry_recall
              }

    def stats(self) -> None:
        self.relations.print()
        return

class StandardSpatialVerification(AbstractSpatialVerification):
    def __init__(self, q_pairs:int):
        super().__init__(q_pairs)
        self.q_pairs = q_pairs
        return

    def process(self, spatial_data:SpatialData, priority_pairs:PriorityQueue, verbose=True):
        counter = 0
        while(not priority_pairs.empty()):
            counter += 1
            weight, source_id, target_id, tEntity = priority_pairs.get()
            self.relations.verifyRelations(source_id, target_id, spatial_data.source_geometries[source_id], tEntity)

        # self.evaluate(verbose=True)

    
    def evaluate(self, verbose: bool = True) -> any:
        return super().evaluate(verbose)

    def _configuration(self) -> dict:
        """No configuration"""
        return {
            "Q-pairs": self.q_pairs
        }