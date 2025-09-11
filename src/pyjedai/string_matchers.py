"""
This code was based on py_stringmatching: https://github.com/anhaidgroup/py_stringmatching
"""
from abc import ABC, abstractmethod
from stringcompare import Jaro
import re


class WhitespaceTokenizer(ABC):
    def tokenize(self, sentence):
        whitespace_pattern = re.compile(r'\s+')
        tokens = whitespace_pattern.split(sentence.strip())
        tokens = [token for token in tokens if token]
        return tokens
    
class StringMatcher(ABC):
    """String Matchers based on py_stringmatching"""
    flag = True
    
    
    def check_instance_type(self, te1, te2) -> None:
        if not isinstance(te1, list) and not isinstance(te1, set): 
            raise TypeError("Must be either list or set")
        if not isinstance(te2, list) and not isinstance(te2, set): 
            raise TypeError("Must be either list or set")
        
    def exact_match(self, te1, te2):
        return te1 == te2
    
    def empty_match(self, te1, te2):
        return len(te1) == 0 or len(te2) == 0
            
        
    @abstractmethod
    def compare(self, te1, te2):
        pass


class Cosine(StringMatcher):
    def compare(self, te1, te2) -> float:

        self.check_instance_type(te1, te2)

        # if exact match return 1.0
        if self.exact_match(te1, te2):
            return 1.0

        # if one of the strings is empty return 0
        if self.empty_match(te1, te2):
            return 0.0

        intersection = len(set(te1) & set(te2))
        norm1 = len(te1) ** 0.5
        norm2 = len(te2) ** 0.5
        return intersection / (norm1 * norm2) if norm1 * norm2 > 0 else 0
        
class Dice(StringMatcher):
    def compare(self, te1, te2) -> float:
        print("TIFASIII??")
        self.check_instance_type(te1, te2)

        set1 = set(te1)
        set2 = set(te2)

        # if exact match return 1.0
        if self.exact_match(set1, set2):
            return 1.0

        # if one of the strings is empty return 0
        if self.empty_match(set1, set2):
            return 0.0

        return 2.0 * float(len(set1 & set2)) / float(len(set1) + len(set2))

class Jaccard(StringMatcher):
    def compare(self, te1, te2) -> float:
        self.check_instance_type(te1, te2)

        set1 = set(te1)
        set2 = set(te2)

        # if exact match return 1.0
        if self.exact_match(set1, set2):
            return 1.0

        # if one of the strings is empty return 0
        if self.empty_match(set1, set2):
            return 0.0

        intersection = len(set(te1) & set(te2))
        return intersection/(len(set1) + len(set2) + intersection)

class GeneralizedJaccard(StringMatcher):
    def compare(self, te1, te2) -> float:
        self.check_instance_type(te1, te2)

        set1 = set(te1)
        set2 = set(te2)

        # if exact match return 1.0
        if self.exact_match(set1, set2):
            return 1.0

        # if one of the strings is empty return 0
        if self.empty_match(set1, set2):
            return 0.0

        set1_x = set()
        set2_y = set()
        match_score = 0.0
        match_count = 0
        list_matches = []
        threshold=0.5
        for element in set1:
            for item in set2:
                score = Jaro().compare(element, item)
                if score > 1 or score < 0:
                    raise ValueError('Similarity measure should' + \
                                    ' return value in the range [0,1]')
                if score > threshold:
                    list_matches.append((element, item, score))

        # position of first string, second string and sim score in tuple
        first_string_pos = 0
        second_string_pos = 1
        sim_score_pos = 2

        # sort the score of all the pairs
        list_matches.sort(key=lambda x: x[sim_score_pos], reverse=True)

        # select score in increasing order of their weightage, 
        # do not reselect the same element from either set.
        for element in list_matches:
            if (element[first_string_pos] not in set1_x and
                element[second_string_pos] not in set2_y):
                set1_x.add(element[first_string_pos])
                set2_y.add(element[second_string_pos])
                match_score += element[sim_score_pos]
                match_count += 1

        return float(match_score) / float(len(set1) + len(set2) - match_count)

class OverlapCoefficient(StringMatcher):
    def compare(self, te1, te2) -> float:
        self.check_instance_type(te1, te2)

        set1 = set(te1)
        set2 = set(te2)

        # if exact match return 1.0
        if self.exact_match(set1, set2):
            return 1.0

        # if one of the strings is empty return 0
        if self.empty_match(set1, set2):
            return 0.0
        return float(len(set1 & set2)) / min(len(set1), len(set2))

