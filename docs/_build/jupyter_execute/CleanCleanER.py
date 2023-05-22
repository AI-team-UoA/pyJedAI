#!/usr/bin/env python
# coding: utf-8

# # Clean-Clean ER
# 
# ---
# 
# In this notebook we present the pyJedAI approach in the well-known ABT-BUY dataset. Clean-Clean ER in the link discovery/deduplication between two sets of entities.

# Dataset: __Abt-Buy dataset__
# 
# The Abt-Buy dataset for entity resolution derives from the online retailers Abt.com and Buy.com. The dataset contains 1076 entities from abt.com and 1076 entities from buy.com as well as a gold standard (perfect mapping) with 1076 matching record pairs between the two data sources. The common attributes between the two data sources are: product name, product description and product price.

# # Instalation
# 
# pyJedAI is an open-source library that can be installed from PyPI.
# 
# For more: [pypi.org/project/pyjedai/](https://pypi.org/project/pyjedai/)

# In[ ]:


get_ipython().system('pip install pyjedai -U')


# In[2]:


get_ipython().system('pip show pyjedai')


# Imports

# In[3]:


import os
import sys
import pandas as pd
import networkx
from networkx import draw, Graph


# In[4]:


import pyjedai
from pyjedai.utils import (
    text_cleaning_method,
    print_clusters,
    print_blocks,
    print_candidate_pairs
)
from pyjedai.evaluation import Evaluation, write


# # Workflow Architecture
# 
# ![workflow-example.png](https://github.com/AI-team-UoA/pyJedAI/blob/main/documentation/workflow-example.png?raw=true)

# # Data Reading
# 
# pyJedAI in order to perfrom needs only the tranformation of the initial data into a pandas DataFrame. Hence, pyJedAI can function in every structured or semi-structured data. In this case Abt-Buy dataset is provided as .csv files. 
# 

# In[5]:


from pyjedai.datamodel import Data
from pyjedai.evaluation import Evaluation


# In[8]:


d1 = pd.read_csv("./../data/ccer/D2/abt.csv", sep='|', engine='python', na_filter=False).astype(str)
d2 = pd.read_csv("./../data/ccer/D2/buy.csv", sep='|', engine='python', na_filter=False).astype(str)
gt = pd.read_csv("./../data/ccer/D2/gt.csv", sep='|', engine='python').astype(str)

data = Data(dataset_1=d1,
            id_column_name_1='id',
            dataset_2=d2,
            id_column_name_2='id',
            ground_truth=gt)


# pyJedAI offers also dataset analysis methods (more will be developed)

# In[9]:


data.print_specs()


# In[10]:


data.dataset_1.head(5)


# In[11]:


data.dataset_2.head(5)


# In[12]:


data.ground_truth.head(3)


# # Block Building
# 
# It clusters entities into overlapping blocks in a lazy manner that relies on unsupervised blocking keys: every token in an attribute value forms a key. Blocks are then extracted, possibly using a transformation, based on its equality or on its similarity with other keys.
# 
# The following methods are currently supported:
# 
# - Standard/Token Blocking
# - Sorted Neighborhood
# - Extended Sorted Neighborhood
# - Q-Grams Blocking
# - Extended Q-Grams Blocking
# - Suffix Arrays Blocking
# - Extended Suffix Arrays Blocking

# In[13]:


from pyjedai.block_building import (
    StandardBlocking,
    QGramsBlocking,
    ExtendedQGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking,
)


# In[14]:


qgb = SuffixArraysBlocking()
blocks = qgb.build_blocks(data, attributes_1=['name'], attributes_2=['name'])


# In[15]:


qgb.report()


# In[16]:


_ = qgb.evaluate(blocks, with_classification_report=True)


# # Block Cleaning
# 
# ___Optional step___
# 
# Its goal is to clean a set of overlapping blocks from unnecessary comparisons, which can be either redundant (i.e., repeated comparisons that have already been executed in a previously examined block) or superfluous (i.e., comparisons that involve non-matching entities). Its methods operate on the coarse level of individual blocks or entities.

# In[17]:


from pyjedai.block_cleaning import BlockFiltering


# In[18]:


bf = BlockFiltering(ratio=0.8)
filtered_blocks = bf.process(blocks, data, tqdm_disable=False)


# In[19]:


_ = bf.evaluate(filtered_blocks)


# # Comparison Cleaning
# 
# ___Optional step___
# 
# Similar to Block Cleaning, this step aims to clean a set of blocks from both redundant and superfluous comparisons. Unlike Block Cleaning, its methods operate on the finer granularity of individual comparisons.
# 
# The following methods are currently supported:
# 
# - Comparison Propagation
# - Cardinality Edge Pruning (CEP)
# - Cardinality Node Pruning (CNP)
# - Weighed Edge Pruning (WEP)
# - Weighed Node Pruning (WNP)
# - Reciprocal Cardinality Node Pruning (ReCNP)
# - Reciprocal Weighed Node Pruning (ReWNP)
# - BLAST
# 
# Most of these methods are Meta-blocking techniques. All methods are optional, but competive, in the sense that only one of them can part of an ER workflow. For more details on the functionality of these methods, see here. They can be combined with one of the following weighting schemes:
# 
# - Aggregate Reciprocal Comparisons Scheme (ARCS)
# - Common Blocks Scheme (CBS)
# - Enhanced Common Blocks Scheme (ECBS)
# - Jaccard Scheme (JS)
# - Enhanced Jaccard Scheme (EJS)

# In[20]:


from pyjedai.block_cleaning import BlockPurging


# In[21]:


cbbp = BlockPurging()
cleaned_blocks = cbbp.process(filtered_blocks, data, tqdm_disable=False)


# In[22]:


cbbp.report()


# In[24]:


_ = cbbp.evaluate(cleaned_blocks)


# ## Meta Blocking

# In[28]:


from pyjedai.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST,
    ReciprocalCardinalityNodePruning,
    ReciprocalWeightedNodePruning,
    ComparisonPropagation
)


# In[29]:


mb = CardinalityEdgePruning(weighting_scheme='X2')
candidate_pairs_blocks = mb.process(filtered_blocks, data, tqdm_disable=True)


# In[31]:


_ = mb.evaluate(candidate_pairs_blocks)


# # Entity Matching
# 
# It compares pairs of entity profiles, associating every pair with a similarity in [0,1]. Its output comprises the similarity graph, i.e., an undirected, weighted graph where the nodes correspond to entities and the edges connect pairs of compared entities.

# In[32]:


from pyjedai.matching import EntityMatching


# In[34]:


EM = EntityMatching(
    metric='dice',
    similarity_threshold=0.5,
    attributes = ['description', 'name']
)

pairs_graph = EM.predict(candidate_pairs_blocks, data, tqdm_disable=True)


# In[35]:


draw(pairs_graph)


# In[38]:


_ = EM.evaluate(pairs_graph)


# # Entity Clustering
# 
# It takes as input the similarity graph produced by Entity Matching and partitions it into a set of equivalence clusters, with every cluster corresponding to a distinct real-world object.

# In[39]:


from pyjedai.clustering import ConnectedComponentsClustering


# In[41]:


ccc = ConnectedComponentsClustering()
clusters = ccc.process(pairs_graph, data)


# In[42]:


ccc.report()


# In[45]:


_ = ccc.evaluate(clusters)


# <hr>
# <div align="right">
# K. Nikoletos, G. Papadakis & M. Koubarakis
# </div>
# <div align="right">
# <a href="https://github.com/Nikoletos-K/pyJedAI/blob/main/LICENSE">Apache License 2.0</a>
# </div>
