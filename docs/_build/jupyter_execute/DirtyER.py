#!/usr/bin/env python
# coding: utf-8

# # Dirty ER
# 
# ---
# 
# In this notebook we present the pyJedAI approach in the well-known ABT-BUY dataset. Dirty ER, is the process of dedeplication of one set.

# # Instalation
# 
# pyJedAI is an open-source library that can be installed from PyPI.
# 
# For more: [pypi.org/project/pyjedai/](https://pypi.org/project/pyjedai/)

# In[ ]:


get_ipython().system('python --version')


# In[ ]:


get_ipython().system('pip install pyjedai -U')


# In[2]:


get_ipython().system('pip show pyjedai')


# Imports

# In[5]:


import os
import sys
import pandas as pd
import networkx
from networkx import draw, Graph

from pyjedai.utils import print_clusters, print_blocks, print_candidate_pairs
from pyjedai.evaluation import Evaluation


# # Reading the dataset - Dirty ER example
# 
# pyJedAI in order to perfrom needs only the tranformation of the initial data into a pandas DataFrame. Hence, pyJedAI can function in every structured or semi-structured data. In this case Abt-Buy dataset is provided as .csv files. 
# 
# <div align="center">
#     <img align="center" src="https://github.com/AI-team-UoA/pyJedAI/blob/main/documentation/reading-process.png?raw=true" width="800" />
# </div>
# 
# 
# ## pyjedai <Data\> module
# 
# Data module offers a numpber of options
# - Selecting the parameters (columns) of the dataframe, in D1 (and in D2)
# - Prints a detailed text analysis
# - Stores a hidden mapping of the ids, and creates it if not exists.

# In[7]:


from pyjedai.datamodel import Data

d1 = pd.read_csv("./../data/der/cora/cora.csv", sep='|')
gt = pd.read_csv("./../data/der/cora/cora_gt.csv", sep='|', header=None)
attr = ['Entity Id','author', 'title']


# Data is the connecting module of all steps of the workflow

# In[9]:


data = Data(
    dataset_1=d1,
    id_column_name_1='Entity Id',
    ground_truth=gt,
    attributes_1=attr
)


# # Workflow with Block Cleaning Methods
# 
# In this notebook we created the bellow architecture:
# 
# ![workflow1-cora.png](https://github.com/AI-team-UoA/pyJedAI/blob/main/documentation/workflow1-cora.png?raw=true)
# 
# 

# ## Block Building
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

# In[10]:


from pyjedai.block_building import (
    StandardBlocking,
    QGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking,
    ExtendedQGramsBlocking
)


# In[11]:


bb = SuffixArraysBlocking(suffix_length=2)
blocks = bb.build_blocks(data)


# In[13]:


_ = bb.evaluate(blocks)


# ## Block Cleaning
# 
# ___Optional step___
# 
# Its goal is to clean a set of overlapping blocks from unnecessary comparisons, which can be either redundant (i.e., repeated comparisons that have already been executed in a previously examined block) or superfluous (i.e., comparisons that involve non-matching entities). Its methods operate on the coarse level of individual blocks or entities.

# In[14]:


from pyjedai.block_cleaning import BlockFiltering


# In[16]:


bc = BlockFiltering(ratio=0.9)
blocks = bc.process(blocks, data)


# In[18]:


_ = bc.evaluate(blocks)


# ## Comparison Cleaning
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


bp = BlockPurging(smoothing_factor=0.008)
blocks = bp.process(blocks, data)


# In[23]:


_ = bp.evaluate(blocks)


# ### Meta Blocking

# In[24]:


from pyjedai.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST,
    ReciprocalCardinalityNodePruning,
    ComparisonPropagation
)


# In[25]:


mb = WeightedEdgePruning(weighting_scheme='CBS')
blocks = mb.process(blocks, data)


# In[28]:


_ = mb.evaluate(blocks)


# ## Entity Matching
# 
# It compares pairs of entity profiles, associating every pair with a similarity in [0,1]. Its output comprises the similarity graph, i.e., an undirected, weighted graph where the nodes correspond to entities and the edges connect pairs of compared entities.

# In[29]:


from pyjedai.matching import EntityMatching


# In[31]:


attr = {
    'author' : 0.6, 
    'title' : 0.4
}

EM = EntityMatching(
    metric='jaccard',
    similarity_threshold=0.5
)

pairs_graph = EM.predict(blocks, data)


# In[32]:


draw(pairs_graph)


# In[34]:


_ = EM.evaluate(pairs_graph)


# ## Entity Clustering
# 
# It takes as input the similarity graph produced by Entity Matching and partitions it into a set of equivalence clusters, with every cluster corresponding to a distinct real-world object.

# In[39]:


from pyjedai.clustering import ConnectedComponentsClustering


# In[40]:


ec = ConnectedComponentsClustering()
clusters = ec.process(pairs_graph, data)


# In[43]:


_ = ec.evaluate(clusters)


# # Workflow with Similarity Joins
# 
# In this notebook we created the bellow archtecture:
# 
# ![workflow2-cora.png](https://github.com/AI-team-UoA/pyJedAI/blob/main/documentation/workflow2-cora.png?raw=true)
# 
# 

# ## Data Reading

# Data is the connecting module of all steps of the workflow

# In[44]:


from pyjedai.datamodel import Data
d1 = pd.read_csv("./../data/der/cora/cora.csv", sep='|')
gt = pd.read_csv("./../data/der/cora/cora_gt.csv", sep='|', header=None)
attr = ['Entity Id','author', 'title']
data = Data(
    dataset_1=d1,
    id_column_name_1='Entity Id',
    ground_truth=gt,
    attributes_1=attr
)


# ## Similarity Joins

# In[45]:


from pyjedai.joins import ΕJoin, TopKJoin


# In[47]:


join = ΕJoin(similarity_threshold = 0.5,
             metric = 'jaccard',
             tokenization = 'qgrams_multiset',
             qgrams = 2)

g = join.fit(data)


# In[48]:


_ = join.evaluate(g)


# In[49]:


topk_join = TopKJoin(K=20,
             metric = 'jaccard',
             tokenization = 'qgrams',
             qgrams = 3)

g = topk_join.fit(data)


# In[50]:


draw(g)


# In[51]:


topk_join.evaluate(g)


# ## Entity Clustering

# In[52]:


from pyjedai.clustering import ConnectedComponentsClustering


# In[54]:


ccc = ConnectedComponentsClustering()

clusters = ccc.process(g, data)


# In[56]:


_ = ccc.evaluate(clusters)


# <hr>
# <div align="right">
# K. Nikoletos, G. Papadakis & M. Koubarakis
# </div>
# <div align="right">
# <a href="https://github.com/Nikoletos-K/pyJedAI/blob/main/LICENSE">Apache License 2.0</a>
# </div>
