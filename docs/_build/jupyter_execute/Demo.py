#!/usr/bin/env python
# coding: utf-8

# # Demo
# 
# 
# ----
# 
# 
# 
# In this notebook we present the pyJedAI approach. pyJedAI is a an end-to-end and an upcoming python framework for Entity Resolution that will be a manual of the Entity Resolution. Its usages will outperform other state-of-the-art ER frameworks as it's easy-to-use and highly optimized as it is consisted from other established python libraries (i.e pandas, networkX, ..).

# # Instalation
# 
# pyJedAI is an open-source library that can be installed from PyPI.
# 
# For more: [pypi.org/project/pyjedai/](https://pypi.org/project/pyjedai/)

# In[5]:


get_ipython().system('pip install pyjedai -U')


# In[6]:


get_ipython().system('pip show pyjedai')


# # Reading the dataset - Clean-Clean ER example
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
# - Stores a hidden mapping of the ids, and creates if it is not exist.
# 

# In[7]:


import pandas as pd

from pyjedai.datamodel import Data

d1 = pd.read_csv("./../data/ccer/D2/abt.csv", sep='|', engine='python', na_filter=False).astype(str)
d2 = pd.read_csv("./../data/ccer/D2/buy.csv", sep='|', engine='python', na_filter=False).astype(str)
gt = pd.read_csv("./../data/ccer/D2/gt.csv", sep='|', engine='python')

data = Data(
    dataset_1=d1,
    attributes_1=['id','name','description'],
    id_column_name_1='id',
    dataset_2=d2,
    attributes_2=['id','name','description'],
    id_column_name_2='id',
    ground_truth=gt,
)


# In[8]:


data.print_specs()


# In[9]:


data.dataset_1.head(2)


# In[10]:


data.dataset_2.head(2)


# In[11]:


data.ground_truth.head(2)


# # Creating workflow using pyJedAI methods
# 
# Multiple algorithms, techniques and features have already been implemented. This way, we can import the method and proceed to the workflow architecture.
# 
# For example we demostrate a variety of algorithms in each step, as it is shown in the bellow cell.

# In[12]:


from pyjedai.workflow import WorkFlow, compare_workflows
from pyjedai.block_building import (
    StandardBlocking, QGramsBlocking, ExtendedQGramsBlocking, 
    SuffixArraysBlocking, ExtendedSuffixArraysBlocking
)
from pyjedai.block_cleaning import BlockFiltering, BlockPurging
from pyjedai.comparison_cleaning import (
    WeightedEdgePruning, WeightedNodePruning, CardinalityEdgePruning, 
    CardinalityNodePruning, BLAST, ReciprocalCardinalityNodePruning, 
    ReciprocalWeightedNodePruning, ComparisonPropagation
)
from pyjedai.matching import EntityMatching
from pyjedai.clustering import ConnectedComponentsClustering


# ## Building a simple WorkFlow
# 
# The main workflow that pyjedai supports, consists of 8 steps:
# 
# - __Data Reading:__ Raw data in the pandas\<DataFrame\> format are transformed to pyjedai\<Data\>
# - __Block Building__: In this step we create blocks of entities based on some tokenization techniques like QGrams, SuffixArray, etc.
# - __Block Filtering, Block Purging, Cardnality Edge Pruning__: These methods reduce the amount of comparisons by removing entities from the blocks or producing a much more dence index.    
# - __Entity Matching__: The similarity checking phase. Each block is now tested for similarity. Meaning that all entities contained in the block will be mesaured for similarity using a metric like Jaccard.
# - __Connected Components Clustering__: Creates clusters of similarity based on the graph produced from the entity matching step.
# - __Results__: Finally, pyjedai produces a set of pairs that apper to be duplicates, scores as well as visualizations that help users understand the workflow performance. 
# 
# For this demo, we created a simple architecture as we see bellow:

# 
# ![workflow-example.png](https://github.com/AI-team-UoA/pyJedAI/blob/main/documentation/workflow-example.png?raw=true)

# In[13]:


w = WorkFlow(
    block_building = dict(
        method=QGramsBlocking, 
        params=dict(qgrams=3)
    ),
    block_cleaning = [
        dict(
            method=BlockFiltering, 
            params=dict(ratio=0.8)
        ),
        dict(
            method=BlockPurging,
            params=dict(smoothing_factor=1.025)
        )
    ],
    comparison_cleaning = dict(method=CardinalityEdgePruning),
    entity_matching = dict(
        method=EntityMatching, 
        metric='sorensen_dice',
        similarity_threshold=0.5,
        attributes = ['description', 'name']
    ),
    clustering = dict(method=ConnectedComponentsClustering),
    name="Worflow-QGramsBlocking"
)


# # Evaluation and detailed reporting

# In[14]:


w.run(data, workflow_tqdm_enable=True, verbose=False)


# In[15]:


w.to_df()


# # Visualization

# In[16]:


w.visualize()


# In[17]:


w.visualize(separate=True)


# # Multiple workflows
# 
# pyJedAI provides methods for comparing multiple workflows. For example, we can test the above example with all the Block Building methods provided.
# 

# In[18]:


block_building_methods = [StandardBlocking, QGramsBlocking, ExtendedQGramsBlocking, SuffixArraysBlocking, ExtendedSuffixArraysBlocking]
workflows = []
for bbm in block_building_methods:
    workflows.append(WorkFlow(
        block_building = dict(
            method=bbm, 
        ),
        block_cleaning = [
            dict(
                method=BlockFiltering, 
                params=dict(ratio=0.8)
            ),
            dict(
                method=BlockPurging,
                params=dict(smoothing_factor=1.025)
            )
        ],
        comparison_cleaning = dict(method=CardinalityEdgePruning),
        entity_matching = dict(
            method=EntityMatching,
            metric='sorensen_dice',
            similarity_threshold=0.5,
            attributes = ['description', 'name']
        ),
        clustering = dict(method=ConnectedComponentsClustering),
        name="Workflow-"+str(bbm.__name__)
    ))
    workflows[-1].run(data, workflow_tqdm_enable=True)


# In[19]:


compare_workflows(workflows, with_visualization=True)


# <hr>
# <div align="right">
# K. Nikoletos, G. Papadakis & M. Koubarakis
# </div>
# <div align="right">
# <a href="https://github.com/Nikoletos-K/pyJedAI/blob/main/LICENSE">Apache License 2.0</a>
# </div>
