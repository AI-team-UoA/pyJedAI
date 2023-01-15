#!/usr/bin/env python
# coding: utf-8

# <table>
#     <tr>
#         <td>  
#             <img  src="https://raw.githubusercontent.com/AI-team-UoA/pyJedAI/main/documentation/pyjedai.logo.drawio.png?raw=true "style="width: 300px;padding: 40px"/>
#         </td>
#         <td> 
#             <img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" style="width: 250px;"/>
#          </td>
#     </tr>
# </table>
# 
# <div align="center"> 
#     <hr>
#     <font size="3">Hyper-Parameter Tuning with Optuna Tutorial</font>
# </div>
# <hr>
# 
# Optimization and fine-tuning for the hyper-parameters using a novel framework named Optuna.

# # Instalation
# 
# pyJedAI is an open-source library that can be installed from PyPI.
# 
# For more: [pypi.org/project/pyjedai/](https://pypi.org/project/pyjedai/)

# In[1]:


get_ipython().system('pip install pyjedai -U')


# In[2]:


get_ipython().system('pip show pyjedai')


# Imports

# In[3]:


import plotly.express as px
import logging
import sys
import optuna
import plotly
import os
import sys
import pandas as pd
from optuna.visualization import *
import plotly.io as pio
import plotly.express as px
pio.templates.default = "plotly_white"


# ## Data Reading

# In[4]:


from pyjedai.datamodel import Data

data = Data(
    dataset_1=pd.read_csv("./../data/D2/abt.csv", sep='|', engine='python', na_filter=False).astype(str),
    attributes_1=['id','name','description'],
    id_column_name_1='id',
    dataset_2=pd.read_csv("./../data/D2/buy.csv", sep='|', engine='python', na_filter=False).astype(str),
    attributes_2=['id','name','description'],
    id_column_name_2='id',
    ground_truth=pd.read_csv("./../data/D2/gt.csv", sep='|', engine='python'),
)

data.process()


# ## WorkFlow

# In[5]:


from pyjedai.workflow import WorkFlow, compare_workflows
from pyjedai.block_building import StandardBlocking, QGramsBlocking, ExtendedQGramsBlocking, SuffixArraysBlocking, ExtendedSuffixArraysBlocking
from pyjedai.block_cleaning import BlockFiltering, BlockPurging
from pyjedai.comparison_cleaning import WeightedEdgePruning, WeightedNodePruning, CardinalityEdgePruning, CardinalityNodePruning, BLAST, ReciprocalCardinalityNodePruning, ReciprocalWeightedNodePruning, ComparisonPropagation
from pyjedai.matching import EntityMatching
from pyjedai.clustering import ConnectedComponentsClustering


# In[6]:


db_name = "pyjedai"
title = "Test"
storage_name = "sqlite:///{}.db".format(db_name)
study_name = title  # Unique identifier of the study.


# ## Objective function
# 
# 
# In the bellow cell, we define which parameters we want to be fine-tuned and the boundaries that we suggest. Also we set as the goal score to be maximized the F1-Score.
# 

# In[7]:


'''
 OPTUNA objective function
'''
def objective(trial):
    
    w = WorkFlow(
        block_building = dict(
            method=QGramsBlocking, 
            params=dict(qgrams=trial.suggest_int("qgrams", 3, 10)),
            attributes_1=['name'],
            attributes_2=['name']
        ),
        block_cleaning = [
            dict(
                method=BlockPurging,
                params=dict(smoothing_factor=1.025)
            ),
            dict(
                method=BlockFiltering, 
                params=dict(
                    ratio = trial.suggest_float("ratio", 0.7, 0.95)
                )
            )
        ],
        comparison_cleaning = dict(method=CardinalityEdgePruning),
            entity_matching = dict(
            method=EntityMatching, 
            metric='sorensen_dice',
            similarity_threshold= trial.suggest_float("similarity_threshold", 0.05, 0.9),
            attributes = ['description', 'name']
        ),
        clustering = dict(method=ConnectedComponentsClustering),
        name="Worflow-Test"
    )
    w.run(data, workflow_step_tqdm_disable=True, verbose=False)
    f1, precision, recall = w.get_final_scores()
    
    return f1


# In[8]:


study_name = title  # Unique identifier of the study.
num_of_trials = 30
study = optuna.create_study(
    directions=["maximize"],
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True
)
print("Optuna trials starting")
study.optimize(
    objective, 
    n_trials=num_of_trials, 
    show_progress_bar=True
)
print("Optuna trials finished")


# # Optuna Visualizations

# In[9]:


study.trials_dataframe(attrs=("number", "value", "params", "state"))


# In[10]:


fig = plot_optimization_history(study)
fig.show()


# In[11]:


fig = plot_parallel_coordinate(study)
fig.show()


# In[12]:


fig = plot_parallel_coordinate(study, params=["qgrams"])
fig.show()


# In[13]:


fig = plot_contour(study)
fig.show()


# In[14]:


fig = plot_contour(study, params=["qgrams", "ratio"])
fig.show()


# In[15]:


fig = plot_slice(study,  params=["qgrams", "ratio"])
fig.show()


# In[16]:


fig = plot_slice(study,  params=["qgrams", "ratio"])
fig.show()


# In[17]:


fig = plot_param_importances(study)
fig.show()


# In[18]:


fig = plot_edf(study)
fig.show()


# In[19]:


fig = optuna.visualization.plot_param_importances(
    study, target=lambda t: t.duration.total_seconds(), target_name="duration"
)
fig.show()

