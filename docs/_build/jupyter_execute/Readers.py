#!/usr/bin/env python
# coding: utf-8

# <div align="center"> 
#     <br>
#     <br>
#     <img src="https://raw.githubusercontent.com/AI-team-UoA/pyJedAI/main/documentation/pyjedai.logo.drawio.png?raw=true" alt="drawing" width="400"/>
#     <br>
#     <hr>
#     <font size="3">pyJedAI Data Reading examples</font>
#     <br>
# </div>
# <hr>
# 
# pyJedAI needs as input a pandas.DataFrame. In this notebook we provide some examples of data reading and transformation to DataFrame.
# 
# ![reading-process.jpg](https://github.com/AI-team-UoA/pyJedAI/blob/main/documentation/reading-process.png?raw=true)

# In[1]:


import pandas as pd


# # CSV Reader
# 
# Example Dataset: CORA

# In[2]:


d1 = pd.read_csv("./data/cora/cora.csv", sep='|')
gt = pd.read_csv("./data/cora/cora_gt.csv", sep='|', header=None)


# In[ ]:


d1.head(1)


# # JSON Reader

# In[14]:


d1 = pd.read_json("./data/cora/cora.json")
gt = pd.read_json("./data/cora/cora_gt.json")


# In[15]:


d1.head(1)


# # Excel Reader

# In[7]:


d1 = pd.read_excel("./data/cora/cora.xlsx")
gt = pd.read_excel("./data/cora/cora_gt.xlsx")


# In[8]:


d1.head(1)


# # RDF/OWL Reader

# In[87]:


import rdfpandas as rfd
import rdflib

rdfd1 = rdflib.Graph().parse('./data/rdf/restaurants/restaurant1.nt')
rdfd2 = rdflib.Graph().parse('./data/rdf/restaurants/restaurant2.nt')

def rdf_to_df(graph_parsed) -> pd.DataFrame:
    subject = []
    predicate = []
    rdfobject = []
    df = pd.DataFrame(columns=['subject', 'predicate', 'object'])
    for s, p, o in graph_parsed:
        subject.append(s)
        predicate.append(p)
        rdfobject.append(o)
    df['predicate'] = predicate
    df['subject'] = subject
    df['object'] = rdfobject

    return df
    
d1 = rdf_to_df(rdfd1)
d2 = rdf_to_df(rdfd2)


# In[34]:


d1.head(2)


# In[35]:


d2.head(2)


# # Relational DBs Reader

# In[97]:


from sqlite3 import connect
conn = connect(':memory:')
d1.to_sql('d1', conn)
d2.to_sql('d2', conn)
gt.to_sql('gt', conn)
sql_d1 = pd.read_sql('SELECT * FROM d1', conn)
sql_d2 = pd.read_sql('SELECT * FROM d2', conn)
sql_gt = pd.read_sql('SELECT * FROM gt', conn)


# In[98]:


sql_d1.head(1)


# ### PostgreSQL

# In[ ]:


from sqlalchemy import create_engine

POSTGRES_ADDRESS = 'db' ## INSERT YOUR DB ADDRESS
POSTGRES_PORT = '5439'
POSTGRES_USERNAME = 'username' ## CHANGE THIS TO YOUR POSTGRES USERNAME
POSTGRES_PASSWORD = 'root' ## CHANGE THIS TO YOUR POSTGRES PASSWORD
POSTGRES_DBNAME = 'database' ## CHANGE THIS TO YOUR DATABASE NAME
postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(
    username=POSTGRES_USERNAME,
    password=POSTGRES_PASSWORD,
    ipaddress=POSTGRES_ADDRESS,
    port=POSTGRES_PORT,
    dbname=POSTGRES_DBNAME
))

# Create the connection
cnx = create_engine(postgres_str)


# In[ ]:


pd.read_sql('SELECT * FROM d1', cnx)


# # SPARKQL Reader
# 

# In[16]:


from pandas import json_normalize
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery("""
        SELECT *
        WHERE
        {
          ?athlete  rdfs:label      "Cristiano Ronaldo"@en ;
                    dbo:birthPlace  ?place .
         ?place     a               dbo:City ;
                    rdfs:label      ?cityName .
        }
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
d1 = json_normalize(results["results"]["bindings"])


# In[17]:


d1

