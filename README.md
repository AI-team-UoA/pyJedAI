<div align="center">
<br>
<img align="center" src="https://github.com/Nikoletos-K/pyJedAI/blob/main/documentation/pyjedai.logo.drawio.png?raw=true" alt="pyJedAI" width="400"/>
</div>
<br><br>
<div align="center">
An open-source library that leverages Python’s data science ecosystem to build <br> powerful end-to-end Entity Resolution workflows.
</div>

---

[![Tests](https://github.com/Nikoletos-K/pyJedAI/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Nikoletos-K/pyJedAI/actions/workflows/tests.yml)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![macOS](https://svgshare.com/i/ZjP.svg)](https://svgshare.com/i/ZjP.svg)
[![Windows](https://svgshare.com/i/ZhY.svg)](https://svgshare.com/i/ZhY.svg)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

# Overview

pyJedAI is a Python framework that aims to provide experts and non-experts with robust and fast solutions to a variety of Entity Resolution problems. This framework constitutes the sole open-source Link Discovery tool that is capable of exploiting the latest breakthroughs in Deep Learning and NLP techniques, which are publicly available through the Python data science ecosystem. This applies to both blocking and matching, thus ensuring high time efficiency, high scalability as well as high effectiveness, without requiring any labelled instances from the user.

### Key-Features

- Input data-type independent. Both structured and semi-structured data can be processed.
- Various implemented algorithms.
- Easy-to-use.
- Utilizes some of the famous and cutting-edge machine learning packages.
- Offers Supervised and un-supervised ML techniques.

__Open demos are available in:__

<div align="center">
<a href="https://nbviewer.org/github/Nikoletos-K/pyJedAI/blob/main/Demo.ipynb">
<img align="center" src="https://nbviewer.org/static/img/nav_logo.svg" width=120/> 
</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://github.com/Nikoletos-K/pyJedAI/blob/main/Demo.ipynb">
<img align="center" src="https://miro.medium.com/max/1400/1*Edn_LpbSpLeNKfWkEdG2Jg.png" width=120/> 
</a>
</div>
<!--  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://colab.research.google.com/github/Nikoletos-K/pyJedAI/blob/main/CleanCleanER-AbtBuy.ipynb">
<img align="center" src="https://blog.educationecosystem.com/wp-content/uploads/2021/01/1_Lad06lrjlU9UZgSTHUoyfA.png" width=120/> 
</a>
--->


<details>
<summary><h4>Details on the Architecture</h4></summary>
<br>
The purpose of this framework is to demonstrate how ER can be accomplished by expert and novice users in an intuitive, yet efficient and effective way. pyJedai addresses the following task: Given a source and a target dataset, S and T, respectively, discover the set of links L = {(s,owl:sameAS, t)|s ∈ S ∧ t ∈ T}. Its architecture appears in the bellow figure. The first module is the data reader, which specifies the user input. pyJedAI supports both semi-structured and structured data as input. The former, which include SPARQL endpoints and RDF/OWL dumps, are read by <a href="https://rdflib.dev">RDFLib</a>. The latter, which include relational databases as well as CSV and JSON files, are read by <a href="https://pandas.pydata.org">pandas</a>. In this way, pyJedAI is able to interlink any combination of semi-structured and structured data sources, which is a unique feature. <img align="right" src="https://github.com/Nikoletos-K/pyJedAI/blob/main/documentation/pyJedAIarchitecture.png?raw=true" alt="pyJedAI-Architecture" width="500"/> The second step in pyJedAI’s pipeline performs block building, a coarsegrained process that clusters together similar entities. The end result consists of a set of candidate pairs, which are examined analytically by the subsequent steps. pyJedAI implements the same established methods for similarity joins and blocking as JedAI, such as Standard Blocking and Sorted Neighborhood, but goes beyond all Link Discovery tools by incorporating recent, state-of-the-art libraries for nearest neighbor search like <a href="https://falconn-lib.org">FALCONN</a> and <a href="https://github.com/facebookresearch/faiss">FAISS</a>. <br>

<br>

 The entity matching step estimates the actual similarity between the candidate pairs. Unlike all other Link Discovery tools, which rely exclusively on string similarity measures like edit distance and Jaccard coefficient, pyJedAI leverages the latest advanced NLP techniques, like pre-trained embeddings (e.g., word2vect, fastText and Glove) and transformer language models (i.e., BERT and its variants). More specifically, pyJedAI supports packages like <a href="https://github.com/luozhouyang/python-string-similarity">strsimpy</a>, <a href="https://radimrehurek.com/gensim/">Gensim</a>and <a href="https://huggingface.co">Hugging Face</a>. This unique feature boosts pyJedAI’s accuracy to a significant extent, without requiring any labelled instances from the user. The last step performs entity clustering to further increase the accuracy. The relevant techniques consider the global information provided by the similarity scores of all candidate pairs in order to take local decisions for each pair of entity descriptions. pyJedAI implements and offers the same established algorithms as JedAI, using <a href="https://networkx.org">NetworkX</a> to ensure high time efficiency. Finally, users are able to evaluate, visualize and store the results of the selected pipeline through the intuitive interface of Jupyter notebooks. In this way, pyJedAI facilitates its use by researchers and practitioners that are familiar with the data science ecosystem, regardless of their familiarity with ER and Link
Discovery, in general.

</details>


# Install

Install the latest version of pyjedai __[requires python >= 3.9]__:
```
pip install pyjedai
```

More on [PyPI](pypi.org/project/pyjedai/).

# Dependencies

<div align="center">
<img align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/2560px-Pandas_logo.svg.png" width=120/> &nbsp;&nbsp;
<img align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/1280px-NumPy_logo_2020.svg.png" width=120/> &nbsp;&nbsp;
<img align="center" src="https://logoeps.com/wp-content/uploads/2012/10/python-logo-vector.png" width=120/> &nbsp;&nbsp;&nbsp;
<img align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/883px-Jupyter_logo.svg.png" width=70/>  <br>
<img align="center" src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width=150/>
<img align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Plotly_logo_for_digital_final_%286%29.png/1200px-Plotly_logo_for_digital_final_%286%29.png" width=150/>
<img align="center" src="https://www.fullstackpython.com/img/logos/scipy.png" width=150/>  <br><br>
<img align="center" src="https://www.kornosk.me/resources/language-model/featured.png" width=150/> &nbsp;&nbsp;&nbsp;
<img align="center" src="https://repository-images.githubusercontent.com/1349775/202c4680-8f7c-11e9-91c6-745fdcbeffe8" width=150/> &nbsp;&nbsp;&nbsp;
<img align="center" src="https://networkx.org/_static/networkx_logo.svg" width=150/> &nbsp;&nbsp;&nbsp;
<img align="center" src="https://raw.githubusercontent.com/RDFLib/OWL-RL/master/OWL-RL.png" width=70/> 
</div>

See the full list of dependencies and all versions used, in this [file](https://github.com/Nikoletos-K/pyJedAI/blob/main/requirements.txt).

# Bugs, Discussions & News

[GitHub Discussions](https://github.com/Nikoletos-K/pyJedAI/discussions) is the discussion forum for general questions and discussions and our recommended starting point. Please report any bugs that you find [here](https://github.com/Nikoletos-K/pyJedAI/issues).


# Team & Authors

<img align="right" src="https://www.di.uoa.gr/themes/corporate_lite/logo_en.png" alt="pyJedAI" width="400"/>

- [Konstantinos Nikoletos](https://nikoletos-k.github.io)
- [George Papadakis](https://gpapadis.wordpress.com)
- [Manolis Koubarakis](https://cgi.di.uoa.gr/~koubarak/)

Research and development is made under the supervision of the Pr. Manolis Koubarakis. This is a research project by the [AI-Team](https://ai.di.uoa.gr) of the Department of Informatics and Telecommunications at the University of Athens.

<br>

# License

Released under the Apache-2.0 license [(see LICENSE.txt)](https://github.com/Nikoletos-K/pyJedAI/blob/main/LICENSE).

Copyright © 2022 pyJedAI Developers
