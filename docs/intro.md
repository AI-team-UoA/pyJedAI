(intro)=

# Overview

pyJedAI is a python framework, aiming to offer experts and novice users, robust and fast solutions for multiple types of Entity Resolution problems. It is builded using state-of-the-art python frameworks. pyJedAI constitutes the sole open-source Link Discovery tool that is capable of exploiting the latest breakthroughs in Deep Learning and NLP techniques, which are publicly available through the Python data science ecosystem. This applies to both blocking and matching, thus ensuring high time efficiency, high scalability as well as high effectiveness, without requiring any labelled instances from the user.

## Key-Features

- Input data-type independent. Both structured and semi-structured data can be processed.
- Various implemented algorithms.
- Easy-to-use.
- Utilizes some of the famous and cutting-edge machine learning packages.
- Offers supervised and un-supervised ML techniques.

# Install

```{warning}
We are currently experiencing issues with pypi installation due to a problem with the [py-stringmatching](https://pypi.org/project/py-stringmatching/) dependency. We are actively working on resolving it! A temporary solution is to first install:
`pip install 'numpy>=1.7.0,<2.0'` and then `pip install pyjedai`.
```

pyJedAI has been tested in Windows and Linux OS. 

```{admonition} Basic requirements
- Python version greater or equal to **3.8**.
- For Windows, Microsoft Visual C++ 14.0 is required. Download it from [Microsoft Official site](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
```

__PyPI__

Install the latest version of pyjedai:

```
pip install pyjedai
```
More on [PyPI](https://pypi.org/project/pyjedai).

__Git__

Set up locally:
```
git clone https://github.com/AI-team-UoA/pyJedAI.git
```
go to the root directory with `cd pyJedAI` and type:
```
pip install .
```


__Docker__

Available at [Docker Hub](https://hub.docker.com/r/aiteamuoa/pyjedai), or clone this repo and:
```
docker build -f Dockerfile
```


# Tutorials

Find all the code of the tutotials in the [pyjedai/docs/tutorials](https://github.com/AI-team-UoA/pyJedAI/tree/main/docs/tutorials) folder.

::::{grid}
:gutter: 3

:::{grid-item-card} Simple use case
<a href="https://nbviewer.org/github/AI-team-UoA/pyJedAI/blob/main/docs/tutorials/Demo.ipynb">
<img align="center" src="https://nbviewer.org/static/img/nav_logo.svg" width=120/> 
</a>
:::

:::{grid-item-card} Detailed notebook
<a href="https://github.com/AI-team-UoA/pyJedAI/blob/main/docs/tutorials/CleanCleanER.ipynb">
<img align="center" src="https://miro.medium.com/max/1400/1*Edn_LpbSpLeNKfWkEdG2Jg.png" width=120/> 
</a>
:::

:::{grid-item-card} Web Application using Colab
<a href="https://colab.research.google.com/drive/18VgEOKAc2ObFFxDNb2sjhBLKKsNvfEPo?usp=sharing">
<img align="center" src="https://3.bp.blogspot.com/-apoBeWFycKQ/XhKB8fEprwI/AAAAAAAACM4/Sl76yzNSNYwlShIBrheDAum8L9qRtWNdgCLcBGAsYHQ/s1600/colab.png" width=120/> 
</a>
:::
::::


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

<br>

See the full list of dependencies and all versions used, in this [file](https://github.com/AI-team-UoA/pyJedAI/blob/main/pyproject.toml).



__Status__

[![Tests](https://github.com/AI-team-UoA/pyJedAI/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/AI-team-UoA/pyJedAI/actions/workflows/tests.yml)
[![made-with-python](https://readthedocs.org/projects/pyjedai/badge/?version=latest)](https://pyjedai.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/AI-team-UoA/pyjedai/branch/master/graph/badge.svg?token=4QR0X315CL)](https://codecov.io/gh/AI-team-UoA/pyjedai)


__Statistics & Info__

![PyPI - Downloads](https://img.shields.io/pypi/dm/pyjedai)
[![PyPI version](https://img.shields.io/pypi/v/pyjedai.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/pyjedai/)

# Bugs, Discussions & News

[GitHub Discussions](https://github.com/Nikoletos-K/pyJedAI/discussions) is the discussion forum for general questions and discussions and our recommended starting point. Please report any bugs that you find [here](https://github.com/Nikoletos-K/pyJedAI/issues).


# Java - Web Application 

<img align="right" src="https://github.com/scify/JedAIToolkit/blob/master/documentation/JedAI_logo.png?raw=true" alt="pyJedAI" width="150"/>

For Java users checkout the initial [JedAI](https://github.com/scify/JedAIToolkit). There you can find Java based code and a Web Application for interactive creation of ER workflows. <br><br> JedAI constitutes an open source, high scalability toolkit that offers out-of-the-box solutions for any data integration task, e.g., Record Linkage, Entity Resolution and Link Discovery. At its core lies a set of domain-independent, state-of-the-art techniques that apply to both RDF and relational data.


# Team & Authors

::::{grid}
:gutter: 3

:::{grid-item-card} [Lefteris Stetsikas](https://github.com/Teris45)
Research Associate at [University of Athens](https://en.uoa.gr)
:::

:::{grid-item-card} [Konstantinos Nikoletos](https://nikoletos-k.github.io)
Research Associate at [University of Athens](https://en.uoa.gr)
:::


:::{grid-item-card} [Jakub Maciejewski](https://www.linkedin.com/in/jakub-maciejewski-0270291b7/)
Research Associate at [University of Athens](https://en.uoa.gr)
:::

::::

::::{grid}
:gutter: 3

:::{grid-item-card} [George Papadakis](https://gpapadis.wordpress.com)
Postdoctoral Researcher at [University of Athens](https://en.uoa.gr){bdg-primary}`Entity Resolution expert`
:::

:::{grid-item-card} [Ekaterini Ioannou](https://www.tilburguniversity.edu/staff/ekaterini-ioannou)
Assistant Professor at [Tilburg University](https://www.tilburguniversity.edu)
{bdg-primary}`Product Matching expert`
:::

:::{grid-item-card} [Manolis Koubarakis](https://cgi.di.uoa.gr/~koubarak/)
Professor at [University of Athens](https://en.uoa.gr)
:::
::::


Research and development is made under the supervision of Pr. Manolis Koubarakis. This is a research project by the [AI-Team](https://ai.di.uoa.gr) of the Department of Informatics and Telecommunications at the University of Athens.

# License

Released under the Apache-2.0 license [(see LICENSE.txt)](https://github.com/Nikoletos-K/pyJedAI/blob/main/LICENSE).

Copyright © 2023 AI-Team, University of Athens

<div align="center">
 <hr>
  <br>
 <a href="https://stelar-project.eu">
  <img align="center" src="https://stelar-project.eu/wp-content/uploads/2022/08/Logo-Stelar-1-f.png" width=180/>
 </a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href="https://ec.europa.eu/info/index_en">
  <img align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/1200px-Flag_of_Europe.svg.png" width=140/>
 </a>
 <br><br>
 <b>This project is being funded in the context of <a href="https://stelar-project.eu">STELAR</a>, an <a href="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en">HORIZON-Europe</a> project.</b><br>
</div>

