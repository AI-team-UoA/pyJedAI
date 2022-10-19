# About

<!-- The bellow diagram depicts the main idea and scope of pyJedAI. -->

<br>

<div align="center">
<img align="right" src="https://github.com/Nikoletos-K/pyJedAI/blob/main/documentation/3DERslide.png?raw=true?raw=true" alt="pyJedAI-Architecture" width="500"/>
</div>

<br>

# Details on the Architecture

The purpose of this framework is to demonstrate how ER can be accomplished by expert and novice users in an intuitive, yet efficient and effective way. pyJedai addresses the following task: Given a source and a target dataset, S and T, respectively, discover the set of links L = {(s,owl:sameAS, t)|s ∈ S ∧ t ∈ T}. Its architecture appears in the bellow figure. The first module is the data reader, which specifies the user input. pyJedAI supports both semi-structured and structured data as input. The former, which include SPARQL endpoints and RDF/OWL dumps, are read by <a href="https://rdflib.dev">RDFLib</a>. The latter, which include relational databases as well as CSV and JSON files, are read by <a href="https://pandas.pydata.org">pandas</a>. In this way, pyJedAI is able to interlink any combination of semi-structured and structured data sources, which is a unique feature.  The second step in pyJedAI’s pipeline performs block building, a coarsegrained process that clusters together similar entities. The end result consists of a set of candidate pairs, which are examined analytically by the subsequent steps. pyJedAI implements the same established methods for similarity joins and blocking as JedAI, such as Standard Blocking and Sorted Neighborhood, but goes beyond all Link Discovery tools by incorporating recent, state-of-the-art libraries for nearest neighbor search like <a href="https://falconn-lib.org">FALCONN</a> and <a href="https://github.com/facebookresearch/faiss">FAISS</a>.

<br>

<div align="center">
<img align="center" src="https://github.com/Nikoletos-K/pyJedAI/blob/main/documentation/demo-architecture.png?raw=true?raw=true" alt="pyJedAI-Architecture" width="500"/>
</div>

<br>

 The entity matching step estimates the actual similarity between the candidate pairs. Unlike all other Link Discovery tools, which rely exclusively on string similarity measures like edit distance and Jaccard coefficient, pyJedAI leverages the latest advanced NLP techniques, like pre-trained embeddings (e.g., word2vect, fastText and Glove) and transformer language models (i.e., BERT and its variants). More specifically, pyJedAI supports packages like <a href="https://github.com/luozhouyang/python-string-similarity">strsimpy</a>, <a href="https://radimrehurek.com/gensim/">Gensim</a>and <a href="https://huggingface.co">Hugging Face</a>. This unique feature boosts pyJedAI’s accuracy to a significant extent, without requiring any labelled instances from the user. The last step performs entity clustering to further increase the accuracy. The relevant techniques consider the global information provided by the similarity scores of all candidate pairs in order to take local decisions for each pair of entity descriptions. pyJedAI implements and offers the same established algorithms as JedAI, using <a href="https://networkx.org">NetworkX</a> to ensure high time efficiency. Finally, users are able to evaluate, visualize and store the results of the selected pipeline through the intuitive interface of Jupyter notebooks. In this way, pyJedAI facilitates its use by researchers and practitioners that are familiar with the data science ecosystem, regardless of their familiarity with ER and Link
Discovery, in general.
