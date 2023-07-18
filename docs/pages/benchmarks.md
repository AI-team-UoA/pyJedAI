Benchmarks
=============

## Dataset specifying the best configuration

|  	| D1 	|  	|  	| D2 	|  	|  	| D3 	|  	|  	| D5 	|  	|  	| D8 	|  	|  	| D10 	|  	|  	|
|---	|:---:	|:---:	|---	|:---:	|:---:	|---	|:---:	|:---:	|---	|:---:	|:---:	|---	|:---:	|:---:	|---	|:---:	|:---:	|---	|
|  	| Recall 	| Precision 	| F1 	| Recall 	| Precision 	| F1 	| Recall 	| Precision 	| F1 	| Recall 	| Precision 	| F1 	| Recall 	| Precision 	| F1 	| Recall 	| Precision 	| F1 	|
| D1 	| 1.000 	| 0.788 	| 0.881 	| 1.000 	| 0.263 	| 0.416 	| 0.989 	| 0.281 	| 0.438 	| 1.000 	| 0.322 	| 0.488 	| 0.933 	| 0.417 	| 0.576 	| 1.000 	| 0.263 	| 0.416 	|
| D2 	| 0.000 	| 0.000 	| 0.000 	| 0.942 	| 0.952 	| 0.947 	| 0.695 	| 0.798 	| 0.743 	| 0.851 	| 0.922 	| 0.885 	| 0.270 	| 0.997 	| 0.424 	| 0.746 	| 0.853 	| 0.796 	|
| D3 	| 0.037 	| 0.539 	| 0.069 	| 0.431 	| 0.354 	| 0.389 	| 0.674 	| 0.584 	| 0.625 	| 0.486 	| 0.425 	| 0.454 	| 0.092 	| 0.545 	| 0.158 	| 0.523 	| 0.472 	| 0.496 	|
| D4 	| 0.000 	| 0.000 	| 0.000 	| 0.808 	| 0.794 	| 0.801 	| 0.920 	| 0.702 	| 0.796 	| 0.931 	| 0.844 	| 0.886 	| 0.038 	| 0.961 	| 0.072 	| 0.845 	| 0.659 	| 0.741 	|
| D5 	| 0.096 	| 0.812 	| 0.172 	| 0.856 	| 0.288 	| 0.431 	| 0.673 	| 0.235 	| 0.348 	| 0.792 	| 0.330 	| 0.466 	| 0.753 	| 0.671 	| 0.709 	| 0.859 	| 0.302 	| 0.446 	|
| D6 	| 0.000 	| 0.000 	| 0.000 	| 0.000 	| 0.000 	| 0.000 	| 0.474 	| 0.748 	| 0.580 	| 0.805 	| 0.924 	| 0.860 	| 0.014 	| 0.970 	| 0.028 	| 0.858 	| 0.940 	| 0.897 	|


## Datasets specs

| Dataset 	| E1 	| E2 	| Entities E1 	| Entities E2 	| Duplicates 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| D1 	| Restaurants 1  	| Restaurants 2  	| 339 	| 2,256 	| 89 	|
| D2 	|  Abt  	|  Buy  	| 1,076 	| 1,076 	| 1,076 	|
| D3 	|  Amazon  	|  Google Pr.  	| 1,354 	| 3,039 	| 1,104 	|
| D4 	|  IMDb  	|  TMDb  	| 5,118 	| 6,056 	| 1,968 	|
| D5 	|  Walmart  	|  Amazon  	| 2,554 	| 22,074 	| 853 	|
| D6 	|  IMDb 	|  DBpedia 	| 27,615 	| 23,182 	| 22,863 	|


## Configurations specifics

|  	| Block Building 	| Blocking Cleaning 	|  	| Comprison Cleaning 	|  	| Entity Matching 	|  	|  	| Entity Clustering 	|  	|
|---	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
|  	|  	| Method 	| Ratio 	| Pruning algorithm 	| Weighting Scheme 	| Algorithm 	| Representation Model 	| Similarity Function 	| Algorithm 	| Similarity Threshold 	|
| D1 	| Standard Blocking 	| Block Filtering 	| 0.050 	| BLAST 	| ARCS 	| Profile Matcher 	| CHARACTER_BIGRAMS 	| COSINE_SIMILARITY 	| Unique Mapping Clustering 	| 0.90 	|
| D2 	| Standard Blocking 	| Block Filtering 	| 0.900 	| WEP 	| EJS 	| Profile Matcher 	| CHARACTER_TRIGRAMS_TF_IDF 	| ARCS_SIMILARITY 	| Unique Mapping Clustering 	| 0.90 	|
| D3 	| Standard Blocking 	| Block Filtering 	| 0.600 	| WNP 	| ARCS 	| Profile Matcher 	| TOKEN_BIGRAMS_TF_IDF 	| COSINE_SIMILARITY 	| Unique Mapping Clustering 	| 0.05 	|
| D4 	| Standard Blocking 	| Block Filtering 	| 0.925 	| CEP 	| ECBS 	| Profile Matcher 	| CHARACTER_FOURGRAMS_TF_IDF 	| ARCS_SIMILARITY 	| Unique Mapping Clustering 	| 0.85 	|
| D5 	| Standard Blocking 	| Block Filtering 	| 0.075 	| WEP 	| ARCS 	| Profile Matcher 	| CHARACTER_BIGRAMS_TF_IDF 	| COSINE_SIMILARITY 	| Unique Mapping Clustering 	| 0.65 	|
| D6 	| Standard Blocking 	| Block Filtering 	| 0.575 	| BLAST 	| X2 	| Profile Matcher 	| TOKEN_UNIGRAMS_TF_IDF 	| ARCS_SIMILARITY 	| Unique Mapping Clustering 	| 0.25 	|