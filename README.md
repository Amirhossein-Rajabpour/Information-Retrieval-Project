## Information retrieval course project - Fall 2021


Implementing a `search engine` using different search models and algorithms like `binary search`, `tf-idf`, and `word embeddings`. Also, implementing `K-means` clustering and `KNN` algorithms to speed up the search.

### Dataset
There are two datasets. One has 7k news articles, and the second one has 50k news articles used for clustering. Articles in the second dataset have categories (`sport`, `economy`, `politics`, `culture`, and `health`), and they were used in `KNN` for labeling articles in the first dataset. Datasets are not in this repository.


### Inputs
First, you create an inverted index model with option 1 or load a previous model with option 2. Options 3 and 4 are for Zipf and Heaps law, respectively. Options 5 is for initializing our kmeans model (which takes about an hour). Option 6 is for labeling our small dataset (that contains 7k news articles) using our bigger dataset (that contains 50k news articles).

### Search
As second input, you should choose which model to use (between simple `binary` model, `tf-idf` model, `word2vec` model, `k-means` model or `KNN`) for search and then write your query.

### Output
Top 5 search results and their scores are shown to you.
