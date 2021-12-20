import hazm
from process_query import query_stemming


def initialize_word2vec():
    # TODO: load word2vec model

    # TODO: create word2vec vector for each doc (weighted average with tf-idf as word weights)
    pass


def query_word2vec(query, word2vec_model):
    # first we should preprocess the query like our dataset
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()
    query = normalizer.normalize(query)
    splitted_query = query.split(" ")
    query = query_stemming(splitted_query, stemmer)

    # TODO: create word2vec vector for query (weighted average with tf-idf as word weights)

    # TODO: calculate cosine similarity for query and docs, return best docs
