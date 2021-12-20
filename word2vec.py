import hazm
import numpy as np
from numpy.linalg import norm
from process_query import query_stemming
from gensim.models import Word2Vec
import pickle

def load_doc_tfidf(path):
    with open(path, 'rb') as doc_tfidf_file:
        doc_tfidf = pickle.load(doc_tfidf_file)
    return doc_tfidf


def initialize_word2vec():
    # load word2vec model
    my_model = "D:\\uni\\semester 7\\Information Retrieval\\Project\\IR_Code\\my_w2v_model.model"
    hazm_model = ""
    w2v_model = Word2Vec.load(my_model)

    doc_tfidf_path = "D:\\uni\\semester 7\\Information Retrieval\\Project\\IR_Code\\docs_tf_idf.obj"
    doc_tfidf = load_doc_tfidf(doc_tfidf_path)

    # create word2vec vector for each doc (weighted average with tf-idf as word weights)
    doc_embeddings = []
    for doc in doc_tfidf:
        doc_vector = np.zeros(300)
        weights_sum = 0
        for token, weight in doc.items():
            doc_vector += w2v_model.wv[token] * weight
            weights_sum += weight
        doc_embeddings.append(doc_vector/weights_sum)
        doc.embeddings = doc_vector/weights_sum
    return doc_embeddings


def cos_similarity_emb(query, doc):
    similarity_score = np.dot(query, doc) / (norm(query) * norm(doc))
    return similarity_score


def query_word2vec(query, word2vec_model, collection):
    # first we should preprocess the query like our dataset
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()
    query = normalizer.normalize(query)
    splitted_query = query.split(" ")
    query = query_stemming(splitted_query, stemmer)

    # TODO: create word2vec vector for query (weighted average with tf-idf as word weights)
    query_embeddings = []   # watch types (list and numpy)

    # calculate cosine similarity for query and docs, return k best docs
    doc_scores = {}
    for doc in collection:
        doc_scores[doc] = cos_similarity_emb(query_embeddings, doc.embeddings)

    doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
    # return top K docs
    k = 5
    first_K_pairs = {i: doc_scores[i] for i in list(doc_scores)[:k]}
    return first_K_pairs

