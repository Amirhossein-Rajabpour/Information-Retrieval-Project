import math
import hazm
import numpy as np
from numpy.linalg import norm
from process_query import query_stemming
from gensim.models import Word2Vec
import pickle
from tfidf import calculate_tf_query, calculate_idf


def load_doc_tfidf(path):
    with open(path, 'rb') as doc_tfidf_file:
        doc_tfidf = pickle.load(doc_tfidf_file)
    return doc_tfidf


def cos_similarity_emb(query, doc):
    similarity_score = np.dot(query, doc) / (norm(query) * norm(doc))
    return similarity_score


def set_doc_embeddings(doc_embeddings, collection):
    i = 0
    for doc in collection:
        doc.embeddings = doc_embeddings[i]
        i += 1
    return collection


def calculate_query_word_scores(query, terms, collection):
    seen_terms_query = []
    query_scores = {}
    for term in query:  # here term is string
        if term not in seen_terms_query:  # avoid calculating tfidf more than one time for each term in doc
            query_scores[term] = (1 + math.log10(calculate_tf_query(term, query))) * calculate_idf(len(collection),
                                                                                                   terms.get(term))
            seen_terms_query.append(term)
    return query_scores


def initialize_word2vec(w2v_model_path, collection):
    # load word2vec model
    w2v_model = Word2Vec.load(w2v_model_path)

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

    collection = set_doc_embeddings(doc_embeddings, collection)
    return collection


def query_word2vec(query, w2v_model_path, terms, collection):
    # load word2vec model
    w2v_model = Word2Vec.load(w2v_model_path)

    # first we should preprocess the query like our dataset
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()
    query = normalizer.normalize(query)
    splitted_query = query.split(" ")
    query = query_stemming(splitted_query, stemmer)

    # create word2vec vector for query (weighted average with tf-idf as word weights)
    query_word_scores = calculate_query_word_scores(query, terms, collection)
    query_vector = np.zeros(300)
    weights_sum = 0
    for token, weight in query_word_scores.items():
        query_vector += w2v_model.wv[token] * weight
        weights_sum += weight
    query_embedding = query_vector/weights_sum

    # calculate cosine similarity for query and docs, return k best docs
    doc_scores = {}
    for doc in collection:
        doc_scores[doc] = cos_similarity_emb(query_embedding, doc.embeddings)

    doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
    # return top K docs
    k = 5
    first_K_pairs = {i: doc_scores[i] for i in list(doc_scores)[:k]}
    return first_K_pairs

