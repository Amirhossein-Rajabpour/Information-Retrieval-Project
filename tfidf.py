import math
import hazm
from process_query import query_stemming


# calculate idf for each term
def calculate_idf(collection_size, term):  # term is object
    return math.log10(collection_size / len(term.freq_in_each_doc))  # N/num of docs containing t


# calculate term frequency (tf) for a given term. term is object
def calculate_tf(term, doc):
    return term.freq_in_each_doc.get(doc.id)


# TODO: calculate cos similarity
def cos_similarity(query, doc):
    # return cos_score between query and doc
    score = 0
    return score


# TODO: create champion list
def create_champion_list(terms):
    pass


def tf_idf(query, terms, collection):
    for doc in collection:  # create a term score vector for each doc
        seen_terms = []
        for term in doc.content:  # here term is string
            if term not in seen_terms:  # avoid calculating tfidf more than one time for each term in doc
                doc.term_scores[term] = (1 + math.log10(calculate_tf(terms.get(term), doc))) * calculate_idf(len(collection), terms.get(term))
                seen_terms.append(term)

    # first we should preprocess the query like our dataset
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()
    query = normalizer.normalize(query)
    splitted_query = query.split(" ")
    query = query_stemming(splitted_query, stemmer)
    seen_terms_query = []
    query_scores = {}
    for term in query:  # here term is string
        if term not in seen_terms_query:  # avoid calculating tfidf more than one time for each term in doc
            # TODO: how to calculate term frequency in query?
            query_scores[term] = (1 + math.log10(calculate_tf(terms.get(term), query))) * calculate_idf(len(collection),terms.get(term))
            seen_terms_query.append(term)

    similarities = {}  # {doc: similarity}
    for doc in collection:
        similarities[doc] = cos_similarity(query_scores, doc.term_scores)

    # sort scores dictionary
    similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))

    # return top K docs
    k = 5
    first_K_pairs = {k: similarities[k] for k in list(similarities)[:k]}