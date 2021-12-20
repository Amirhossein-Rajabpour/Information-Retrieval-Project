import math
import pickle


# calculate idf for each term
def calculate_idf(collection_size, term):  # term is object
    return math.log10(collection_size / len(term.freq_in_each_doc))  # N/num of docs containing t


# calculate term frequency (tf) for a given term. term is object
def calculate_tf_doc(term, doc):
    return term.freq_in_each_doc.get(str(doc.id))


def calculate_tf_query(term, query):    # here term is string
    frequency = 0
    for word in query:
        if word == term:
            frequency += 1
    return frequency


def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))


# calculate cos similarity
def cos_similarity(query_vector, doc_scores_vector):
    # return cos_score between query and doc
    # find terms (that are in query) in doc and dont iterate over all term in doc
    numerator = 0
    for term in query_vector:
        if term in doc_scores_vector:
            numerator += (query_vector.get(term) * doc_scores_vector.get(term))

    cos_score = numerator / (magnitude(list(query_vector.values())) * magnitude(list(doc_scores_vector.values())))
    return cos_score


# create champion list
def create_champion_list(terms, r=300):
    # for each term we should consider only r most important docs
    champion_list = {}
    for term in terms:
        # freq_in_each_doc should be sorted by freq and then choose first r
        sorted_freq_in_docs = dict(sorted(terms[term].freq_in_each_doc.items(), key=lambda item: terms[term].freq_in_each_doc[1], reverse=True))
        first_r_docs = {i: sorted_freq_in_docs[i] for i in list(sorted_freq_in_docs)[:r]}
        champion_list[term] = first_r_docs
    return champion_list


# return docs that have at least one word in common with query
def index_elimination(query, collection):
    docs_after_elimination = []
    for doc in collection:
        for term in query:
            if term in doc.content:
                docs_after_elimination.append(doc)
                break
    return docs_after_elimination


def create_doc_tfidf_file(collection):
    doc_tfidf = []
    for doc in collection:
        doc_tfidf.append(doc.term_scores)
    with open('docs_tf_idf.obj', 'wb') as docs_tf_idf_file:
        pickle.dump(doc_tfidf, docs_tf_idf_file)


def tf_idf(query, terms, collection):
    seen_terms_query = []
    query_scores = {}
    for term in query:  # here term is string
        if term not in seen_terms_query:  # avoid calculating tfidf more than one time for each term in doc
            query_scores[term] = (1 + math.log10(calculate_tf_query(term, query))) * calculate_idf(len(collection), terms.get(term))
            seen_terms_query.append(term)

    for doc in collection:  # create a term score vector for each doc
        seen_terms = []
        for term in doc.content:  # here term is string
            if term not in seen_terms:  # avoid calculating tfidf more than one time for each term in doc
                doc.term_scores[term] = (1 + math.log10(calculate_tf_doc(terms.get(term), doc))) * calculate_idf(len(collection), terms.get(term))
                seen_terms.append(term)

    # TODO: change the place of this god damn function
    create_doc_tfidf_file(collection)

    # eliminating docs with no mutual word with query (for faster calculation)
    docs_after_elimination = index_elimination(query, collection)
    similarities = {}  # {doc: similarity, ...}
    for doc in docs_after_elimination:
        similarities[doc] = cos_similarity(query_scores, doc.term_scores)

    # sort scores dictionary
    similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))

    # return top K docs
    k = 5
    first_K_pairs = {i: similarities[i] for i in list(similarities)[:k]}
    return first_K_pairs
