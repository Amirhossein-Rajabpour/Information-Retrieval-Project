import math

# TODO: calculate idf for each term
def calculate_idf(collection_size, term):   # term is object
    return math.log10(collection_size / len(term.freq_in_each_doc))     # N/num of docs containing t


# TODO: calculate term frequency (tf) for a given term. term is object
def calculate_tf(term, doc):
    # return frequency of the term in that doc
    return term.freq_in_each_doc.get(doc.id)


# TODO: calculate cos similarity
def cos_similarity(query, doc):
    # return cos_score between query and doc
    pass

# TODO: create champion list
def create_champion_list(terms):
    pass


def tf_idf(query, terms, collection):

    for doc in collection:  # create a term score vector for each doc
        seen_terms = []
        for term in doc.content:    # here term is string
            if term not in seen_terms:  # avoid calculating tfidf more than one time for each term in doc
                doc.term_scores[term] = (1 + math.log10(calculate_tf(terms.get(term), doc))) * calculate_idf(len(collection), terms.get(term))
                seen_terms.append(term)
