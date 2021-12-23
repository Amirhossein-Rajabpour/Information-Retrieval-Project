import math

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


def extract_term_scores_from_docs(term, dict_id_docs):    # term is object
    doc_scores = {}
    term_docs = list(term.docs_lists())
    for doc_id in term_docs:
        doc_scores[dict_id_docs.get(doc_id)] = dict_id_docs.get(doc_id).term_scores.get(term.string)
    return doc_scores


# create champion list
def add_champion_list(dict_id_docs, terms, r=300):
    # for each term we should consider only r most important docs
    for term in terms:  # term is string
        terms[term].champion_list = extract_term_scores_from_docs(terms[term], dict_id_docs)
        # sort it
        terms[term].champion_list = dict(sorted(terms[term].champion_list.items(), key=lambda item: item[1], reverse=True))
        # take first r of it
        terms[term].champion_list = {i: terms[term].champion_list[i] for i in list(terms[term].champion_list)[:r]}
    return terms


def extract_champion_from_query(query, terms_with_champion):
    champion_list = set()
    for word in query:
        champion_list.update(terms_with_champion[word].champion_list)
    return list(champion_list)


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


# return docs that have at least one word in common with query
def index_elimination(query, collection):
    docs_after_elimination = []
    for doc in collection:
        for term in query:
            if term in doc.content:
                docs_after_elimination.append(doc)
                break
    return docs_after_elimination


def create_id_docs_dict(collection):
    dict_id_docs = {}
    for doc in collection:
        dict_id_docs[str(doc.id)] = doc
    return dict_id_docs


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
                doc.term_scores[term] = (1 + math.log10(calculate_tf_doc(terms.get(term), doc)))
                seen_terms.append(term)

    dict_id_docs = create_id_docs_dict(collection)

    # eliminating docs with no mutual word with query (for faster calculation)
    docs_after_elimination = index_elimination(query, collection)
    terms_with_champion = add_champion_list(dict_id_docs, terms)
    champion_list = extract_champion_from_query(query, terms_with_champion)

    # intersect champion list and docs after elimination
    docs_to_search = intersection(champion_list, docs_after_elimination)

    similarities = {}  # {doc: similarity, ...}
    for doc in docs_to_search:
        similarities[doc] = cos_similarity(query_scores, doc.term_scores)

    # sort scores dictionary
    similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))

    # return top K docs
    k = 5
    first_K_pairs = {i: similarities[i] for i in list(similarities)[:k]}
    return first_K_pairs
