# TODO: calculate idf for each term
def calculate_idf(terms):   # terms is positional index
    # return dic{term_str: idf}
    pass


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


def tif_idf(query, collection, terms):
    pass