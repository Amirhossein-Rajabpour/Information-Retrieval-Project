import hazm


def query_stemming(splitted_query, stemmer):
    for i in range(len(splitted_query)):
        splitted_query[i] = stemmer.stem(splitted_query[i])
    return splitted_query


def search_single_word(word, positional_index):
    if word in positional_index.keys():
        word_object = positional_index.get(word)
        return word_object.docs_lists()


def search_multi_word(query, positional_index):
    pass


def process_query(query, positional_index):
    # first we should preprocess the query like our dataset
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()
    query = normalizer.normalize(query)
    splitted_query = query.split(" ")
    query = query_stemming(splitted_query, stemmer)

    # now we should search in our dataset
    if len(query) == 1:
        doc_ids = search_single_word(query[0], positional_index)
    else:
        search_multi_word(query, positional_index)
