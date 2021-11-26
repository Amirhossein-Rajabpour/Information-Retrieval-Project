import hazm


def query_stemming(splitted_query, stemmer):
    for i in range(len(splitted_query)):
        splitted_query[i] = stemmer.stem(splitted_query[i])
    return splitted_query


def search_single_word(word, positional_index):
    word_object = positional_index.get(word)
    return word_object.docs_lists()


def search_multi_word(query, positional_index):
    list_for_different_words = []
    for word in query:
        list_for_different_words.append(positional_index.get(word).docs_lists())
    # now i should continue the search in the mutual docs
    intersection_list = set(list_for_different_words[0])
    for index in range(1, len(list_for_different_words)):
        intersection_list = intersection_list.intersection(list_for_different_words[index])
    return list(intersection_list)


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
        doc_ids = search_multi_word(query, positional_index)
    return doc_ids

