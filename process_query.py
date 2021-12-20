def query_stemming(splitted_query, stemmer):
    for i in range(len(splitted_query)):
        splitted_query[i] = stemmer.stem(splitted_query[i])
    return splitted_query


def find_titles(list_of_doc_ids, collection):
    list_of_doc_titles = []
    # print("Doc ids sorted by score:\n", list_of_doc_ids, "\n")
    for doc in collection:
        if str(doc.id) in list_of_doc_ids:
            list_of_doc_titles.append(doc.title)
    return list_of_doc_titles


def search_single_word(word, positional_index):
    word_object = positional_index.get(word)
    return word_object.docs_lists()


def find_longest_substring(query, document, positional_index):    # in this function the longest substring in the document should be found
    longest_substring = 1
    reached_max_score = False
    for sub in range(len(query), 1, -1):    # iterates from [len, 2]: from longest to shortest substrings
        if not reached_max_score:
            for term_index in range(len(query)-sub+1):
                term_positions = positional_index[query[term_index]].pos_in_each_doc[str(document.id)]

                # check next positions
                for pos in term_positions:
                    for i in range(1, len(query)+1):
                        if document.content[pos + i] == query[i]:
                            longest_substring = i + 1
                        else:
                            break
                if longest_substring > 1:
                    reached_max_score = True
                    break
        else:   # it means we have reached max score
            return longest_substring

    return longest_substring


def score_docs(intersection_list, query, collection, positional_index):  # in this function docs will be scored
    doc_scores = {}
    for doc_id in intersection_list:
        doc_scores[doc_id] = 1      # first the scores of the docs that have all the words are initialized to 1

    for doc_id in doc_scores.keys():
        doc_scores[doc_id] = find_longest_substring(query, collection[int(doc_id)], positional_index)     # doc_id is string but in collection doc_id is int (index)

    # sort the dictionary (docs) by its scores
    doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
    return doc_scores.keys()


def intersections_for_different_lengths(query, positional_index):
    intersections = []
    list_for_different_words = []
    for word in query:
        list_for_different_words.append(positional_index.get(word).docs_lists())

    for length in range(len(query), 0, -1):
        intersection_with_length = set()    # this set() means we should OR our sets for this length
        for start in range(len(query)-length+1):
            inner_intersection = set(list_for_different_words[start])
            for index in range(start+1, start+length):
                inner_intersection = inner_intersection.intersection(list_for_different_words[index])
            intersection_with_length.update(inner_intersection)
        intersections.append(list(intersection_with_length))

    return intersections


def search_multi_word(query, positional_index, collection):
    # list_for_different_words = []
    # for word in query:
    #     list_for_different_words.append(positional_index.get(word).docs_lists())
    # # now i should continue the search in the mutual docs
    # intersection_list = set(list_for_different_words[0])
    # for index in range(1, len(list_for_different_words)):
    #     intersection_list = intersection_list.intersection(list_for_different_words[index])
    # intersection_list = list(intersection_list)
    intersection_list = intersections_for_different_lengths(query, positional_index)
    ranked_docs_total = set()
    ranked_docs2 = score_docs(intersection_list[0], query, collection, positional_index)

    print("Length of query: ", len(query))
    for i in range(len(intersection_list)):
        print(len(intersection_list[i]), "results for query with length of ", len(query)-i)
    print("")

    # for i in range(len(intersection_list)):
    #     ranked_docs = score_docs(intersection_list[i], query, collection, positional_index)
    #     ranked_docs_total.update(list(ranked_docs))
    # print(ranked_docs_total)
    return list(ranked_docs2)


def process_query(query, positional_index, collection):
    # we should search in our dataset
    if len(query) == 1:
        doc_ids = search_single_word(query[0], positional_index)
    else:
        doc_ids = search_multi_word(query, positional_index, collection)

    list_of_doc_titles = find_titles(doc_ids, collection)
    return list_of_doc_titles, list(doc_ids)

