import numpy as np


# it returns the positions of that term in that doc
def find_positions_in_doc(term, doc):
    positions = []
    for i in range(len(doc)):
        if doc[i] == term:
            positions.append(i)
    return positions


# TODO: create positional index
# the structure of my positional index is like this:
# positional_index = {
#   word1: [frequency in all docs, {doc1: (freq in doc1, [positions in doc1]), doc2:(freq in doc2, [positions in doc2]), ...}],
#   word2: [frequency in all docs, {doc1: (freq in doc1, [positions in doc1]), doc2:(freq in doc2, [positions in doc2]), ...}],
#   ...
# }
def create_positional_index(array_of_docs):     # we need doc id
    positional_index = {}
    for doc in array_of_docs:
        for term in doc.body:     # the same for doc.title
            positions_in_doc = find_positions_in_doc(term, doc.body)
            if term not in positional_index.keys():
                positions_dictionary = {doc.id: (len(positions_in_doc), positions_in_doc)}  # positions_dictionary didn't exist before for this term
                positional_index[term] = [len(positions_in_doc), positions_dictionary]  # add the new term to the positional_index dictionary
            else:
                # TODO: here we should check that each term should be evaluated in each doc only once!!
                positions_dictionary[doc.id] = (len(positions_in_doc), positions_in_doc)    # positions_dictionary exists for this term and should be updated
                # TODO: test
                positional_index[term] = [positional_index.get(term)[0] + len(positions_in_doc), positions_dictionary]  # update the whole frequency of that term and its new positions_dictionary (with new doc)

    return positional_index


if __name__ == '__main__':
    tmp_arr = np.array([[['اصلاح', 'کتاب', 'و', 'استفاده', 'از', 'نیم\u200cفاصله', 'پرداز', 'را', 'آس', 'می\u200cکند'],
                         [0],
                         ['ما', 'ه', 'برا', 'وصل', 'کردن', 'آمدیم!', 'ول', 'برا', 'پردازش،', 'جدا', 'به', 'نیست؟']],
                        [[], [0], []]],
                       dtype=object, )

    tmp_positional_index = create_positional_index(tmp_arr)
    print("positional index:\n", tmp_positional_index)
