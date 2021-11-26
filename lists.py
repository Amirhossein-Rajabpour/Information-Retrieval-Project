import numpy as np
from main import *


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
def create_positional_index(array_of_docs):
    positional_index = {}
    for doc in array_of_docs:
        for term in doc.content:
            if term not in positional_index.keys():     # we are evaluating a new term (this is the first time that this term has appeared in all docs)
                positions_in_doc = find_positions_in_doc(term, doc.content)
                positions_dictionary = {doc.id: (len(positions_in_doc), positions_in_doc)}  # positions_dictionary didn't exist before for this term
                positional_index[term] = [len(positions_in_doc), positions_dictionary]  # add the new term to the positional_index dictionary
            elif term in positional_index.keys() and doc.id in positional_index.get(term)[1].keys():
                pass    # everything is already calculated for this term (this term appeared in this doc more than once)
            else:   # we had this term before in other docs but it is new in this doc
                positions_in_doc = find_positions_in_doc(term, doc.content)
                positional_index.get(term)[1][doc.id] = (len(positions_in_doc), positions_in_doc)    # TODO: positions_dictionary exists for this term and should be updated (error)
                positional_index[term] = [positional_index.get(term)[0] + len(positions_in_doc), positions_dictionary]  # update the whole frequency of that term and its new positions_dictionary (with new doc)

    return positional_index


if __name__ == '__main__':
    tmp_arr = np.array(['ما', 'ه', 'ما'])
    tmp_arr2 = np.array(['ما', 'ه', 'ما', 'برا', 'وصل', 'کردن', 'آمدیم!', 'ول', 'برا', 'پردازش،', 'جدا', 'به', 'نیست؟','اصلاح', 'کتاب', 'و', 'استفاده', 'از', 'نیم\u200cفاصله', 'پرداز', 'را', 'آس', 'می\u200cکند'])

    array_of_docs = []
    document1 = Document(id=111, title="title", content=tmp_arr, url="url")
    document2 = Document(id=222, title="title2", content=tmp_arr2, url="url2")

    array_of_docs.append(document1)
    array_of_docs.append(document2)

    tmp_positional_index = create_positional_index(array_of_docs)
    print("positional index:\n", tmp_positional_index)
