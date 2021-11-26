import numpy as np
from main import *

class Term:
    def __init__(self, string):
        self.string = string
        self.total_freq = 0
        self.pos_in_each_doc = {}   # {doc_ic: list positions, ...}
        self.freq_in_each_doc = {}  # {doc_id: freq, ...}

    def docs_lists(self):
        return self.pos_in_each_doc.keys()

    def add_doc(self, doc_id, term_position):   # term is identified in this doc for the first time
        self.pos_in_each_doc[doc_id] = []
        self.pos_in_each_doc[doc_id].append(term_position)
        self.freq_in_each_doc[doc_id] = 1
        self.total_freq += 1

    def update_doc(self, doc_id, term_position):
        self.pos_in_each_doc[doc_id].append(term_position)
        self.freq_in_each_doc[doc_id] += 1
        self.total_freq += 1


# it returns the positions of that term in that doc
def find_positions_in_doc(term, doc):
    positions = []
    for i in range(len(doc)):
        if doc[i] == term:
            positions.append(i)
    return positions


def create_positional_index(array_of_docs):

    terms = {}  # {term_string: term object}
    for doc in array_of_docs:
        term_index = 0
        for term in doc.content:
            if term not in terms.keys():   # if term is new in our collection
                term_obj = Term(string=term)
                term_obj.add_doc(doc.id, term_index)
                terms[term] = term_obj
            elif term in terms.keys() and doc.id not in terms.get(term).docs_lists():   # first time in this doc
                terms[term].add_doc(doc.id, term_index)
            else:   # term is not new in collection and in this doc
                terms.get(term).update_doc(doc.id, term_index)
            term_index += 1
    return terms



if __name__ == '__main__':
    tmp_arr = np.array(['ما', 'ه', 'ما'])
    tmp_arr2 = np.array(['ما', 'ه', 'ما', 'برا', 'وصل', 'کردن', 'آمدیم!', 'ول', 'برا', 'پردازش،', 'جدا', 'به', 'نیست؟','اصلاح', 'کتاب', 'و', 'استفاده', 'از', 'نیم\u200cفاصله', 'پرداز', 'را', 'آس', 'می\u200cکند'])

    array_of_docs = []
    document1 = Document(id=111, title="title", content=tmp_arr, url="url")
    document2 = Document(id=222, title="title2", content=tmp_arr2, url="url2")

    array_of_docs.append(document1)
    array_of_docs.append(document2)

    terms_pos_index = create_positional_index(array_of_docs)
    print("positional index:\n", tmp_positional_index)
