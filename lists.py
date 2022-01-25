class Term:
    def __init__(self, string):
        self.string = string
        self.total_freq = 0
        self.pos_in_each_doc = {}   # {doc_ic: list positions, ...}
        self.freq_in_each_doc = {}  # {doc_id: freq, ...}
        self.champion_list = {}

    def docs_lists(self):
        return self.pos_in_each_doc.keys()

    def set_total_freq(self, total_freq):
        self.total_freq = total_freq

    def set_pos_in_each_doc(self, pos_in_each_doc):
        self.pos_in_each_doc = pos_in_each_doc

    def set_freq_in_each_doc(self, freq_in_each_doc):
        self.freq_in_each_doc = freq_in_each_doc

    def add_doc(self, doc_id, term_position):   # term is identified in this doc for the first time
        self.pos_in_each_doc[str(doc_id)] = []
        self.pos_in_each_doc[str(doc_id)].append(term_position)
        self.freq_in_each_doc[str(doc_id)] = 1
        self.total_freq += 1

    def update_doc(self, doc_id, term_position):
        self.pos_in_each_doc[str(doc_id)].append(term_position)
        self.freq_in_each_doc[str(doc_id)] += 1
        self.total_freq += 1


def create_positional_index(array_of_docs):
    terms = {}  # {term_string: term object}
    for doc in array_of_docs:
        term_index = 0
        for term in doc.content:
            if term not in terms.keys():   # if term is new in our collection
                term_obj = Term(string=term)
                term_obj.add_doc(doc.id, term_index)
                terms[term] = term_obj
            elif term in terms.keys() and str(doc.id) not in terms.get(term).docs_lists():   # first time in this doc
                terms[term].add_doc(doc.id, term_index)
            else:   # term is not new in collection and in this doc
                terms.get(term).update_doc(doc.id, term_index)
            term_index += 1
    return terms
