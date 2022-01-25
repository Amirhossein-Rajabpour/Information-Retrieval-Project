import hazm
# extracting Tokens from data
from process_query import query_stemming


def tokenize(doc):
    doc = "".join(doc)
    doc = hazm.sent_tokenize(doc)
    doc = "".join(doc)
    doc = hazm.word_tokenize(doc)
    return doc


# normalizing texts
def normalize(doc, normalizer):
    doc = "".join(doc)
    doc = normalizer.normalize(doc)
    return doc


# removing stop words and frequently used words
def remove_stop_words(doc):
    return doc


# stemming (may be a bit tricky)
def stem(doc, stemmer):
    doc = doc.split(" ")
    for term_index in range(len(doc)):
        doc[term_index] = stemmer.stem(doc[term_index])
    return doc


def preprocess_query(query):
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()
    query = normalizer.normalize(query)
    splitted_query = query.split(" ")
    query = query_stemming(splitted_query, stemmer)
    return query


# call all the preprocessing here
def preprocessing(array_of_docs, with_stemming=True):
    stop_words = hazm.stopwords_list()
    array_of_docs_preprocessed = []
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()

    for doc in array_of_docs:
        doc.content = normalize(doc.content, normalizer)
        if with_stemming:
            doc.content = stem(doc.content, stemmer)
        else:
            doc.content = doc.content.split(" ")

        array_of_docs_preprocessed.append(doc)
    return array_of_docs_preprocessed
