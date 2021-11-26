import hazm
import numpy as np
from main import *


# extracting Tokens from data
def tokenize(doc):
    print("enter tokenize")
    doc = "".join(doc)
    doc = hazm.sent_tokenize(doc)
    doc = "".join(doc)  # TODO: here we remove the effect of sent_tokenize because we merge them all again in an array
    doc = hazm.word_tokenize(doc)
    print("doc", doc)
    return doc


# normalizing texts
def normalize(doc, normalizer):
    doc = "".join(doc)
    doc = normalizer.normalize(doc)
    print("normalized", doc)
    return doc


# removing stop words and frequently used words
def remove_stop_words(doc):
    return doc


# stemming (may be a bit tricky)
def stem(doc, stemmer):
    doc = doc.split(" ")
    for term_index in range(len(doc)):
        print("stemm")
        print(doc[term_index])
        print(stemmer.stem(doc[term_index]))
        doc[term_index] = stemmer.stem(doc[term_index])
    print("doc after stemming", doc)
    return doc


# call all the preprocessing here
def preprocessing(array_of_docs, with_stop_words=True):
    array_of_docs_preprocessed = []
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()

    for doc in array_of_docs:
        doc.content = normalize(doc.content, normalizer)
        doc.content = stem(doc.content, stemmer)
        # doc[0] = tokenize(doc[0])
        if with_stop_words:
            doc.content = remove_stop_words(doc.content)
        array_of_docs_preprocessed.append(doc)
    return array_of_docs_preprocessed


if __name__ == '__main__':


    tmp_arr = np.array(["ما اصلاح کتاب ها و استفاده از نیم‌فاصله پردازش را آسان مي كند"])
    tmp_arr2 = np.array(["اصلاح نويسه ها و استفاده از نیم‌فاصله پردازش را آسان مي كند"])

    array_of_docs = []
    document1 = Document(id=1, title="title", content=tmp_arr, url="url")
    document2 = Document(id=2, title="title2", content=tmp_arr2, url="url")

    array_of_docs.append(document1)
    array_of_docs.append(document2)

    prepro_arr = preprocessing(array_of_docs, with_stop_words=False)
    print(prepro_arr)
