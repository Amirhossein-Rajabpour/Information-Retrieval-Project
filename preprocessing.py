import hazm
import numpy as np


# TODO: extracting Tokens from data
def tokenize(doc):
    print("enter tokenize")
    doc = "".join(doc)
    doc = hazm.sent_tokenize(doc)
    doc = "".join(doc)  # TODO: here we remove the effect of sent_tokenize because we merge them all again in an array
    doc = hazm.word_tokenize(doc)
    print("doc", doc)
    return doc


# TODO: normalizing texts
def normalize(doc, normalizer):
    doc = "".join(doc)
    doc = normalizer.normalize(doc)
    print("normalized", doc)
    return doc


# TODO: removing stop words and frequently used words
def remove_stop_words(doc):
    return doc


# TODO: stemming (may be a bit tricky)
def stem(doc, stemmer):
    doc = doc.split(" ")
    for term_index in range(len(doc)):
        print("stemm")
        print(doc[term_index])
        print(stemmer.stem(doc[term_index]))
        doc[term_index] = stemmer.stem(doc[term_index])
    print("doc after stemming", doc)
    return doc


# TODO: call all the preprocessing here
def preprocessing(array_of_docs, with_stop_words=True):
    array_of_docs_preprocessed = []
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()

    for doc in array_of_docs:
        doc[0] = normalize(doc[0], normalizer)
        doc[2] = normalize(doc[2], normalizer)

        doc[0] = stem(doc[0], stemmer)
        doc[2] = stem(doc[2], stemmer)

        # doc[0] = tokenize(doc[0])
        # doc[2] = tokenize(doc[2])

        if with_stop_words:
            doc[0] = remove_stop_words(doc[0])
            doc[2] = remove_stop_words(doc[2])

        array_of_docs_preprocessed.append(doc)
    return array_of_docs_preprocessed


if __name__ == '__main__':
    tmp_data_for_preprocessing = np.array([[["اصلاح کتاب ها و استفاده از نیم‌فاصله پردازش را آسان مي كند"], [0], ["ما هم برای وصل کردن آمدیم! ولی برای پردازش، جدا بهتر نیست؟"]],
                                           [[], [0], []]],
                                          dtype=object, )
    arr = preprocessing(tmp_data_for_preprocessing, with_stop_words=True)
    print(arr)
