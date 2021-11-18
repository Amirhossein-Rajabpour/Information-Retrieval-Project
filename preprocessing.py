# TODO: extracting Tokens from data
def tokenize(doc):
    return doc


# TODO: normalizing texts
def normalize(doc):
    return doc


# TODO: removing stop words and frequently used words
def remove_stop_words(doc):
    return doc


# TODO: stemming (may be a bit tricky)
def stem(doc):
    return doc


# TODO: call all the preprocessing here
def preprocessing(array_of_docs):
    array_of_docs_preprocessed = []
    for doc in array_of_docs:
        doc = tokenize(doc)
        doc = normalize(doc)
        doc = remove_stop_words(doc)
        doc = stem(doc)
        array_of_docs_preprocessed.append(doc)
    return array_of_docs_preprocessed