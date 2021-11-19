import hazm

# TODO: extracting Tokens from data
def tokenize(doc, sent_tokenizer, word_tokenizer):
    return doc


# TODO: normalizing texts
def normalize(doc, normalizer):
    return doc


# TODO: removing stop words and frequently used words
def remove_stop_words(doc):
    return doc


# TODO: stemming (may be a bit tricky)
def stem(doc, stemmer):
    return doc


# TODO: call all the preprocessing here
def preprocessing(array_of_docs):

    array_of_docs_preprocessed = []
    sent_tokenizer = hazm.SentenceTokenizer()
    word_tokenizer = hazm.WordTokenizer()
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()


    for doc in array_of_docs:
        doc[0] = tokenize(doc[0], sent_tokenizer, word_tokenizer)
        doc[2] = tokenize(doc[2], sent_tokenizer, word_tokenizer)

        doc[0] = normalize(doc[0], normalizer)
        doc[2] = normalize(doc[2], normalizer)

        doc[0] = remove_stop_words(doc[0])
        doc[2] = remove_stop_words(doc[2])

        doc[0] = stem(doc[0], stemmer)
        doc[2] = stem(doc[2], stemmer)

        array_of_docs_preprocessed.append(doc)
    return array_of_docs_preprocessed
