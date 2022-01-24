import word2vec


def initialize_knn(collection_50k, collection_7k, k):
    # for every doc in collection 7k calculate the cos similarities for all 50k and then choose k NN
    for doc in collection_7k:
        doc_similarity_scores = {}
        for doc_in_50 in collection_50k:
            doc_similarity_scores[doc_in_50] = word2vec.cos_similarity_emb(doc.embeddings, doc_in_50.embeddings)

        doc_similarity_scores = dict(sorted(doc_similarity_scores.items(), key=lambda item: item[1], reverse=True))
        first_k_docs = {i: doc_similarity_scores[i] for i in list(doc_similarity_scores)[:k]}
        topics = [doc.topic for doc in first_k_docs]
        doc.topic = max(set(topics), key=topics.count)

    # return all docs (57k) with their topics
    return collection_50k + collection_7k


def search_knn():
    pass
