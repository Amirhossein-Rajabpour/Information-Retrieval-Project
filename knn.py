import word2vec

def initialize_knn(collection_50k, collection_7k, k):
    # for every doc in collection 7k calculate the cos similarities for all 50k and then choose k NN
    print('initializing knn ...')
    i = 0
    for doc in collection_7k:
        if i % 500 == 0:
            print('doc num: ', i)

        doc_similarity_scores = {}
        for doc_in_50 in collection_50k:
            doc_similarity_scores[doc_in_50] = word2vec.cos_similarity_emb(doc.embeddings, doc_in_50.embeddings)

        doc_similarity_scores = dict(sorted(doc_similarity_scores.items(), key=lambda item: item[1], reverse=True))
        first_k_docs = {i: doc_similarity_scores[i] for i in list(doc_similarity_scores)[:k]}
        topics = [doc.topic for doc in first_k_docs]
        doc.topic = max(set(topics), key=topics.count)
        i += 1

    # return all docs (57k) with their topics
    return collection_50k + collection_7k


def search_knn(query_embedding, collection_57k, topic):
    # compare query embedding with docs in that category
    similarities = {}
    docs_topic = [doc for doc in collection_57k if doc.topic == topic]
    print(len(docs_topic), 'docs in this topic')
    for doc in docs_topic:
        similarities[doc] = word2vec.cos_similarity_emb(query_embedding, doc.embeddings)

    # return top "z" docs
    # sort scores dictionary
    similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
    z = 5
    first_z_pairs = {i: similarities[i] for i in list(similarities)[:z]}
    return first_z_pairs
