import random
import word2vec
import numpy as np
from gensim.models import Word2Vec
import pickle

class Center:
    def __init__(self, embedding):
        self.embedding = embedding


def initialize_dict(collection_50k, clusters_dict, center_ids):
    for id in center_ids:
        center = Center(embedding=collection_50k[id].embeddings)
        clusters_dict[center] = []
    return clusters_dict


def termination_condition(a1, a2):
    terminate = True
    for j in range(len(a1)):
        for i in range(a1[j].embedding.shape[0]):
            if a1[j].embedding[i] > a2[j].embedding[i] * 1.1 or a1[j].embedding[i] < a2[j].embedding[i] * 1.1:
                terminate = False
    return terminate


def initialize_kmeans(collection_50k, k):
    clusters_dict = {}
    # randomly choose "k" docs as centers
    center_ids = random.sample(range(len(collection_50k)), k)
    clusters_dict = initialize_dict(collection_50k, clusters_dict, center_ids)

    print('dict initialized')

    MAX_ITERATION = 700
    for it in range(MAX_ITERATION):
        print('iteration ', it)
        tmp_dict = dict.fromkeys(clusters_dict.keys(), [])
        # calculate cosine similarity (distance) between each doc and centers
        for doc in collection_50k:
            highest_score = -1
            closest_cnt = ''
            for cnt in clusters_dict.keys():
                score = word2vec.cos_similarity_emb(doc.embeddings, cnt.embedding)
                if score > highest_score:
                    highest_score = score
                    closest_cnt = cnt

            # assign each doc to its nearest center
            tmp_dict[closest_cnt].append(doc)

        # calculate new centers
        for cnt in tmp_dict.keys():
            embeddings = [doc.embeddings for doc in tmp_dict[cnt]]
            cnt.embedding = sum(embeddings) / len(embeddings)

        if termination_condition(list(tmp_dict.keys()), list(clusters_dict.keys())):
            break
        clusters_dict = tmp_dict

    print('finish iteration')

    # save model
    with open('kmeans_model.obj', 'wb') as kmeans_file:
        pickle.dump(clusters_dict, kmeans_file)

    print('file saved')

    # return the dictionary: keys are centers, values are docs in that cluster
    return clusters_dict


def search_kmeans(query_embedding, clusters_dict, b=2):
    # compare query vector with cluster centers (cosine similarity)
    cnt_scores = {}
    for cnt in clusters_dict.keys():
        score = word2vec.cos_similarity_emb(query_embedding, cnt.embedding)
        cnt_scores[cnt] = score

    # find "b" closest clusters
    cnt_scores = dict(sorted(cnt_scores.items(), key=lambda item: item[1], reverse=True))
    cnt_scores_list = list(cnt_scores)

    # compare query vector with docs in "b" clusters
    similarities = {}  # {doc: similarity, ...}
    for i in range(1, b+1):
        for doc in clusters_dict[cnt_scores_list[i]]:
            similarities[doc] = word2vec.cos_similarity_emb(query_embedding, doc.embeddings)

    # return top "z" docs
    # sort scores dictionary
    similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
    z = 5
    first_z_pairs = {i: similarities[i] for i in list(similarities)[:z]}
    return first_z_pairs
