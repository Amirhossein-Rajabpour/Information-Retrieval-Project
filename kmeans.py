from word2vec import cos_similarity_emb
import random

def initialize_dict(collection_50k, clusters_dict, center_ids):
    for id in center_ids:
        clusters_dict[collection_50k.iloc[[id]].embeddings] = []   # is it correct? [id]
    return clusters_dict


def initialize_kmeans(collection_50k, k):
    clusters_dict = {}
    # TODO: randomly choose "k" docs as centers
    center_ids = random.sample(range(collection_50k.shape[0]), k)
    clusters_dict = initialize_dict(collection_50k, clusters_dict, center_ids)

    while True:
        tmp_dict = dict.fromkeys(clusters_dict.keys(), [])
        # TODO 1: calculate cosine similarity (distance) between each doc and centers
        for doc in collection_50k:
            highest_score = 0
            closest_cnt = ''
            for cnt in clusters_dict.keys():
                score = cos_similarity_emb(doc.embedding, cnt)
                if score > highest_score:
                    highest_score = score
                    closest_cnt = cnt

            # TODO 2: assign each doc to its nearest center
            tmp_dict[closest_cnt].append(doc)

        # TODO 3: calculate new centers
        for old_key, v in tmp_dict.items():
            h = sum(tmp_dict[old_key]) / len(tmp_dict[old_key])
            tmp_dict[h] = tmp_dict.pop(old_key)

        if tmp_dict.keys() == clusters_dict:
            break
        clusters_dict = tmp_dict

    # TODO: return the dictionary: keys are centers, values are docs in that cluster
    return clusters_dict


def search_kmeans(query, clusters_dict, b):
    # TODO: compare query vector with cluster centers (cosine similarity)

    # TODO: find "b" closest clusters

    # TODO: compare query vector with docs in "b" clusters

    # TODO: return top docs
    pass
