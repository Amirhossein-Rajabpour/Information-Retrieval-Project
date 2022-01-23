def initialize_kmeans(collection_50k, k):
    clusters_dict = {}
    # TODO: randomly choose "k" docs as centers

    # TODO 1: calculate cosine similarity (distance) between each doc and centers

    # TODO 2: assign each doc to its nearest center

    # TODO 3: calculate new centers

    # TODO: repeat TODOs 1 to 3

    # TODO: return a dictionary: keys are centers, values are docs in that cluster
    return clusters_dict
    pass

def search_kmeans(query, clusters_dict, b):
    # TODO: compare query vector with cluster centers (cosine similarity)

    # TODO: find "b" closest clusters

    # TODO: compare query vector with docs in "b" clusters

    # TODO: return top docs
    pass
