# Author: Amirhoseein Rajabpour 9731085
# Information Retrieval project - Fall 2021
import lists
import preprocessing
import process_query
import tfidf
import word2vec
import kmeans
import knn
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
import pickle
import os


class Document:
    def __init__(self, id, title, content, url, topic):
        self.id = id
        self.title = title
        self.content = content
        self.url = url
        self.topic = topic
        self.term_scores = {}
        self.embeddings = []


def save_model(positional_index):
    positional_index_json = {}
    for term_str, term_object in positional_index.items():
        positional_index_json[term_str] = term_object.__dict__
    with open("positional_index_json.json", 'w', encoding='utf-8') as json_file:
        json.dump(positional_index_json, json_file, indent=4, ensure_ascii=False)


def load_model(file_name):
    positional_index = {}
    with open(file_name, 'r', encoding='utf-8') as json_file:
        positional_index_json = json.load(json_file)
    for term in positional_index_json:
        term_object = lists.Term(string=term)
        term_object.set_total_freq(positional_index_json[term]["total_freq"])
        term_object.set_pos_in_each_doc(positional_index_json[term]["pos_in_each_doc"])
        term_object.set_freq_in_each_doc(positional_index_json[term]["freq_in_each_doc"])
        positional_index[term] = term_object
    return positional_index


def plot_zipf(term_freq, with_stopwords):
    term_freq_keys = list(term_freq.keys())
    term_freq_values = list(term_freq.values())
    max_frequency = term_freq_values[0]
    plt.title("Zipf law with stopwords")

    if not with_stopwords:
        term_freq_keys = term_freq_keys[30:]
        term_freq_values = term_freq_values[30:]
        max_frequency = term_freq_values[0]
        plt.title("Zipf law without stopwords")

    l1 = []
    l2 = []
    l3 = []
    for i in term_freq_values:
        l3.append(math.log(i, 10))

    for i in range(len(term_freq_keys)):
        l1.append(math.log(i + 1, 10))
        l2.append(math.log(max_frequency / (i + 1), 10))
    plt.plot(l1, l2)
    plt.plot(l1, l3)
    plt.show()


def find_token_words_number(num_docs, collection, stemming_status):  # Heaps law
    collection_heap = preprocessing.preprocessing(collection, with_stemming=stemming_status)

    for num_doc in num_docs:
        total_words = 0
        for i in range(num_doc):
            total_words += len(collection_heap[i].content)

        positional_index_heaps = lists.create_positional_index(collection_heap[:num_doc])
        total_tokens = len(list(positional_index_heaps.keys()))
        total_words2 = 0
        for term in positional_index_heaps:
            total_words2 += positional_index_heaps[term].total_freq

        if stemming_status:
            print("with stemming")
        else:
            print("without stemming")
        print("in ", num_doc, total_tokens, total_words)
        print("*********************")


def save_doc_contents(collection):
    # pickle dump
    doc_contents = []
    for doc in collection:
        doc_contents.append(doc.content)
    with open('collection.obj', 'wb') as collection_file:
        pickle.dump(doc_contents, collection_file)


def load_and_process_50k_collection():
    docs_df_50k_1 = pd.read_excel("dataset/IR00_3_11k News.xlsx")
    docs_df_50k_2 = pd.read_excel("dataset/IR00_3_17k News.xlsx")
    docs_df_50k_3 = pd.read_excel("dataset/IR00_3_20k News.xlsx")

    frames = [docs_df_50k_1, docs_df_50k_2, docs_df_50k_3]
    df_50k = pd.concat(frames)

    collection_50k = []
    for index, row in df_50k.iterrows():
        document = Document(id=index, title='', content=row["content"], url=row["url"], topic=row['topic'])
        collection_50k.append(document)

    collection_50k = preprocessing.preprocessing(collection_50k, with_stemming=True)
    print('collection preprocessed')

    with open('collection50k.obj', 'wb') as coll50k_file:
        pickle.dump(collection_50k, coll50k_file)

    print('collection saved')
    return collection_50k


if __name__ == '__main__':

    clusters_dict = {}  # just for removing warning
    # read excel file (with pandas and numpy)
    # takes excel file as input --> outputs a numpy array
    docs_df = pd.read_excel("dataset/IR1_7k_news.xlsx")

    # create list of Document objects
    collection = []
    for index, row in docs_df.iterrows():
        document = Document(id=index, title=row["title"], content=row["content"], url=row["url"], topic='')
        collection.append(document)

    option = input("1) Create model\n2) Load previous model\n3) Zipf law\n4) Heaps law\n5) Initialize K-means\n6) Initialize KNN\n")
    if option == '1':
        # call functions for pre-processing
        collection = preprocessing.preprocessing(collection, with_stemming=True)

        # create positional index (and other necessary objects)
        positional_index = lists.create_positional_index(collection)
        save_model(positional_index)
        save_doc_contents(collection)

    elif option == '2':
        positional_index = load_model(file_name="positional_index_json.json")

    elif option == '3':
        positional_index = load_model(file_name="positional_index_json.json")
        term_freq = {}
        for term in positional_index.keys():
            term_freq[term] = positional_index[term].total_freq
        term_freq = dict(sorted(term_freq.items(), key=lambda item: item[1], reverse=True))     # term with higher frequencies have lower indexes

        plot_zipf(term_freq, with_stopwords=True)
        plot_zipf(term_freq, with_stopwords=False)

    elif option == '4':
        # num_docs = [500, 1000, 1500, 2000]
        num_docs = [len(collection)]
        find_token_words_number(num_docs, collection, stemming_status=False)
        find_token_words_number(num_docs, collection, stemming_status=True)
        exit()

    elif option == '5':
        if not os.path.isfile('collection50k.obj'):
            collection_50k = load_and_process_50k_collection()
        else:
            with open('collection50k.obj', 'rb') as collection50k_file:
                collection_50k = pickle.load(collection50k_file)

        my_model_path = "w2v models/my_w2v_model.model"
        hazm_model_path = "w2v models/w2v_150k_hazm_300_v2.model"
        positional_index = load_model(file_name="positional_index_json.json")
        collection = preprocessing.preprocessing(collection, with_stemming=True)
        collection_50k = word2vec.initialize_word2vec(my_model_path, positional_index, collection)
        clusters_dict = kmeans.initialize_kmeans(collection_50k, k=10)

    elif option == '6':
        if not os.path.isfile('collection50k.obj'):
            collection_50k = load_and_process_50k_collection()
        else:
            with open('collection50k.obj', 'rb') as collection50k_file:
                collection_50k = pickle.load(collection50k_file)

        collection = preprocessing.preprocessing(collection, with_stemming=True)
        collection_57k = knn.initialize_knn(collection_50k, collection, k=15)

        with open('collection57k.obj', 'wb') as coll57k_file:
            pickle.dump(collection_57k, coll57k_file)


    else:
        print("Wrong input!")
        exit()

    # some functions to handle clients queries
    selected_model = input("1) Binary model\n2) Tf-idf model\n3) Word2vec model\n4) K-means model\n5) KNN model\n")

    if selected_model == "1":
        print("query processing using binary model ...")
        query = input("Write your query:\n")
        query = preprocessing.preprocess_query(query)
        list_of_doc_titles, list_of_doc_ids = process_query.process_query(query, positional_index, collection)

        print("Results sorted by scores:")
        for l in range(len(list_of_doc_titles)):
            print("document id: ", list_of_doc_ids[l])
            print("title: ", list_of_doc_titles[l])
            print("********************************")

    elif selected_model == "2":
        collection = preprocessing.preprocessing(collection, with_stemming=True)
        print("query processing using tf-idf model ...")
        query = input("Write your query:\n")
        query = preprocessing.preprocess_query(query)
        first_K_pairs = tfidf.tf_idf(query, positional_index, collection)
        for doc in first_K_pairs:
            print("document id: ", doc.id)
            print("document title: ", doc.title)
            print("document score: ", first_K_pairs[doc])
            print("document url: ", doc.url)
            print("********************************")

    elif selected_model == "3":
        collection = preprocessing.preprocessing(collection, with_stemming=True)
        print("query processing using word2vec model ...")
        query = input("Write your query:\n")
        query = preprocessing.preprocess_query(query)

        # initialize word2vec model
        my_model_path = "w2v models/my_w2v_model.model"
        hazm_model_path = "w2v models/w2v_150k_hazm_300_v2.model"
        collection = word2vec.initialize_word2vec(my_model_path, positional_index, collection)

        # show results of query
        first_K_pairs = word2vec.query_word2vec(query, my_model_path, positional_index, collection)
        for doc in first_K_pairs:
            print("document id: ", doc.id)
            print("document title: ", doc.title)
            print("document score: ", first_K_pairs[doc])
            print("document url: ", doc.url)
            print("********************************")

    elif selected_model == "4":
        with open('kmeans_model.obj', 'rb') as kmeans_file:
            clusters_dict = pickle.load(kmeans_file)
        # for k in clusters_dict.keys():
        #     print(k.embedding)

        collection = preprocessing.preprocessing(collection, with_stemming=True)
        print("query processing using k-means model ...")
        query = input("Write your query:\n")
        query = preprocessing.preprocess_query(query)
        first_z_pairs = kmeans.search_kmeans(query, positional_index, collection, clusters_dict, b=1)
        for doc in first_z_pairs:
            print("document id: ", doc.id)
            print("document score: ", first_z_pairs[doc])
            print("document url: ", doc.url)
            print("********************************")

    elif selected_model == "5":
        pass
