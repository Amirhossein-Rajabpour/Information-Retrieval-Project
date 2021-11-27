# Author: Amirhoseein Rajabpour 9731085
# Information Retrieval project - Fall 2021
import lists
import preprocessing
import process_query
import pandas as pd
import json
import math
import matplotlib.pyplot as plt


class Document:
    def __init__(self, id, title, content, url):
        self.id = id
        self.title = title
        self.content = content
        self.url = url


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


if __name__ == '__main__':

    # read excel file (with pandas and numpy)
    # takes excel file as input --> outputs a numpy array
    docs_df = pd.read_excel("IR1_7k_news.xlsx")

    # create list of Document objects
    collection = []
    for index, row in docs_df.iterrows():
        document = Document(id=index, title=row["title"], content=row["content"], url=row["url"])
        collection.append(document)

    option = input("1) Create model\n2) Load previous model\n3) Zipf law\n4) Heaps law\n")
    if option == '1':
        # call functions for pre-processing
        collection = preprocessing.preprocessing(collection, with_stop_words=True)

        # create positional index (and other necessary objects)
        positional_index = lists.create_positional_index(collection)
        save_model(positional_index)

    elif option == '2':
        positional_index = load_model(file_name="positional_index_json.json")
    elif option == '3':
        print("Zipf law")
        positional_index = load_model(file_name="positional_index_json.json")
        term_freq = {}
        for term in positional_index.keys():
            term_freq[term] = positional_index[term].total_freq
        term_freq = dict(sorted(term_freq.items(), key=lambda item: item[1], reverse=True))     # term with higher frequencies have lower indexes

        plot_zipf(term_freq, with_stopwords=True)

        plot_zipf(term_freq, with_stopwords=False)

    elif option == '4':
        print("Heaps law")
    else:
        print("Wrong input!")
        exit()

    # write some functions to handle clients queries
    print("query processing...")
    query = input("Write your query:\n")
    list_of_doc_titles, list_of_doc_ids = process_query.process_query(query, positional_index, collection)

    print("Results sorted by scores:")
    for l in range(len(list_of_doc_titles)):
        print("document id: ", list_of_doc_ids[l])
        print("title: ", list_of_doc_titles[l])
        print("********************************")
