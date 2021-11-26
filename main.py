# Author: Amirhoseein Rajabpour 9731085
# Information Retrieval project - Fall 2021
import lists
import preprocessing
import process_query
import pandas as pd
import json


class Document:
    def __init__(self, id, title, content, url):
        self.id = id
        self.title = title
        self.content = content
        self.url = url


def find_titles(list_of_doc_ids, collection):
    list_of_doc_titles = []
    print(list_of_doc_ids)
    for doc in collection:
        if doc.id in list_of_doc_ids:
            list_of_doc_titles.append(doc.title)
    return list_of_doc_titles


def save_model(positional_index):
    positional_index_json = {}
    for term_str, term_object in positional_index.items():
        positional_index_json[term_str] = term_object.__dict__
    with open("positional_index_json.json", 'w', encoding='utf-8') as json_file:
        json.dump(positional_index_json, json_file, indent=4, ensure_ascii=False)


def load_model(file_name):
    pass


if __name__ == '__main__':

    # read excel file (with pandas and numpy)
    # takes excel file as input --> outputs a numpy array
    docs_df = pd.read_excel("IR1_7k_news.xlsx")

    # create list of Document objects
    collection = []
    for index, row in docs_df.iterrows():
        document = Document(id=index, title=row["title"], content=row["content"], url=row["url"])
        collection.append(document)

    option = input("1) Create model\n2) Load previous model\n")
    if option == '1':
        # call functions for pre-processing
        collection = preprocessing.preprocessing(collection, with_stop_words=True)

        # create positional index (and other necessary objects)
        positional_index = lists.create_positional_index(collection)
        save_model(positional_index)

    elif option == '2':
        positional_index = load_model("positional_index_json")
    else:
        print("Wrong input!")
        exit()

    # write some functions to handle clients queries
    print("query processing")
    query = input("Write your query:\n")
    list_of_doc_ids = process_query.process_query(query, positional_index)
    list_of_doc_titles = find_titles(list_of_doc_ids, collection)

    print("Results:\n")
    for l in list_of_doc_titles:
        print(l)