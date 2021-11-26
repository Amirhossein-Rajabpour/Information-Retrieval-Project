# Author: Amirhoseein Rajabpour 9731085
# Information Retrieval project - Fall 2021
import lists
import preprocessing
import pandas as pd


class Document:
    def __init__(self, id, title, content, url):
        self.id = id
        self.title = title
        self.content = content
        self.url = url


if __name__ == '__main__':
    # read excel file (with pandas and numpy)
    # takes excel file as input --> outputs a numpy array
    docs_df = pd.read_excel("IR1_7k_news.xlsx")

    # create list of Document objects
    collection = []
    for index, row in docs_df.iterrows():
        document = Document(id=index, title=row["title"], content=row["content"], url=row["url"])
        collection.append(document)

    # call functions for pre-processing
    collection = preprocessing.preprocessing(collection, with_stop_words=True)

    # create positional index (and other necessary objects)
    positional_index = lists.create_positional_index(collection)

    # TODO 4: write some functions to handle clients queries
    #         takes query --> outputs a list of related doc titles
    print("query processing")
