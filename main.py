# Author: Amirhoseein Rajabpour 9731085
# Information Retrieval project - Fall 2021
import lists
import pandas as pd
# import preprocessing

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
    array_of_docs = []
    for index, row in docs_df.iterrows():
        document = Document(id=index, title=row["title"], content=row["content"], url=row["url"])
        array_of_docs.append(document)

    # TODO 2: call functions for pre-processing
    #         takes numpy array as input --> outputs a pre-processed numpy array
    # array_of_docs = preprocessing.preprocessing(array_of_docs, with_stop_words=True)

    # TODO 3: create positional index (and other necessary objects)
    #         takes a pre-processed numpy array in --> outputs positional index
    positional_index = lists.create_positional_index(array_of_docs)

    # TODO 4: write some functions to handle clients queries
    #         takes query --> outputs a list of related doc titles
