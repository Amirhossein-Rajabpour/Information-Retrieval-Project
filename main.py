# Author: Amirhoseein Rajabpour 9731085
# Information Retrieval project - Fall 2021
import preprocessing
import pandas as pd

if __name__ == '__main__':
    # TODO 1: read excel file (with pandas and numpy)
    #         takes excel file as input --> outputs a numpy array
    docs_df = pd.read_excel("IR1_7k_news.xlsx")
    array_of_docs = docs_df.to_numpy()
    array_of_docs = preprocessing.preprocessing(array_of_docs)

    # TODO 2: call functions for pre-processing
    #         takes numpy array as input --> outputs a pre-processed numpy array

    # TODO 3: create positional postings lists (and other necessary objects)
    #         takes a pre-processed numpy array in --> outputs positional postings lists

    # TODO 4: write some functions to handle clients queries
    #         takes query --> outputs a list of related doc titles
