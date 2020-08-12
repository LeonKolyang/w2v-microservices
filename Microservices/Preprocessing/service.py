import pandas as pd
import streamlit as st
import copy

class Preprocessor:
    def __init__(self, prep_id):
        self.prep_id = prep_id
        
        # input data
        self.dataset = None
        self.ingredients = None

        # midcalculation result
        self.processed_data = None

        # preprocessing results
        self.corpus = None
        self.zutatenverzeichnis = None
        self.stemmed_ingredients = None

        # list of forbidden signs for the stemmer
        # TODO: implement function to alternate list
        self.sign_list = [",", "(", ")", "/", ".", "+","\"", ":","-","„","“","&"]

    def load_dataset(self, dataset):
        self.dataset = dataset
        self.processed_data = copy.deepcopy(dataset) 
        return self.dataset

    def load_ingredients(self, ingredients):
        self.ingredients = ingredients
        return self.ingredients

    # TODO: cleanup get methods
    def get_corpus(self):
        return self.corpus

    def get_zutatenverzeichnis(self):
        return self.zutatenverzeichnis
    
    def get_dataset(self):
        return self.dataset

    def get_processed_data(self):
        return self.processed_data

    def get_ingredients(self):
        return self.ingredients

    def get_stemmed_ingredients(self):
        return self.stemmed_ingredients
    
    def get_attribute(self, attribute):
        attribute = self.__dict__[attribute]
        return attribute

    # drop duplicates and sort a dataframe
    def reduce_duplicates(self, data_in=None, column = "name"):
        data = self.processed_data if data_in is None else data_in 
        data = data.sort_values(column)
        data = data.drop_duplicates(subset = column, keep=False)
        if data_in is None: self.processed_data = data
        return data

    # helperfunction to match the selected ingredients with the dataset
    def match_helper(self, row, ingredients):
        list_row = row["name"].split(" ")
        ing_list = []
        for word in list_row:
            if word in ingredients:
                ing_list.append(word)
        return ing_list

    # wrapper for the match_helper function
    def match_ingredients(self, dataDf=None, column = "ingredient"):
        if not(dataDf): data = self.processed_data
        ingredients = list(self.ingredients["name"])
        data[column] = data.apply(lambda row: self.match_helper(row, ingredients), axis=1)
        if not(dataDf): self.processed_data = data
        return data
    
    # helperfunction to select the value of the row in a singlecolumned dataframe
    def cut_helper(self, x):   
        return x[0]

    # delete empty or too short (<1) entries in the dataset
    def cut_lists(self, column = "ingredient", data=None):
        self.processed_data = self.processed_data[self.processed_data[column].str.len()==1 ]
        self.processed_data.loc[:,(column)] = self.processed_data[column].apply(self.cut_helper)
        return self.processed_data
    
    # create the corpus from the given dataset
    def create_corpus(self, data=None):
        if not(data): data = self.processed_data[["name", "unit"]]
        corpus = []
        columns = list(data.columns)
        for index, row in data.iterrows():
            column_list = [row[col] for col in columns]
            for column in column_list:
                if type(column) == str:
                    word_list = column.split()
                    for word in word_list:
                        if word not in corpus:
                            corpus.append(word)
        self.corpus = pd.DataFrame(data=corpus, columns=["corpus"])
        return self.corpus
    
    #create the zutatenverzeichnis from the given dataset
    def create_zutatenverzeichnis(self, data=None):
        if not(data): data = self.processed_data[["name", "unit"]]
        zutatenverzeichnis = []
        columns = list(data.columns)
        for index, row in data.iterrows():
            column_list = [row[col] for col in columns]
            corp = ""
            for column in column_list:
                if type(column) == str:
                    corp += column + " "
            zutatenverzeichnis.append(corp)
        self.zutatenverzeichnis = pd.DataFrame(data=zutatenverzeichnis, columns=["name"])
        return self.zutatenverzeichnis

    # delete the signs specified in the sign_list from the given dataset
    def clear_signs(self, data, sign_list=None):
        if not(sign_list): sign_list = self.sign_list

        def stripper(row, sign_list):
            stripped_row = ""
            for element in row:
                for sign in element:
                    if sign not in sign_list:
                        stripped_row += sign
            #if row == stripped_row: clean_check=True
            print(stripped_row)
            row[0] = stripped_row
            return row

        data.apply(lambda row: stripper(row, sign_list), axis=1)
        return data 

    # wrapper for deleting a column from dataframe
    def strip_column(self, column, data_in=None):
        data = self.processed_data if data_in is None else data_in 
        data = data.drop(column, axis=1)
        if data_in is None: self.processed_data = data
        return data

    # helperfunction to check fpr missing attributes of the preprocessor
    def check_attributes(self):
        empty_attributes = [attribute for attribute, value in self.__dict__.items() if value is None]
        return empty_attributes        

    # function to call the whole preprocessing in one task
    def preprocess(self, dataset, ingredients):  

        self.load_dataset(dataset)

        self.load_ingredients(ingredients)

        self.strip_column("amount")

        self.processed_data

        self.reduce_duplicates()

        self.match_ingredients()

        self.cut_lists()

        self.create_corpus()

        self.create_zutatenverzeichnis() 

        self.corpus = self.clear_signs(data=self.corpus)
        self.zutatenverzeichnis = self.clear_signs(data=self.zutatenverzeichnis)

        self.reduce_duplicates(data_in=self.corpus, column="corpus") 


