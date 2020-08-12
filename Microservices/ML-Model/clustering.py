import streamlit as st
import json
import pandas as pd
import numpy as np
import copy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as clr


# class for the clustering model
# handles the incoming data and wraps the scikit-learn implementation into the service

class Clustering:
    def __init__(self):
        # input data
        self.word_vectors = pd.DataFrame()
        self.ingredients = pd.DataFrame()

        # model parameter
        self.no_clusters = None

        # model result
        self.clustered_word_vectors = None
        self.cluster_reinheit = None

        # modelinstance
        self.MODEL = None
        
    # wrapper for the scikit-learn k-means clustering
    # assigns every word to a cluster and calculates the reinheit (pureness) of a cluster
    # stores the result in the modelinstance
    def run(self):
        cluster_data = self.word_vectors
        self.MODEL = KMeans(n_clusters = self.no_clusters, random_state=0)
        clustered = self.MODEL.fit(cluster_data[["x","y"]])
        #assigned_cluster = pd.DataFrame(data=clustered.labels_.tolist(),columns=["cluster"])
        cluster_data["cluster"] = clustered.labels_
        self.clustered_word_vectors = cluster_data
        self.calculate_reinheit()

    def load_data(self, data):
        try:
            word_vectors = pd.DataFrame(data["word_vectors"])
            self.word_vectors = word_vectors
        except Exception as e:
            print(e) 
            pass
        try:
            ingredients = pd.DataFrame(data["ingredients"])
            self.ingredients =  ingredients
        except Exception as e:
            print(e)
            pass

    def load_parameters(self, parameter_list):
        for parameter in parameter_list:
            self.__dict__[parameter["parameter"]] = int(parameter["value"])
        
    def get_data(self):
        return self.word_vectors
        
    def get_result(self, result):
        return self.__dict__[result]
    
    def get_description(self):
        word_vector_check = True
        if self.word_vectors.empty:
            word_vector_check = False

        description = {"word_vectors": word_vector_check,
                        "no_clusters": self.no_clusters}
        return description

    # calculates the reinheit (pureness) of a cluster, to determine, how high the probability of a "Bezeichnung" in a cluster is
    def calculate_reinheit(self):
        clusterlist = self.clustered_word_vectors["cluster"].unique()
        toplist = self.ingredients
        results = pd.DataFrame(index=[clusterlist], columns=["Zugeordnete Wörter", "Daraus Bezeichnungen", "Reinheit"])
        for cluster in clusterlist:
            c_frame= self.clustered_word_vectors.loc[self.clustered_word_vectors["cluster"]==cluster] 
            match_list = []
            for index, row in c_frame.iterrows():
                if row["labels"] in list(toplist["name"]):
                    match_list.append(row["labels"])
                
            results["Zugeordnete Wörter"][cluster] = len(c_frame)
            results["Daraus Bezeichnungen"][cluster] = len(match_list)
            results["Reinheit"][cluster] = round((len(match_list)/len(c_frame))*100,1)
        self.cluster_reinheit = results


# word_vectors = pd.read_csv("../../BaseData/word_vectors.csv", sep = "|", index_col=None)
# ingredients = pd.read_csv("../../BaseData/top_ingredients.csv", sep = "|", index_col=0)
# data = {"word_vectors": word_vectors,
#         "ingredients": ingredients}
# clusterer = Clustering()
# clusterer.load_data(data)

# parameters={"no_clusters":8}
# parameter_list = []
# for parameter, value in parameters.items():
#     parameter_list.append({"parameter": parameter, "value": value} )

# clusterer.load_parameters(parameter_list)

# clusterer.run()

# st.write(clusterer.clustered_word_vectors["cluster"])
# clusterer.calculate_reinheit()
# st.write(clusterer.cluster_reinheit)






