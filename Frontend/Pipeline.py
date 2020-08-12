import pandas as pd
import streamlit as st
import numpy as np
import requests 
import json

zutatenverzeichnis = pd.read_csv("BaseData/zutatenverzeichnis.csv", header=None,sep="|")
zutatenverzeichnis.columns=["name"]
top_ingredients = pd.read_csv("BaseData/top_ingredients.csv", header=0, index_col=0)
top_ingredients = top_ingredients.reset_index(drop=True)

parameters = {"iterations": 10,
                "window_size" : 2,
                "dimensions" : 300,
                "min" : 0,
                "neg" : 0}
# parameters = {"iterations": st.number_input("iterations", value=10),
#                 "window_size" : st.number_input("window_size", value=2),
#                 "dimensions" : st.number_input("dimensions", value=300),
#                 "min" : st.number_input("min", value=5),
#                 "neg" : st.number_input("neg", value=5)}

parameter_list = []
for parameter, value in parameters.items():
    parameter_list.append({"parameter": parameter, "value": value} )

parameters={"no_clusters": 8}
#parameters={"no_clusters":st.number_input("Cluster", value=8)}

clparameter_list = []
for parameter, value in parameters.items():
    clparameter_list.append({"parameter": parameter, "value": value} )

if st.button("test start"):
    url = "https://w2v-mlmodels.herokuapp.com/create_model/word2vec/new"
    put = requests.post(url)
    st.write(put)

if st.button("Start Service"):
    url = "https://w2v-mlmodels.herokuapp.com/create_model/word2vec/new"
    put = requests.post(url)

if st.button("load"):
    zutatenverzeichnis = zutatenverzeichnis.to_dict()
    data_dict = {"data": {"zutatenverzeichnis": zutatenverzeichnis}}
    data = json.dumps(data_dict)
    requests.put( "https://w2v-mlmodels.herokuapp.com/load_model_data/word2vec", data=data)
if st.button("param"):
    data = parameter_list
    data = json.dumps(data)
    requests.put("https://w2v-mlmodels.herokuapp.com/load_model_parameters/word2vec", data=data)

    mlurl = "https://w2v-mlmodels.herokuapp.com/run_model/word2vec"
    put = requests.put(mlurl)
    
    url = "https://w2v-mlmodels.herokuapp.com/get_result/word2vec/all"
    get = requests.get(url)
    word_vectors = json.loads(get.json())
    word_vectors = pd.DataFrame(word_vectors)

    requests.post("https://w2v-mlmodels.herokuapp.com/create_model/clustering/new")
    word_vectors = word_vectors.to_dict()
    top_ingredients = top_ingredients.to_dict()
    data_dict = {"data": {"word_vectors": word_vectors,
                            "ingredients": top_ingredients}}
    data = json.dumps(data_dict)
    requests.put( "https://w2v-mlmodels.herokuapp.com/load_model_data/clustering", data=data)

    mlurl = "https://w2v-mlmodels.herokuapp.com/load_model_parameters/clustering"
    params = clparameter_list
    params = json.dumps(params)
    put = requests.put(mlurl, data=params)

    mlurl = "https://w2v-mlmodels.herokuapp.com/run_model/clustering"
    put = requests.put(mlurl)

parameters={"no_clusters":st.number_input("Cluster", value=8)}
clparameter_list = []
for parameter, value in parameters.items():
    clparameter_list.append({"parameter": parameter, "value": value} )

if st.button("Reload Cluster Parameters"):
    mlurl = "https://w2v-mlmodels.herokuapp.com/load_model_parameters/clustering"
    params = clparameter_list
    params = json.dumps(params)
    put = requests.put(mlurl, data=params)

    mlurl = "https://w2v-mlmodels.herokuapp.com/run_model/clustering"
    put = requests.put(mlurl)


new_phrase = st.text_input("New Phrase")

if st.button("Evaluate new Phrase"):
    data = {"data":{"new_phrase": new_phrase.split()}}
    data = json.dumps(data)
    new = requests.get( "https://w2v-mlmodels.herokuapp.com/evaluate_new_phrase/word2vec", data=data)
    new = json.loads(new.json())
    new = pd.DataFrame(new)
    new.to_csv("new_phrase.csv", index=None, sep="|")
    new = new.rename(columns={"reinheit":"Bezeichnung", "labels":"Wort"})
    st.write("Als Bezeichnung identifiziert:")
    st.write(new.loc[new["Bezeichnung"]==new["Bezeichnung"].max()]["Wort"])
    st.write("Sonstige Zuordnungen")
    st.write(new[["Wort", "Bezeichnung"]])

