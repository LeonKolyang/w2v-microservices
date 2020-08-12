#from Microservices.Preprocessing import service

import streamlit as st
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.colors as clr 


dataset = pd.read_csv("BaseData/rewe_ingredients.csv", header=0, index_col=None)
top_ingredients = pd.read_csv("BaseData/top_ingredients.csv", header=0, index_col=0)
top_ingredients = top_ingredients.reset_index(drop=True)
#top_ingredients.to_csv("BaseData/top_ingredients.csv",header=True, index=False)

st.subheader("Preprocessing")
url = st.selectbox("URL",options=["http://localhost:8000/get_active_preprocessors",
                                    "http://localhost:8000/get_ingredients/0",
                                    "http://localhost:8000/get_dataset/0",
                                     "http://localhost:8000/get_result/0/processed_data",
                                     "http://localhost:8000/get_result/0/corpus",
                                     "http://localhost:8000/get_result/0/zutatenverzeichnis",

                                    "http://localhost:8000/run_preprocessing/0"])

if st.button("Post data"):
    url = "http://localhost:8000/create_preprocessor"
    dataset_json = dataset.to_json(orient='records')  
    post = requests.post(url, data = dataset_json)
    st.write(post.text)

if st.button("Post ings"):
    url = "http://localhost:8000/load_ingredients/0"
    dataset_json = top_ingredients.to_json(orient='records')  
    post = requests.put(url, data = dataset_json)
    st.write(post.text)

# get dataset, ingredients, processed_data, corpus or zutatenverzeichnis
if st.button("GetP"):
    get = requests.get(url)#l, params={"prep_id":0})
    resp = json.loads(get.json())
    resp = pd.DataFrame(resp)
    st.write(resp)

if st.button("Run"):
    put = requests.put(url)
    st.write(put.text)


if st.button("Save"):
    get = requests.get(url)#l, params={"prep_id":0})
    resp = json.loads(get.json())
    resp = pd.DataFrame(resp)
    resp.to_csv("BaseData/zutatenverzeichnis.csv", sep="|", header=True, index=False)

st.subheader("Word2Vec")
zutatenverzeichnis = pd.read_csv("BaseData/zutatenverzeichnis.csv", header=None,sep="|")
zutatenverzeichnis.columns=["name"]


# mlurl = st.selectbox("MLURL", options=["http://localhost:8000/create_model/word2vec",
#                                         "http://localhost:8000/load_model_data/word2vec",
#                                         "http://localhost:8000/load_model_parameters/word2vec",
#                                         "http://localhost:8000/run_model/word2vec",
#                                         "http://localhost:8000/run_model_background/word2vec",
#                                         "http://localhost:8000/get_results/word2vec",
#                                         "http://localhost:8000/get_model_description/word2vec"])


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


#data = st.selectbox("Data", options=["z", "p"])


if st.button("Prepare"):
    requests.post("http://localhost:8000/create_model/word2vec/new")
    
    zutatenverzeichnis = zutatenverzeichnis.to_dict()
    data_dict = {"data": {"zutatenverzeichnis": zutatenverzeichnis}}
    data = json.dumps(data_dict)
    requests.put( "http://localhost:8000/load_model_data/word2vec", data=data)
    data = parameter_list
    data = json.dumps(data)
    requests.put("http://localhost:8000/load_model_parameters/word2vec", data=data)
    get = requests.get("http://localhost:8000/get_model_description/word2vec")
    resp = dict(get.json())
    st.write(get)



# if st.button("Load Parameters"):
#     mlurl = "http://localhost:8000/load_model_parameters/word2vec"
#     data = parameter_list
#     data = json.dumps(data)
#     put = requests.put(mlurl, data=data)
#     get = requests.get("http://localhost:8000/get_model_description/word2vec")
#     resp = dict(get.json())
#     st.write(resp)

if st.button("Run Model"):
    mlurl = "http://localhost:8000/run_model/word2vec"
    put = requests.put(mlurl)
    st.write(put)

if st.button("Get Results"):
    mlurl = "http://localhost:8000/get_result/word2vec/all"
    get = requests.get(mlurl)
    resp = json.loads(get.json())
    resp = pd.DataFrame(resp)
    plt.scatter(resp.x, resp.y)
    st.pyplot()
    resp.to_csv("BaseData/word_vectors.csv", sep="|", index=False)

    st.write(resp)

if st.button("Save Model"):
    url = "http://localhost:8000/save_model/word2vec"
    put = requests.put(url)
    st.write(put)

if st.button("Load old Model"):
    url = "http://localhost:8000/create_model/word2vec/old"
    put = requests.post(url)
    st.write(put)

st.subheader("Clustering")
parameters={"no_clusters":st.number_input("Cluster", value=8)}

clparameter_list = []
for parameter, value in parameters.items():
    clparameter_list.append({"parameter": parameter, "value": value} )

word_vectors = pd.read_csv("BaseData/word_vectors.csv", sep = "|", index_col=None)

data = word_vectors
data = data.to_dict() 
pdict = pd.DataFrame(data)
data_dict = {"data": data}
data = json.dumps(data_dict)

data = json.loads(data)["data"]
wordlist_df = pd.DataFrame(data)


if st.button("clPrepare"):
    requests.post("http://localhost:8000/create_model/clustering/new")
    word_vectors = word_vectors.to_dict() 
    top_ingredients = top_ingredients.to_dict()
    data_dict = {"data": {"word_vectors": word_vectors,
                            "ingredients": top_ingredients}}
    data = json.dumps(data_dict)
    requests.put( "http://localhost:8000/load_model_data/clustering", data=data)
    mlurl = "http://localhost:8000/load_model_parameters/clustering"
    params = clparameter_list
    params = json.dumps(params)
    put = requests.put(mlurl, data=params)
    get = requests.get("http://localhost:8000/get_model_description/clustering")
    resp = dict(get.json())
    st.write(resp)




# if st.button("Load clParameters"):
#     mlurl = "http://localhost:8000/load_model_parameters/clustering"
#     params = clparameter_list
#     params = json.dumps(params)
#     put = requests.put(mlurl, data=params)
#     get = requests.get("http://localhost:8000/get_model_description/clustering")
#     resp = dict(get.json())
#     st.write(resp)

if st.button("Run clModel"):
    mlurl = "http://localhost:8000/run_model/clustering"
    put = requests.put(mlurl)
    st.write(put)

if st.button("Get Results Cluster Visualization"):
    mlurl = "http://localhost:8000/get_result/clustering/clustered_word_vectors"
    get = requests.get(mlurl)
    resp = get.json()
    clresp = pd.read_json(resp,orient="records")

 
    data = clresp

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    cluster_colors = {i: colors[i] for i in range(6)}

    st.write(data.sort_values("x"))


    #for cluster, color in cluster_colors.items():
    #    plt.scatter(data.loc[data["cluster"] == cluster]["x"], data.loc[data["cluster"] == cluster]["y"], c= clr.to_rgba(color, 0.3))
    #plt.scatter(data.x, data.y)

    #st.pyplot()
    #data.to_csv("clustered_words.csv", sep="|", index=None)

if st.button("Get Results Cluster Auswertung"):
    mlurl = "http://localhost:8000/get_result/clustering/cluster_reinheit"
    get = requests.get(mlurl)
    resp = get.json()
    clresp = pd.read_json(resp,orient="records")
    st.write(clresp.sort_values("Reinheit", ascending=False))
    clresp.to_csv("reinheit.csv", index=None, sep="|")


    data = clresp


st.subheader("New Phrase")

new_phrase = "ein Liter Tomaten"

if st.button("Add phrase Debug"):
    url = "http://localhost:8000/create_model/word2vec/old"
    put = requests.post(url)

    requests.post("http://localhost:8000/create_model/clustering/new")
    word_vectors = word_vectors.to_dict() 
    top_ingredients = top_ingredients.to_dict()
    data_dict = {"data": {"word_vectors": word_vectors,
                            "ingredients": top_ingredients}}
    data = json.dumps(data_dict)
    requests.put( "http://localhost:8000/load_model_data/clustering", data=data)
    mlurl = "http://localhost:8000/load_model_parameters/clustering"
    params = clparameter_list
    params = json.dumps(params)
    put = requests.put(mlurl, data=params)
    get = requests.get("http://localhost:8000/get_model_description/clustering")
    resp = dict(get.json())

    mlurl = "http://localhost:8000/run_model/clustering"
    put = requests.put(mlurl)

    data = {"data":{"new_phrase": new_phrase.split()}}
    data = json.dumps(data)
    new = requests.get( "http://localhost:8000/evaluate_new_phrase/word2vec", data=data)
    new = json.loads(new.json())
    new = pd.DataFrame(new)
    new.to_csv("new_phrase.csv", index=None, sep="|")
    st.write(new)

new_phrase = st.text_input("New Phrase")
if st.button("Add phrase"):
    data = {"data":{"new_phrase": new_phrase.split()}}
    data = json.dumps(data)
    new = requests.get( "http://localhost:8000/evaluate_new_phrase/word2vec", data=data)
    new = json.loads(new.json())
    new = pd.DataFrame(new)
    new.to_csv("new_phrase.csv", index=None, sep="|")
    st.write(new)
