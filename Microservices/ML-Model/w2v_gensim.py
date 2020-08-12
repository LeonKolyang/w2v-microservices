import gensim.models
import pandas as pd
import streamlit as st

from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling
import matplotlib.pyplot as plt

class W2V():
    def __init__(self):
        self.zutatenverzeichnis = pd.DataFrame()

        self.iterations = None
        self.window_size = None
        self.dimensions = None
        self.min = None
        self.neg = None

        self.sentences = []

        self.word_vectors = None

        self.MODEL = None

    def load_data(self, data):
        try:
            zutatenverzeichnis = pd.DataFrame(data["zutatenverzeichnis"])
            self.zutatenverzeichnis = zutatenverzeichnis
        except Exception as e:
            print(e) 
            pass
    
    def load_parameters(self, parameter_list):
        for parameter in parameter_list:
            self.__dict__[parameter["parameter"]] = int(parameter["value"])

    def run(self):
        self.sentences = self.buildSentences()
        word_vectors = self.train_model(self.sentences, self.iterations, self.window_size, self.dimensions, self.min, self.neg)
        self.word_vectors = self.get_vectors()

    def add_new_phrase(self, new_phrase):
        
        model = self.MODEL
        model.build_vocab([new_phrase], update=True)
        model.train(new_phrase, total_examples=model.corpus_count, epochs=model.iter)
        self.MODEL = model
        self.word_vectors = self.get_vectors()

        evaluated = pd.DataFrame(data=new_phrase, columns=["labels"])    
        evaluated = evaluated.merge(self.word_vectors,  how="left")
        evaluated["similar"] = evaluated.apply(lambda row: model.most_similar(str(row[0]))[0][0] if not(np.isnan(row[1])) else None, axis=1)
        
        return evaluated

    def save(self):
        model = self.MODEL
        model.save("word2vec.model")

    def load_old(self):
        self.MODEL = gensim.models.Word2Vec.load("word2vec.model")
        self.word_vectors = self.get_vectors()


    def get_result(self, result):
        return self.word_vectors

    def get_description(self):
        z_check = True
        if self.zutatenverzeichnis.empty:
            z_check = False

        description = {"zutatenverzeichnis": z_check,
                        "iterations": self.iterations,
                        "window_size": self.window_size,
                        "dimensions": self.dimensions,
                        "min": self.min,
                        "neg": self.neg}
        return description


    # Baue aus dem Zutatenverzeichnis die Sätze
    def buildSentences(self):
        sentences = []
        for index, row in self.zutatenverzeichnis.iterrows():
            list = row[0].split(" ")
            if len(list[-1]) == 0:
                list = list[:-1]
            sentences.append(list)
        return sentences
        # Testausgabe der Sätze
        #for i, sentence in enumerate(self.sentences):
        #    st.write(sentence)
        #    if i == 10:
        #        break

    def train_model(self, sentences, no_iterations, window_size, dimensions, min, neg, *args):
        #Aufruf des Trainings desWord2Vec Algorithmus mit den in der Arbeit beschriebenen Parametern
        self.MODEL = gensim.models.Word2Vec(sentences, sg=1,min_count=min, size= dimensions, negative=neg, iter=no_iterations, window=window_size)


    # gensim implementierung des scikit-learn Dimensionsreduzieren
    def reduce_dimensions(self, MODEL):
        num_dimensions = 2  # final num dimensions (2D, 3D, etc)

        vectors = [] # positions in vector space
        labels = [] # keep track of words to label our data again later
        for word in MODEL.wv.vocab:
            vectors.append(MODEL.wv[word])
            labels.append(word)

        # convert both lists into numpy vectors for reduction
        vectors = np.asarray(vectors)
        labels = np.asarray(labels)

        # reduce using t-SNE
        vectors = np.asarray(vectors)
        tsne = TSNE(n_components=num_dimensions)
        vectors = tsne.fit_transform(vectors)

        x_vals = [v[0] for v in vectors]
        y_vals = [v[1] for v in vectors]
        return x_vals, y_vals, labels

    def get_vectors(self):
        #Reduzierung der Dimensionen
        x_vals, y_vals, labels = self.reduce_dimensions(self.MODEL)
        #Speichere die finalen Wordembeddings als .csv Datei ab
        vectors = pd.DataFrame({"x":x_vals,"y" :y_vals,"labels":labels})
        return vectors


if __name__ == "__main__":
    
    new_phrase = ["Gramm","rote", "Tomaten"]
    w2v = W2V()
    w2v.load_old()
    w2v.add_new_phrase(new_phrase)
    # st.write(evaluated)
    # w2v.word_vectors.to_csv("new_word_vectors.csv", index = False, sep="|")
    word_vectors = w2v.word_vectors
    #word_vectors = pd.read_csv("new_word_vectors.csv", index_col=None, sep="|")
    evaluated = pd.DataFrame(data=new_phrase, columns=["labels"])

    
    evaluated= evaluated.merge(word_vectors,  how="left")
    st.write(evaluated)
    evaluated["similar"] = evaluated.apply(lambda row: w2v.MODEL.most_similar(str(row[0])) if not(np.isnan(row[1])) else np.NaN, axis=1)
    #evaluated["similar"] = evaluated.apply(lambda row: st.write(str(row[0])) if not(np.isnan(row[1])) else None, axis=1)
    #st.write(similar)
    st.write(evaluated)


    # plt.scatter(w2v.word_vectors.x, w2v.word_vectors.y)
    # plt.scatter(tomate.x, tomate.y,c="r")
    # st.pyplot()
    # st.write(w2v.word_vectors)


