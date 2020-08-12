import pandas as pd
import w2v_gensim
import clustering
import sys

# central handler for model instances
# can only hold one instance per model
# passes the methods to the selected model
class ModelHandler:
    def __init__(self):     
        self.models = {"word2vec": None,
                        "clustering": None}


    def load_model(self, model, state):
        sys.stdout.write(str(id(self)))
        if model == "word2vec":
            if state == "new":
                w2v_model = w2v_gensim.W2V()
                self.models["word2vec"]= w2v_model
            elif state == "old":
                w2v_model = w2v_gensim.W2V()
                w2v_model.load_old()
                self.models["word2vec"]= w2v_model
        if model == "clustering":
            cluster_model = clustering.Clustering()
            self.models["clustering"]= cluster_model
        sys.stdout.write(str(self.models))

        
        
    def run_model(self, model):
        sys.stdout.write(str(id(self)))
        sys.stdout.write(str(self.models))
        model = self.models[model]
        model.run()
    
    # data exchange for the phrase evaluation between the word2vec and the clustering model is done by the modelhandler
    # the modelhandler merges the data of both models with the given phrase to return a result
    def evaluate_new_phrase(self, model, data):
        sys.stdout.write(str(id(self)))
        sys.stdout.write(str(self.models))
        w2v_model = self.models["word2vec"]
        clustering = self.models["clustering"]

        word_cluster = clustering.get_result("clustered_word_vectors")
        word_cluster = word_cluster.dropna()
        cluster_reinheit = clustering.get_result("cluster_reinheit")
        evaluated = w2v_model.add_new_phrase(data["new_phrase"])
        
        evaluated["cluster"] = evaluated.merge(right=word_cluster,left_on="similar", right_on="labels", how="left")["cluster"]

        cluster_reinheit = cluster_reinheit.reset_index()
        cluster_reinheit = cluster_reinheit.rename(columns={"level_0":"cluster"})
        cluster_reinheit["cluster"] = cluster_reinheit["cluster"].astype(float)

        evaluated["reinheit"] = evaluated.merge(right=cluster_reinheit,left_on="cluster", right_index=True, how="left")["Reinheit"]

        evaluated_json = evaluated.to_json()
        return evaluated_json

    def load_model_data(self, model, data):
        sys.stdout.write(str(id(self)))
        sys.stdout.write(str(self.models))
        model = self.models[model]
        model.load_data(data)

    def load_model_parameters(self, model, parameter_list):
        sys.stdout.write(str(id(self)))
        sys.stdout.write(str(self.models))
        model = self.models[model]
        model.load_parameters(parameter_list)
        

    def get_model_result(self, model, result):
        sys.stdout.write(str(id(self)))
        sys.stdout.write(str(self.models))
        model = self.models[model]
        result = model.get_result(result)
        return result
    
    def get_model_description(self, model):
        sys.stdout.write(str(id(self)))
        sys.stdout.write(str(self.models))
        model = self.models[model]
        return model.get_description()
    
    def get_model_data(self, model):
        sys.stdout.write(str(id(self)))
        sys.stdout.write(str(self.models))
        model = self.models[model]
        return model.get_data()

    def save_model(self, model):
        sys.stdout.write(str(id(self)))
        sys.stdout.write(str(self.models))
        model = self.models[model]
        model.save()



