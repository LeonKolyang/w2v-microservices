import pandas as pd
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import json
import sys

import streamlit as st

import service
import exceptions

sys.stdout.write("Application started")


# Entry point for the machine learning models api

# Basemodel for all incoming data
class Data(BaseModel):
    data:   dict

# Basemodel for all incoming modelparameters
class Parameter(BaseModel):
    parameter: str
    value: str


app = FastAPI()

MODELHANDLER = service.ModelHandler()

# creates the instance of a model which will be used
# necessary for all models and tasks
@app.post("/create_model/{model}/{state}")
def create_model(model, state):
    MODELHANDLER.load_model(model, state)
    return model

# loads the required data into a model
# necessary for all models and tasks
@app.put("/load_model_data/{model}")
def load_model_data(model, data: Data):
    data = dict(data)["data"]
    #wordlist_df = pd.DataFrame(data)
    MODELHANDLER.load_model_data(model, data)
    return model

# loads the required parameters into a model
# necessary for all models and tasks
@app.put("/load_model_parameters/{model}")
def load_model_parameters(model, parameters: List[Parameter]):
    parameters = [dict(parameter) for parameter in parameters]
    MODELHANDLER.load_model_parameters(model, parameters)
    return model

# runs a model and stores the results in the modelinstance
@app.put("/run_model/{model}")
def run_model(model):
    run = MODELHANDLER.run_model(model)
    return model

# runs a model in the background
# for controlled larger operations only, do not use in scripts 
@app.put("/run_model_background/{model}")
async def start_background_run(model, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_model, model)
    return model

# central method for using the trained models
@app.get("/evaluate_new_phrase/{model}")
def evaluate_new_phrase(model, data: Data):
    data = dict(data)["data"]
    evaluated = MODELHANDLER.evaluate_new_phrase(model, data)
    return evaluated


# testmethod for modelresults
@app.get("/get_result/{model}/{result}")
def get_result(model, result):
    result = MODELHANDLER.get_model_result(model, result)
    result_json = result.to_json(orient='records')  
    return result_json

# delivers an overview of the model data and parameters
@app.get("/get_model_description/{model}")
def get_model_description(model):
    description = MODELHANDLER.get_model_description(model)
    description_json = json.dumps(description)
    return description 

# delivers the model data
@app.get("/get_model_data/{model}")
def get_model_data(model):
    data = MODELHANDLER.get_model_data(model)
    data_json = json.dumps

# saves a model as file to the server
# for word2vec only
@app.put("/save_model/{model}")
def save_model(model):
    MODELHANDLER.save_model(model)


    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)