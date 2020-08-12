import pandas as pd
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
import streamlit as st

import service
import exceptions

# Entry point for the machine learning preprocessing

# Basemodel for a single ingredient
class Dataset(BaseModel):
    amount: Optional [str] = None
    name:   str
    unit:   Optional [str] = None

# Basemodel for the manually selected ingredients to perform training on the data
class Ingredients(BaseModel):
    name:   str

app = FastAPI()

# preprocessor instances get loaded directly into the main application and can be accessed from here
preprocessor_instances = []

# get ids of actvie preprocessors
@app.get("/get_active_preprocessors/")
def get_active_processors():
    if preprocessor_instances: 
        return [preprocessor.prep_id for preprocessor in preprocessor_instances]
    else:
        return None

# create a new preprocessor
# a dataset is required
@app.post("/create_preprocessor/")
def create_preprocessor(dataset: List[Dataset]):
    prep_id = len(preprocessor_instances)
    preprocessor = service.Preprocessor(prep_id)
    dataset = [dict(data) for data in dataset]
    dataset_df = pd.DataFrame(dataset)
    preprocessor.load_dataset(dataset_df)
    preprocessor_instances.append(preprocessor)
    return preprocessor.prep_id

# returns the dataset of a selected preprocessor
@app.get("/get_dataset/{prep_id}")
def get_dataset(prep_id):
    prep_id = int(prep_id)
    attributes = ["dataset"]

    preprocessor = get_preprocessor(prep_id, attributes)

    dataset = preprocessor.dataset
    dataset_json = dataset.to_json(orient='records')  
    return dataset_json

# load the manually selected ingredients to match them with the dataset
@app.put("/load_ingredients/{prep_id}")
def load_ingredients(prep_id, ingredients: List[Ingredients]):
    prep_id = int(prep_id)
    attributes = []

    preprocessor = get_preprocessor(prep_id, attributes)

    ingredients = [dict(ingredient) for ingredient in ingredients]
    ingredients_df = pd.DataFrame(ingredients)
    preprocessor.load_ingredients(ingredients_df)
    return preprocessor.prep_id

# returns the loaded ingredient list
@app.get("/get_ingredients/{prep_id}")
def get_ingredients(prep_id):
    prep_id = int(prep_id)
    attributes = ["ingredients"]

    preprocessor = get_preprocessor(prep_id, attributes)

    ingredients = preprocessor.ingredients
    ingredients_json = ingredients.to_json(orient='records')  
    return ingredients_json

# runs the preprocessing steps defined in the preprocessing class
@app.put("/run_preprocessing/{prep_id}")
def run_preprocessing(prep_id):
    prep_id = int(prep_id)
    attributes = ["dataset", "ingredients"]

    preprocessor = get_preprocessor(prep_id, attributes)

    preprocessor.preprocess(preprocessor.dataset, preprocessor.ingredients)
    return preprocessor.prep_id

# returns the result of a finished preprocessing task
# attribute is either one of processed data, corpus, zutatenverzeichnis or stemmed ingredients
@app.get("/get_result/{prep_id}/{attribute}")
def get_corpus(prep_id, attribute):
    prep_id = int(prep_id)
    attributes = ["dataset", "ingredients", attribute]

    preprocessor = get_preprocessor(prep_id, attributes)
    result = preprocessor.get_attribute(attribute)
    result_json = result.to_json(orient='records')
    return result_json


# helpfunction to determine the missing attributes of a preprocessor or if it is even existent
def get_preprocessor(prep_id, attributes):
    try:
        preprocessor = preprocessor_instances[prep_id]
    except Exception as e:
        raise HTTPException(status_code=404, detail="Preprocessor does not exist") from e 

    missing_attributes = preprocessor.check_attributes()
    missing_attributes_exception = [attribute for attribute in missing_attributes if attribute in attributes]

    if missing_attributes_exception:
        raise HTTPException(status_code=404, detail={"missing data":missing_attributes_exception}) 
    else:
        return preprocessor



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)