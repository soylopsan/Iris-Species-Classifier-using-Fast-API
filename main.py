from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Species list (0 = setosa, 1 = versicolor, 2 = virginica)
species = ['setosa', 'versicolor', 'virginica']

# Load model from file
with open('iris.pkl', 'rb') as file:
    iris_mdl = pickle.load(file)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class Measure(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/iris/")
async def create_item(msr: Measure):
    """
    # This is an API for an Iris Species classifier based on their features
    ### It works sending measurements in a json like this:
    {\n
        "sepal_length": "float"
        "sepal_width": "float"
        "petal_length": "float"
        "petal_width": "float"
    }\n
    """
    # We predict the type of iris: setosa, versicolor or virginica 
    pred = iris_mdl.predict([[msr.sepal_length, msr.sepal_width, 
        msr.petal_length, msr.petal_width]])
    return {'Prediction: ' + species[int(pred)]}