from fastapi import FastAPI
from pydantic import BaseModel
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()

# We load the Iris Dataset and set the K Neighbors Classifier 
##for 8 neighbors (95% accuracy)
iris = datasets.load_iris()
knn = KNeighborsClassifier(n_neighbors=8)

# We fit the classifier using sckit-learn
X=iris['data']
y=iris['target']
knn.fit(X, y)

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
    pred = knn.predict([[msr.sepal_length, msr.sepal_width, 
        msr.petal_length, msr.petal_width]])
    return {'Prediction: {}'.format(iris.target_names[pred])}