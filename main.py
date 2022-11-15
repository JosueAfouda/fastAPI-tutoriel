#pip install fastapi
#pip install uvicorn

# Librairies
from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris

iris = load_iris()

# Chargement du modèle
loaded_model = load('logreg.joblib')

# Création d'une nouvelle instance fastAPI
app = FastAPI()

# Définir un objet (une classe) pour réaliser des requêtes
# dot notation (.)
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

# Definition du chemin du point de terminaison (API)
@app.post("/predict") # local : http://127.0.0.1:8000/predict

# Définition de la fonction de prédiction
def predict(data : request_body):
    # Nouvelles données sur lesquelles on fait la prédiction
    new_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Prédiction
    class_idx = loaded_model.predict(new_data)[0]

    # Je retourne le nom de l'espèce iris
    return {'class' : iris.target_names[class_idx]}