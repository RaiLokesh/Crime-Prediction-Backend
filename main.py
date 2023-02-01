from typing import Union
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=['*'],
    allow_origins=["http://localhost","http://127.0.0.1:3000", 'http://localhost:8080', "http://localhost:3000", "https://crime-prediction-frontend.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Params(BaseModel):
    day: int
    month: int
    year: int
    hour: int
    minutes: int
    latitude: float
    longitude: float

def DistanceToGraffiti(lat1, lon1):
    r = 6371 # Radius of earth in kilometers
    with open("Notebooks/graffiti.pkl", "rb") as file:
        graffiti = pickle.load(file)
    coordinates = np.array([np.radians(lat1), np.radians(lon1)])
    diff_array = graffiti - coordinates
    # Applying the haversine formula as array functions for speed
    a = np.square(np.sin(diff_array[:, 0]/2.0)) + np.cos(coordinates[0]) * np.cos(graffiti[:, 0]) * np.square(np.sin(diff_array[:, 1]/2.0))
    c = 2 * np.arcsin(np.sqrt(a)) 
    return np.amin(c)*r


def DistanceToFountain(lat1, lon1):
    r = 6371 # Radius of earth in kilometers
    with open("Notebooks/drinking.pkl", "rb") as file:
        drinking = pickle.load(file)
    coordinates = np.array([np.radians(lat1), np.radians(lon1)])
    diff_array = drinking - coordinates
    # Applying the haversine formula as array functions for speed
    a = np.square(np.sin(diff_array[:, 0]/2.0)) + np.cos(coordinates[0]) * np.cos(drinking[:, 0]) * np.square(np.sin(diff_array[:, 1]/2.0))
    c = 2 * np.arcsin(np.sqrt(a)) 
    return np.amin(c)*r

with open("Notebooks/model.pkl", "rb") as file:
    model = pickle.load(file)
with open("Notebooks/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)




@app.get("/")
def read_root():
    test = np.array([[2012, 6, 4, 1, 2, 49.244879, -123.085077, DistanceToGraffiti(49.244879, -123.085077), DistanceToFountain(49.244879, -123.085077)]])
    test = scaler.transform(test)
    return str(model.predict(test))+str(DistanceToGraffiti(49.244879, -123.085077))+str(DistanceToFountain(49.244879, -123.085077))

@app.post("/predict/")
async def prediction(values: Params):
    test = np.array([[values.year, values.month, values.day, values.hour, values.minutes, values.latitude, values.longitude, DistanceToGraffiti(values.latitude, values.longitude), DistanceToFountain(values.latitude, values.longitude)]])
    test = scaler.transform(test)
    return (model.predict(test)[0][0]*100)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# run -> uvicorn main:app
