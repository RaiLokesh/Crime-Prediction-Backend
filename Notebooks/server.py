import pickle
import numpy as np

def DistanceToGraffiti(lat1, lon1):
    r = 6371 # Radius of earth in kilometers
    with open("graffiti.pkl", "rb") as file:
        graffiti = pickle.load(file)
    coordinates = np.array([np.radians(lat1), np.radians(lon1)])
    diff_array = graffiti - coordinates
    # Applying the haversine formula as array functions for speed
    a = np.square(np.sin(diff_array[:, 0]/2.0)) + np.cos(coordinates[0]) * np.cos(graffiti[:, 0]) * np.square(np.sin(diff_array[:, 1]/2.0))
    c = 2 * np.arcsin(np.sqrt(a)) 
    return np.amin(c)*r
#DistanceToGraffiti(49.2650765,-123.1184743)


def DistanceToFountain(lat1, lon1):
    r = 6371 # Radius of earth in kilometers
    with open("drinking.pkl", "rb") as file:
        drinking = pickle.load(file)
    coordinates = np.array([np.radians(lat1), np.radians(lon1)])
    diff_array = drinking - coordinates
    # Applying the haversine formula as array functions for speed
    a = np.square(np.sin(diff_array[:, 0]/2.0)) + np.cos(coordinates[0]) * np.cos(drinking[:, 0]) * np.square(np.sin(diff_array[:, 1]/2.0))
    c = 2 * np.arcsin(np.sqrt(a)) 
    return np.amin(c)*r
#DistanceToFountain(49.2650765,-123.1184743)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
    
test = np.array([[2012, 6, 4, 1, 2, 49.244879, -123.085077, DistanceToGraffiti(49.244879, -123.085077), DistanceToFountain(49.244879, -123.085077)]])
test = scaler.transform(test)

print(model.predict(test))