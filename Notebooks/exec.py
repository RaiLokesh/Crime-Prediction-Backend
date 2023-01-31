import pickle
import numpy as np
import pandas as pd


gr_file_path = 'https://raw.githubusercontent.com/NasirKhalid24/ELE494-Project/master/Datasets/Graffiti.csv'
graffiti = pd.read_csv(gr_file_path)
graffiti = graffiti.apply(np.radians)
graffiti = graffiti.values

df_file_path = 'https://raw.githubusercontent.com/NasirKhalid24/ELE494-Project/master/Datasets/drinking_fountains.csv'
drinking = pd.read_csv(df_file_path)
drinking = drinking[['LATITUDE', 'LONGITUDE']]
drinking = drinking.apply(np.radians)
drinking = drinking.values

with open("graffiti.pkl", "wb") as file:
    pickle.dump(graffiti, file)
    
with open("drinking.pkl", "wb") as file:
    pickle.dump(drinking, file)
    
#with open("data.pkl", "rb") as file:
#    loaded_data = pickle.load(file)

#print(loaded_data)
