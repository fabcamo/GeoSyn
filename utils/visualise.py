# Open a csv file and read the data
# The csv file should have the following columns:
# ,x,z,IC: with the first one being the index, x being the x coordinate, z being the z coordinate and IC being the intensity value
# It also have such a header { ,x,z,IC }

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# define the path to the csv file
cs_file = Path(r"C:/Users/camposmo/OneDrive - Stichting Deltares/Documents/Projects/schemaGAN/forRoeland/simple/cs_830.csv")

# read the data from the csv file
cs_data = pd.read_csv(cs_file)
# print the first 5 rows of the data
print(cs_data.head())

# Rearrange into a table of x no of columns and z no of rows with IC as the values
cs_data_arr = cs_data.pivot(index='z', columns='x', values='IC')
print(cs_data.head())

# plot the data with imshow

plt.imshow(cs_data_arr)
plt.show()
