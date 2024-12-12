import os
import time
import numpy as np
from utils.utils import split_data, save_summary
from geoschemagen.generate_database import generate_database
from utils.create_cptlike import from_schema_to_cptlike

"""
MAIN SCRIPT TO GENERATE A GEOTECHNICAL SCHEMATISATION DATABASE

The user needs to input:
    - output_folder: the folder to save the synthetic data
    - no_realizations: the number of realizations to generate (with or without RF)
    - x_max: the length of the model
    - z_max: the depth of the model
    - vali_ratio: the percentage of total data for validation
    - test_ratio: the percentage of total data for testing
"""

# Define the type of subsoil model you want to randomly generate
"""
    - "A": Very horizontal, 3 or 4 layers, no indentation, no lens, fixed top and bottom.
    - "B": Very horizontal, up to 6 layers, indentations possible, fixed bottom.
    - "C": Horizontal layers with lenses at different positions, up to 4 layers, fixed bottom.
    - "D": Horizontal layers with intercalations sand and clay, up to 7 layers, fixed bottom not always present.
    - "E": Inclined layers
    - "F": Irregular sinuosoidal layers
"""

# USER DEFINED PARAMETERS FOR THE GENERATION
model_type = "B"
no_realizations = 20    # Number of realizations to generate
# Output folder
output_folder = r'D:\GeoSchemaGen\tests'

x_max = 512     # Length (x) of the model
z_max = 32      # Depth (z) of the model

use_RF = True               # On or off: use Random Fields
create_cptlike = True       # On or off: create CPT-like images
save_image = True           # On or off: save the images
save_cptlike_image = True   # On or off: save the cpt-like images
save_csv = False             # On or off: save the csv files

seed = 20202020     # Define a seed for the random number generator
#seed = np.random.randint(20220412, 20230412)

vali_ratio = 0      # Percentage of total data for validation
test_ratio = 0      # Percentage of total data for testing

########################################################################################################################

# In the output_folder location, create a folder to save the synthetic data
# The name of the folder is "type" + model_type + use_RF + date", but if use_RF is True, add RF in the name, else add NoRF
output_folder = os.path.join(output_folder, "type" + model_type + "_RF" + str(use_RF) + "_" + time.strftime("%Y%m%d"))

# if the folder does not exist, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Geometry pre-process
x_coord = np.arange(0, x_max, 1)
z_coord = np.arange(0, z_max, 1)
xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")




if __name__ == "__main__":
    # Check the time and start the timers
    time_current = time.strftime("%d/%m/%Y %H:%M:%S")
    time_start = time.time()

    # Call the generate_database function to create images
    generate_database(output_folder=output_folder,
                      no_realizations=no_realizations,
                      z_max=z_max, x_max=x_max,
                      seed=seed,
                      model_type=model_type,
                      use_RF=use_RF,
                      create_cptlike=create_cptlike,
                      save_image=save_image,
                      save_cptlike_image=save_cptlike_image,
                      save_csv=save_csv)

    #TODO: I commented this to try and find the different models scenarios> playing with the layer boundaries....

    # # Call the function to generate the cpt like images
    # cpt_like_image = from_schema_to_cptlike(path_to_images=output_folder,
    #                                         miss_rate=0.99,
    #                                         min_distance=50,
    #                                         no_cols=z_max,
    #                                         no_rows=x_max)
    #
    #
    # # Split the data into training and validation at random
    # validation_folder = os.path.join(output_folder, "validation")
    # test_folder = os.path.join(output_folder, "test")
    # train_folder = os.path.join(output_folder, "train")
    # split_data(data_path=output_folder, train_folder=train_folder, validation_folder=validation_folder,
    #            test_folder=test_folder, vali_ratio=vali_ratio, test_ratio=test_ratio)
    #
    # # End the timer
    # time_end = time.time()
    #
    # # Save a summary of the run times and seed
    # save_summary(output_folder, time_start, time_end, seed, no_realizations)
    #


