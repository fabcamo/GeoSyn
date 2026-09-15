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



import os
import time
import numpy as np
from utils.utils import split_data, save_summary
from geoschemagen.generate_database import generate_database
from utils.create_cptlike import from_schema_to_cptlike

"""
MODIFIED SCRIPT TO GENERATE A GEOTECHNICAL SCHEMATISATION DATABASE FOR ALL MODEL TYPES

This script runs all model types sequentially (A, B, C, D, E, F) with 5000 realizations each
and saves the outputs in separate folders.
"""

# USER DEFINED PARAMETERS
no_realizations = 5000    # Number of realizations to generate
output_base_folder = r'D:\GeoSchemaGen\tests'  # Base folder to save outputs

x_max = 512     # Length (x) of the model
z_max = 32      # Depth (z) of the model

use_RF = True               # On or off: use Random Fields
create_cptlike = True       # On or off: create CPT-like images
save_image = True           # On or off: save the images
save_cptlike_image = False  # On or off: save the cpt-like images
save_csv = False            # On or off: save the csv files

seed = 20202020  # Define a seed for the random number generator
vali_ratio = 0   # Percentage of total data for validation
test_ratio = 0   # Percentage of total data for testing

# Define the model types
model_types = ["A", "B", "C", "D", "E", "F"]

if __name__ == "__main__":
    # Start the overall timer
    overall_time_start = time.time()

    for model_type in model_types:
        print(f"Starting generation for model type {model_type}...")

        # Create a unique output folder for the current model type
        model_output_folder = os.path.join(output_base_folder, f"type{model_type}_RF{use_RF}_" + time.strftime("%Y%m%d"))
        if not os.path.exists(model_output_folder):
            os.makedirs(model_output_folder)

        # Generate the database for the current model type
        generate_database(output_folder=model_output_folder,
                          no_realizations=no_realizations,
                          z_max=z_max, x_max=x_max,
                          seed=seed,
                          model_type=model_type,
                          use_RF=use_RF,
                          create_cptlike=create_cptlike,
                          save_image=save_image,
                          save_cptlike_image=save_cptlike_image,
                          save_csv=save_csv)

        print(f"Completed generation for model type {model_type}. Output saved to {model_output_folder}")

    # End the overall timer
    overall_time_end = time.time()
    print(f"All model types processed. Total time: {overall_time_end - overall_time_start:.2f} seconds.")


