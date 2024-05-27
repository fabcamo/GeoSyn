import os
import time
import numpy as np
from utils.utils import split_data, save_summary
from geoschemagen.generate_database import generate_database

# User input for training
output_folder = 'D:\\GeoSchemaGen\\tests\\outputs'
no_realizations = 20
vali_ratio = 0.1
test_ratio = 0.1
x_max = 512
z_max = 32

# Geometry pre-process
x_coord = np.arange(0, x_max, 1)
z_coord = np.arange(0, z_max, 1)
xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")

if __name__ == "__main__":
    # Check the time and start the timers
    time_current = time.strftime("%d/%m/%Y %H:%M:%S")
    time_start = time.time()

    # Generate seed
    seed = np.random.randint(20220412, 20230412)

    # Call the generate_database function to create images
    generate_database(output_folder, no_realizations, z_max, x_max, seed)

    # Split the data into training and validation at random
    validation_folder = os.path.join(output_folder, "validation")
    test_folder = os.path.join(output_folder, "test")
    split_data(output_folder, os.path.join(output_folder, "train"), validation_folder, test_folder, vali_ratio, test_ratio)

    # End the timer
    time_end = time.time()

    # Save a summary of the run times and seed
    save_summary(output_folder, time_start, time_end, seed, no_realizations)
