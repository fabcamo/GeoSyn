import numpy as np
from geoschemagen.create_schema import create_schema

def generate_database(output_folder: str, no_realizations: int, z_max: int, x_max: int, seed:int):
    """
    Generate a database of synthetic data with given parameters and save results in the specified output folder.

    Args:
        output_folder (str): The folder to save the synthetic data.
        no_realizations (int): Number of realizations to generate.
        z_max (int): Depth of the model.
        x_max (int): Length of the model.
        seed (int): Seed for random number generation.
    Return:
        None
    """
    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Start the counter
    counter = 0
    # Loop through the number of realizations
    while counter < no_realizations:
        try:
            # Print the realization number
            print('Generating model no.:', counter+1)
            # Generate the synthetic data
            create_schema(output_folder, counter, z_max, x_max, seed)
            # Increment the counter
            counter += 1

        # Catch any exceptions and print the error
        except Exception as e:
            print(f"Error in generating model no. {counter + 1}: {e}")
            continue