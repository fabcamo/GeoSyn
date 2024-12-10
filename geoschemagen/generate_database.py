import numpy as np
from geoschemagen.create_schema import create_schema, create_schema_noRF, create_schema_eight_layers, \
    create_schema_eight_layers_noRF, create_schema_typeA, create_schema_typeB, create_schema_typeC, create_schema_typeD, \
    create_schema_typeE, create_schema_typeF, create_schema_typeA_h5



def generate_database(output_folder: str,
                      no_realizations: int,
                      z_max: int,
                      x_max: int,
                      seed:int,
                      model_type:str,
                      use_RF:bool = True):
    """
    Generate a database of synthetic data with given parameters and save results in the specified output folder.

    Args:
        output_folder (str): The folder to save the synthetic data.
        no_realizations (int): The number of realizations to generate.
        z_max (int): The depth of the model.
        x_max (int): The length of the model.
        seed (int): The seed for the random number generator.
        model_type (str): The type of subsoil model to generate.
        use_RF (bool): Whether to use Random Fields. Default is True.

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
            # Check which model the user wants to generate
            if model_type == "A":
                # Just sine (1) or just cosine (2) per model
                combine_trigo = np.random.choice([1, 2])
                create_schema_typeA_h5(output_folder=output_folder, counter=counter, z_max=z_max, x_max=x_max, trigo_type=combine_trigo, seed=seed, RF=use_RF, save_image=False, save_csv=False)
            elif model_type == "B":
                # Mix of both sine and cosine in the same model
                combine_trigo = 0
                create_schema_typeB(output_folder, counter, z_max, x_max, combine_trigo, seed, use_RF)
            elif model_type == "C":
                combine_trigo = 0
                create_schema_typeC(output_folder, counter, z_max, x_max, combine_trigo, seed, use_RF)
            elif model_type == "D":
                combine_trigo = 0
                create_schema_typeD(output_folder, counter, z_max, x_max, combine_trigo, seed, use_RF)
            elif model_type == "E":
                combine_trigo = False
                create_schema_typeE(output_folder, counter, z_max, x_max, combine_trigo, seed, use_RF)
            elif model_type == "F":
                create_schema_typeF(output_folder, counter, z_max, x_max, seed, use_RF)
            else:
                print("Model type selected not supported")

            # Increment the counter
            counter += 1

        # Catch any exceptions and print the error
        except Exception as e:
            print(f"Error in generating model no. {counter + 1}: {e}")
            continue

