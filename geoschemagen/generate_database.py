import os
import numpy as np
from geoschemagen.create_schema import create_schema, create_schema_noRF, create_schema_eight_layers, \
    create_schema_eight_layers_noRF, create_schema_typeA, create_schema_typeB, create_schema_typeC, create_schema_typeD, \
    create_schema_typeE, create_schema_typeF, create_schema_typeS

# Bundled defaults with tunable boundary parameters for every model type (A-F, S)
DEFAULT_MODEL_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "model_params.json")


def generate_database(output_folder: str,
                      no_realizations: int,
                      z_max: int,
                      x_max: int,
                      seed:int,
                      model_type:str,
                      use_RF:bool = True,
                      create_cptlike:bool = False,
                      save_image:bool = False,
                      save_cptlike_image:bool = False,
                      save_csv:bool = False,
                      save_h5:bool = True,
                      config_path:str = DEFAULT_MODEL_CONFIG):
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
        create_cptlike (bool): Whether to create CPT-like images. Default is False.
        save_image (bool): Whether to save the images. Default is False.
        save_csv (bool): Whether to save the CSV files. Default is False.
        save_h5 (bool): Whether to save the HDF5 (.h5) files. Default is True.
        config_path (str): Path to a JSON file overriding the selected model_type's boundary
            parameters (amplitude, period, phase_shift, vertical_shift, etc.). Defaults to
            geoschemagen/config/model_params.json.

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
                create_schema_typeA(output_folder=output_folder,
                                    counter=counter,
                                    z_max=z_max,
                                    x_max=x_max,
                                    trigo_type=combine_trigo,
                                    seed=seed,
                                    RF=use_RF,
                                    create_cptlike=create_cptlike,
                                    save_image=save_image,
                                    save_cptlike_image=save_cptlike_image,
                                    save_csv=save_csv,
                                    save_h5=save_h5,
                                    config_path=config_path)

            elif model_type == "B":
                # Mix of both sine and cosine in the same model
                combine_trigo = 0
                create_schema_typeB(output_folder=output_folder,
                                    counter=counter,
                                    z_max=z_max,
                                    x_max=x_max,
                                    trigo_type=combine_trigo,
                                    seed=seed,
                                    RF=use_RF,
                                    create_cptlike=create_cptlike,
                                    save_image=save_image,
                                    save_cptlike_image=save_cptlike_image,
                                    save_csv=save_csv,
                                    save_h5=save_h5,
                                    config_path=config_path)

            elif model_type == "C":
                combine_trigo = 0
                create_schema_typeC(output_folder=output_folder,
                                    counter=counter,
                                    z_max=z_max,
                                    x_max=x_max,
                                    trigo_type=combine_trigo,
                                    seed=seed,
                                    RF=use_RF,
                                    create_cptlike=create_cptlike,
                                    save_image=save_image,
                                    save_cptlike_image=save_cptlike_image,
                                    save_csv=save_csv,
                                    save_h5=save_h5,
                                    config_path=config_path)

            elif model_type == "D":
                combine_trigo = 0
                create_schema_typeD(output_folder=output_folder,
                                    counter=counter,
                                    z_max=z_max,
                                    x_max=x_max,
                                    trigo_type=combine_trigo,
                                    seed=seed,
                                    RF=use_RF,
                                    create_cptlike=create_cptlike,
                                    save_image=save_image,
                                    save_cptlike_image=save_cptlike_image,
                                    save_csv=save_csv,
                                    save_h5=save_h5,
                                    config_path=config_path)

            elif model_type == "E":
                combine_trigo = False
                create_schema_typeE(output_folder=output_folder,
                                    counter=counter,
                                    z_max=z_max,
                                    x_max=x_max,
                                    trigo_type=combine_trigo,
                                    seed=seed,
                                    RF=use_RF,
                                    create_cptlike=create_cptlike,
                                    save_image=save_image,
                                    save_cptlike_image=save_cptlike_image,
                                    save_csv=save_csv,
                                    save_h5=save_h5,
                                    config_path=config_path)

            elif model_type == "F":
                create_schema_typeF(output_folder=output_folder,
                                    counter=counter,
                                    z_max=z_max,
                                    x_max=x_max,
                                    seed=seed,
                                    RF=use_RF,
                                    create_cptlike=create_cptlike,
                                    save_image=save_image,
                                    save_cptlike_image=save_cptlike_image,
                                    save_csv=save_csv,
                                    save_h5=save_h5,
                                    config_path=config_path)

            elif model_type == "S":
                # Legacy schemaGAN model: 5 layers, unrestricted sine/cosine boundaries, random order
                # Boundary parameters (amplitude, period, phase_shift, vertical_shift) are read from
                # geoschemagen/config/model_params.json (or config_path, if given)
                create_schema_typeS(output_folder=output_folder,
                                    counter=counter,
                                    z_max=z_max,
                                    x_max=x_max,
                                    seed=seed,
                                    RF=use_RF,
                                    create_cptlike=create_cptlike,
                                    save_image=save_image,
                                    save_cptlike_image=save_cptlike_image,
                                    save_csv=save_csv,
                                    save_h5=save_h5,
                                    config_path=config_path)

            else:
                print("Model type selected not supported")

            # Increment the counter
            counter += 1

        # Catch any exceptions and print the error
        except Exception as e:
            print(f"Error in generating model no. {counter + 1}: {e}")
            continue

