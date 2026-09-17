import os
import shutil
from geoschemagen.utils.utils import split_data
from geoschemagen.generate_database import generate_database
import pytest


@pytest.mark.parametrize("model_type", ["A", "B", "C", "D", "E", "F", "S"])
def test_generation_images(model_type):
    """
    Test the generation of images and CSV files for different model types.

    Args:
        model_type (str): The type of the model to test.
    """

    no_realizations = 10    # Number of realizations to generate
    output_base_folder = f'tests/tmp_{model_type}'  # Base folder to save outputs
    os.makedirs(output_base_folder, exist_ok=True)

    x_max = 512     # Length (x) of the model
    z_max = 32      # Depth (z) of the model

    use_RF = True               # On or off: use Random Fields
    create_cptlike = False       # On or off: create CPT-like images
    save_image = False           # On or off: save the images
    save_cptlike_image = False  # On or off: save the cpt-like images
    save_csv = True            # On or off: save the csv files
    save_h5 = False             # On or off: save the h5 files

    seed = 14  # Define a seed for the random number generator
    vali_ratio = 0.15   # Percentage of total data for validation
    test_ratio = 0.15   # Percentage of total data for testing

    # Generate the database for the current model type
    generate_database(output_folder=output_base_folder,
                    no_realizations=no_realizations,
                    z_max=z_max, x_max=x_max,
                    seed=seed,
                    model_type=model_type,
                    use_RF=use_RF,
                    create_cptlike=create_cptlike,
                    save_image=save_image,
                    save_cptlike_image=save_cptlike_image,
                    save_csv=save_csv,
                    save_h5=save_h5)

    validation_folder = os.path.join(output_base_folder, "validation")
    test_folder = os.path.join(output_base_folder, "test")
    train_folder = os.path.join(output_base_folder, "train")
    split_data(
        data_path=output_base_folder,
        train_folder=train_folder,
        validation_folder=validation_folder,
        test_folder=test_folder,
        vali_ratio=vali_ratio,
        test_ratio=test_ratio,
    )

    # compare the CSV against results
    folders = ["test", "train", "validation"]
    for fold in folders:
        files = [f for f in os.listdir(f"./tests/_results/{model_type}/{fold}") if f.endswith(".csv")]
        for file in files:
            assert compare_csv(
                os.path.join(f"./tests/_results/{model_type}/{fold}", file),
                os.path.join(output_base_folder, fold, file)
                )

    shutil.rmtree(output_base_folder)


from itertools import zip_longest


def compare_csv(file1, file2):
    """
    Compares the contents of two CSV files, allowing for very small
    floating-point differences in the IC values.
    """

    with open(file1, "r") as f1:
        content1 = f1.readlines()

    with open(file2, "r") as f2:
        content2 = f2.readlines()

    # Different number of lines means different files
    if len(content1) != len(content2):
        return False

    for line1, line2 in zip(content1, content2):

        # If lines are exactly equal, nothing else to check
        if line1 == line2:
            continue

        values1 = line1.strip().split(",")
        values2 = line2.strip().split(",")

        # x and z must still be exactly identical
        if values1[:-1] != values2[:-1]:
            return False

        # Allow tiny floating-point differences in IC
        try:
            if abs(float(values1[-1]) - float(values2[-1])) > 1e-12:
                return False
        except ValueError:
            # Handles header or other non-numeric differences
            return False

    return True
