import os
import shutil
import time
import numpy as np


def split_data(data_path: str, train_folder: str, validation_folder: str, test_folder: str,
               vali_ratio: float = 0.1666666, test_ratio: float = 0.1666666, shuffle: bool = True):
    """
    Split data into train, validation, and test sets.

    Args:
        data_path (str): Path to the source data directory.
        train_folder (str): Path to the directory where train data will be saved.
        validation_folder (str): Path to the directory where validation data will be saved.
        test_folder (str): Path to the directory where test data will be saved.
        vali_ratio (float, optional): Ratio of data for training (default is 0.1666666).
        test_ratio (float, optional): Ratio of data for testing (default is 0.1666666).
        shuffle (bool, optional): Whether to shuffle the data indices (default is True).
    Return:
        None
    """

    # Create directories if they don't exist
    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)
    if not os.path.isdir(validation_folder):
        os.makedirs(validation_folder)
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)

    # Get list of CSV files in data_path
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

    # Calculate the number of files for each set
    nb_files = len(csv_files)
    nb_vali = int(nb_files * vali_ratio)
    nb_test = int(nb_files * test_ratio)
    nb_train = nb_files - (nb_vali + nb_test)

    # Shuffle the indexes
    indexes = np.arange(nb_files)
    if shuffle:
        np.random.shuffle(indexes)

    # Select indexes for train, validation, and test sets
    indexes_train = indexes[:nb_train]
    indexes_validation = indexes[nb_train:nb_train + nb_vali]
    indexes_test = indexes[nb_train + nb_vali:]

    # Copy files to respective folders
    for i in indexes_train:
        csv_file = csv_files[i]
        png_file = csv_file.replace(".csv", ".png")
        shutil.copy(os.path.join(data_path, csv_file), os.path.join(train_folder, csv_file))
        shutil.copy(os.path.join(data_path, png_file), os.path.join(train_folder, png_file))

    for i in indexes_validation:
        csv_file = csv_files[i]
        png_file = csv_file.replace(".csv", ".png")
        shutil.copy(os.path.join(data_path, csv_file), os.path.join(validation_folder, csv_file))
        shutil.copy(os.path.join(data_path, png_file), os.path.join(validation_folder, png_file))

    for i in indexes_test:
        csv_file = csv_files[i]
        png_file = csv_file.replace(".csv", ".png")
        shutil.copy(os.path.join(data_path, csv_file), os.path.join(test_folder, csv_file))
        shutil.copy(os.path.join(data_path, png_file), os.path.join(test_folder, png_file))

    # Delete the csv and png files from the original folder
    for file_name in csv_files:
        file_path = os.path.join(data_path, file_name)
        os.remove(file_path)
        png_file = file_name.replace(".csv", ".png")
        file_path = os.path.join(data_path, png_file)
        os.remove(file_path)

    # Move matching files from cptlike folder to test folder
    # Define the cptlike and test folders
    source_folder = test_folder
    destination_folder = os.path.join(data_path, "cptlike_images")
    move_matching_files(source_folder, destination_folder)
    # Move matching files from cptlike folder to validation folder
    source_folder = validation_folder
    move_matching_files(source_folder, destination_folder)
    # Move matching files from cptlike folder to train folder
    source_folder = train_folder
    move_matching_files(source_folder, destination_folder)

    # Delete the cptlike folder
    shutil.rmtree(destination_folder)



def move_matching_files(source_folder, destination_folder):
    # Extract file numbers from source folder
    file_numbers = []
    for filename in os.listdir(source_folder):
        if filename.endswith("."):
            continue
        file_number = filename.split("_")[-1].split(".")[0]
        file_numbers.append(file_number)

    # Move matching files from destination folder to source folder
    for filename in os.listdir(destination_folder):
        if filename.endswith("."):
            continue
        file_number = filename.split("_")[-1].split(".")[0]
        if file_number in file_numbers:
            source_file_path = os.path.join(destination_folder, filename)
            destination_file_path = os.path.join(source_folder, filename)
            shutil.move(source_file_path, destination_file_path)



def save_summary(output_folder: str, time_start: float, time_end: float, seed: int, no_realizations: int):
    """
    Save a summary of the run times and seed.

    Args:
        output_folder (str): Path to the output folder.
        time_start (float): Start time of the execution.
        time_end (float): End time of the execution.
        seed (int): Random seed.
        no_realizations (int): Number of realizations generated.
    Return:
        None
    """
    # Calculate the execution time
    execution_time = abs(time_start - time_end)
    # Format time taken to run into> Hours : Minutes : Seconds
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    time_str = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

    # Save the summary to a file
    time_current = time.strftime("%d/%m/%Y %H:%M:%S")
    # Create the file path
    file_path = os.path.join(output_folder, 'random_seed.txt')
    # Write the summary to the file
    with open(file_path, 'w') as f:
        f.write("Executed on: {}\n".format(time_current))
        f.write("Execution time: {}\n\n".format(time_str))
        f.write("Seed: {}\n\n".format(seed))
        f.write("No. of realizations: {}\n\n".format(no_realizations))