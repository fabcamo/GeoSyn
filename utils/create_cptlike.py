import numpy as np
import pandas as pd
import os


def apply_miss_rate_per_rf(dfs: list, miss_rate: float, min_distance: int):
    """
    Apply a given missing rate to each image in the list of dataframes. The missing rate is applied to the 'IC' column
    of the input dataframes. The missing data is stored in a separate list and the full data is stored in another list.

    Args:
        dfs (list): List of pandas dataframes containing the data.
        miss_rate (float): The missing rate to apply to the data.
        min_distance (int): The minimum distance between missing data points.

    Returns:
        missing_data (list): List of numpy arrays containing the missing data.
        full_data (list): List of numpy arrays containing the full data.
    """

    # Initialize two lists to store missing and full data
    missing_data, full_data = [], []
    # Define the column name of interest
    value_name = 'IC'

    # Print a message to indicate the start of the missing rate application process
    print('Applying missing rate')

    # Iterate through each dataframe in the input list
    for counter, rf in enumerate(dfs):
        # Initialize a list to store data grouped by z values
        data_z = []
        # Group the dataframe by the 'z' column
        grouped = rf.groupby("z")
        # Iterate over the groups and extract the 'IC' column data
        for name, group in grouped:
            data_z.append(list(group[value_name]))

        # Convert the list to a numpy array of floats
        data_z = np.array(data_z, dtype=float)
        # Apply missing rate and return the missing data array
        data_m = remove_random_columns(data_z, miss_rate, min_distance)
        # Append the missing and full data arrays to their respective lists
        missing_data.append(data_m)
        full_data.append(data_z)

        print('schematization to CPT-like no.', counter, ' done')

    # Return the lists of missing and full data arrays
    return missing_data, full_data


def remove_random_columns(data_z, miss_rate: float, min_distance: int):
    """
    Remove a random number of columns from the matrix at a specified rate,
    given a minimum distance between missing data points.

    Args:
        data_z (np.array): A 2D numpy array of data.
        miss_rate (float): The rate at which to remove columns.
        min_distance (int): The minimum distance between missing data points.

    Returns:
        miss_list (np.array): A 2D numpy array with columns removed
    """

    # Transpose the input data to operate on columns instead of rows
    data_z = np.transpose(data_z)

    # Create a matrix of zeros of same shape as data_z
    # This will be used to indicate which data is missing
    data_m = np.zeros_like(data_z)

    # Determine which columns to keep based on the miss_rate and min_distance
    columns_to_keep_index = check_min_spacing(data_z, miss_rate, min_distance)

    # Set the values in data_m to 1 for the columns that are to be kept
    for column_index in columns_to_keep_index:
        data_m[column_index, :] = np.ones_like(data_m[column_index, :])

    # Remove a random number of rows from the bottom of each column
    # to simulate the missing depth data
    data_m = remove_random_depths(data_z, data_m)

    # Multiply the original data by the missing data indicator
    # Missing data will be represented as zero in the final output
    miss_list = np.multiply(data_z, data_m)

    # Transpose the output back to its original orientation
    miss_list = np.transpose(miss_list)

    # Return the final array with random columns removed
    return miss_list


def check_min_spacing(data_z, miss_rate: float, min_distance: int):
    """
    Select the columns to keep for each cross-section based on a missing rate and minimum distance between data points.

    Args:
        data_z (np.array): A 2D numpy array of data.
        miss_rate (float): The rate at which to remove columns.
        min_distance (int): The minimum distance between missing data points.

    Returns:
        columns_to_keep_index (list): A list of indices of columns to keep.
    """

    # Get the number of columns from the transposed data
    all_columns = data_z.shape[0]
    # Calculate how many columns should be missing based on the specified rate
    no_missing_columns = int(miss_rate * all_columns)
    # Calculate how many columns to keep
    no_columns_to_keep = abs(no_missing_columns - all_columns)
    # Initialize an empty list to store indices of columns to keep
    columns_to_keep_index = []

    # Loop until the desired number of columns to keep is achieved
    while len(columns_to_keep_index) != no_columns_to_keep:
        # Generate a random index within the range of columns
        rand_index = int(np.random.uniform(0, all_columns))
        # Define the range of indices to check for duplicates, based on the minimum distance
        range_to_check = range(rand_index - min_distance, rand_index + min_distance + 1)
        # If the random index is already in the list, restart the loop
        if rand_index in columns_to_keep_index:
            continue

        # Check that none of the indices within the range_to_check are in the list already
        if all(index not in columns_to_keep_index for index in range_to_check):
            # If the check passes, add the random index to the list of indices to keep
            columns_to_keep_index.append(rand_index)

    return columns_to_keep_index


def remove_random_depths(data_z, data_m):
    """
    Remove a random amount of data from the bottom of each column in the input matrix.

    Args:
        data_z (np.array): A 2D numpy array of data.
        data_m (np.array): A 2D numpy array indicating where data has been removed.

    Returns:
        data_m (np.array): A 2D numpy array with data removed from the bottom of each column.
    """

    # Get the number of columns (length) and rows (depth) from the input data
    data_length = data_z.shape[0]
    data_depth = data_z.shape[1]

    # Iterate over each column
    for j in range(data_length):
        # Generate a random number with a bias towards lower numbers using a triangular distribution
        # This number will determine how many rows from the bottom of the current column will be removed
        n_rows = int(np.random.triangular(0, 0, data_depth / 2))

        # If there are rows to remove, replace the corresponding rows in data_m with zeros
        if n_rows > 0:
            data_m[j, -n_rows:] = np.zeros(n_rows)

    # Return the updated data_m matrix, which indicates where data has been removed
    return data_m


def load_remove_reshape_data(path_to_images: str, miss_rate: float, min_distance: int, no_rows: int, no_cols: int):
    """
    Load CSV data, remove some of it to simulate incomplete data, then reshape it for further processing.

    Args:
        path_to_images (str): The path to the directory containing the CSV files.
        miss_rate (float): The rate at which to remove columns.
        min_distance (int): The minimum distance between missing data points.
        no_rows (int): The number of rows in the reshaped data.
        no_cols (int): The number of columns in the reshaped data.

    Returns:
        original_img (np.array): The full data reshaped for image processing.
        cptlike_img (np.array): The incomplete data (cpt like image) reshaped for image processing.
    """

    # Load all CSV files from the specified directory
    all_csv = read_all_csv_files(path_to_images)
    # Apply a missing rate to create simulated incomplete data
    missing_data, full_data = apply_miss_rate_per_rf(all_csv, miss_rate, min_distance)

    # Get the number of samples (i.e., CSV files)
    no_samples = len(all_csv)

    # Reshape the missing and full data into the specified number of rows and columns
    # Note that we convert the data type to float32 to ensure compatibility with certain operations downstream
    missing_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in missing_data])
    full_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in full_data])

    # Further reshape the data to include a color channel (of size 1) for compatibility with image processing operations
    original_img = np.reshape(full_data, (no_samples, no_rows, no_cols, 1))
    cptlike_img = np.reshape(missing_data, (no_samples, no_rows, no_cols, 1))

    return original_img, cptlike_img


def read_all_csv_files(directory: str):
    """
    Read all .csv files in the specified directory and return a list of pandas dataframes.

    Args:
        directory (str): The path to the directory containing the .csv files.

    Returns:
        csv_data (list): A list of pandas dataframes containing the data from the .csv files.

    """

    # Print a message to indicate the start of the data reading process
    print('Reading all the data')

    # Ensure that the directory path is compatible with the operating system
    directory = os.path.join(directory)
    # Initialize an empty list to store the dataframes of data read from each .csv file
    csv_data = []

    # Use os.walk() to iterate through all files in the directory, including those in subdirectories
    for root, dirs, files in os.walk(directory):
        # Iterate through the list of files in the current directory
        for file in files:
            # Check if the current file is a .csv file
            if file.endswith(".csv"):
                # Read the .csv file into a pandas dataframe
                df = pd.read_csv(os.path.join(directory, file), delimiter=',')
                # Append the dataframe to the list of dataframes
                csv_data.append(df)

    # Return the list of dataframes
    return csv_data
