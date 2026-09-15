import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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



def create_cptlike_array(image_matrix: np.array, x_max: int, z_max: int, plot: bool = False):
    """
    Take the synthetic 2D image generated and apply a missing rate to simulate incomplete data for CPT-like images.

    Args:
        image_matrix (np.array): The synthetic 2D image generated
        x_max (int): The maximum value of the x-axis.
        z_max (int): The maximum value of the z-axis.
        plot (bool): Whether to plot the original and CPT-like images side by side. Default is False.

    Returns:
        cptlike_img (np.array): The incomplete data (cpt like image) reshaped for image processing.
    """
    # Reshape the image matrix to the required format for further processing
    data_or = image_matrix.reshape(x_max, z_max).T

    # Remove a random number of columns from the matrix at a specified rate
    data = remove_random_columns(data_or, miss_rate=0.99, min_distance=50)

    if plot:
        # Plot both images side by side
        fig, ax = plt.subplots(2, 1, figsize=(12, 6))

        ax[0].imshow(data_or)
        ax[0].set_title("Original Data")
        ax[0].axis("off")  # Optionally turn off axes for better visualization

        ax[1].imshow(data)
        ax[1].set_title("CPT-like Data")
        ax[1].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close()


    return data


def extract_values_from_csv(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract the 'IC' column as a numpy array
    values = df['IC'].values

    # Ensure the output is a 1D ndarray
    values = np.reshape(values, (16384,))

    return values


def save_cptlike_csv(cptlike_img, output_folder, csv_file):
    """
    Save the CPT-like image array into a CSV file with the specified format.

    Args:
        cptlike_img (np.array): The CPT-like image data.
        output_folder (str): The directory where the CSV file will be saved.
        csv_file (str): The original CSV filename (used to name the output file).
    """

    # Get the shape of the array
    rows, cols = cptlike_img.shape

    # Create a list to store the formatted data
    data_list = []

    # Populate the list with index, column, row, and value
    index = 0
    for row in range(rows):
        for col in range(cols):
            data_list.append([index, col, row, cptlike_img[row, col]])
            index += 1

    # Convert list to DataFrame
    df_out = pd.DataFrame(data_list, columns=["index", "column", "row", "value"])

    # Generate output file path
    filename = os.path.splitext(os.path.basename(csv_file))[0] + "_cptlike.csv"
    output_path = os.path.join(output_folder, filename)

    # Save DataFrame as CSV
    df_out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def save_cptlike_csv_columnwise(cptlike_img, output_folder, csv_file):
    """
    Save the CPT-like image array into a CSV file in column-wise format.

    Args:
        cptlike_img (np.array): The CPT-like image data.
        output_folder (str): The directory where the CSV file will be saved.
        csv_file (str): The original CSV filename (used to name the output file).
    """

    # Get the shape of the array
    rows, cols = cptlike_img.shape

    # Create a list to store the formatted data
    data_list = []

    # Populate the list in column-wise order
    index = 0
    for col in range(cols):  # Iterate over columns first
        for row in range(rows):  # Then iterate over rows
            data_list.append([index, col, row, cptlike_img[row, col]])
            index += 1

    # Convert list to DataFrame
    df_out = pd.DataFrame(data_list, columns=["index", "col", "row", "IC"])

    # Generate output file path
    filename = os.path.splitext(os.path.basename(csv_file))[0] + "_cptlike.csv"
    output_path = os.path.join(output_folder, filename)

    # Save DataFrame as CSV
    df_out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


seed = np.random.randint(20532524)  # Generate a random seed using NumPy
np.random.seed(20234023)  # Set the seed for NumPy's random number generator


# Define input and output directories
csv_input = r"D:\schemaGAN\data\compare"
output_folder = r"D:\schemaGAN\data\BCS\cptlike"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all CSV files in the input directory
for csv_file in glob.glob(os.path.join(csv_input, "cs_*.csv")):
    values = extract_values_from_csv(csv_file)
    print(values)
    # Create a CPT-like image from the extracted values
    cptlike_img = create_cptlike_array(values, x_max=512, z_max=32, plot=True)
    print(cptlike_img)

    # Save the CPT-like image as a CSV (column-wise)
    save_cptlike_csv_columnwise(cptlike_img, output_folder, csv_file)

