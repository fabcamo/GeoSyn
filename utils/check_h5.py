import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_and_plot_h5(h5_path: str, scale_factor: float = 2):
    """
    Load an HDF5 file, check its contents, and plot the stored matrix exactly as it is.
    Adjust the figure size by a scale factor.

    Args:
        h5_path (str): Path to the HDF5 file.
        scale_factor (float): Factor by which to scale the figure size.
    """
    # Open the HDF5 file
    with h5py.File(h5_path, "r") as f:
        # Print the keys in the file (datasets and attributes)
        print("Contents of the HDF5 file:")
        for key in f.keys():
            print(f"  Dataset: {key}")

        # Print attributes
        print("\nAttributes:")
        for attr in f.attrs:
            print(f"  {attr}: {f.attrs[attr]}")

        # Access the image matrix dataset
        image_matrix = f["image_matrix"][:]

        # Check the shape of the loaded matrix
        z_max, x_max = image_matrix.shape
        print(f"Matrix shape: {image_matrix.shape}")

        # Calculate figure size based on the matrix dimensions (matching the original format)
        figsize = (x_max / 100 * scale_factor, z_max / 100 * scale_factor)

        # Create the figure once with the correct size
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the matrix without interpolation (use 'none')
        ax.imshow(image_matrix, cmap='viridis', interpolation='none', aspect='auto')  # aspect='auto'
        ax.axis("off")  # Turn off axis
        ax.set_title("Image Matrix from HDF5")

        # Show the plot
        plt.show()


# Example usage:
h5_file_path = r"D:\GeoSchemaGen\tests\typeA_RFTrue_20241209\typeA_1.h5"
load_and_plot_h5(h5_file_path, scale_factor=2)  # Increase size by factor of 2
