import h5py
import matplotlib.pyplot as plt


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
        image_matrix = f["ICvalues_matrix"][:]
        # Access the cpt-like image dataset
        cptlike_image = f["cptlike_matrix"][:]


        # Calculate figure size based on the matrix dimensions (matching the original format)
        #z_max, x_max = image_matrix.shape
        #figsize = (x_max / 100 * scale_factor, z_max / 50 * scale_factor)  # Adjusting for 2 plots vertically

        # Create a 2x1 grid for plotting
        fig, axs = plt.subplots(2, 1, figsize=(12,6))

        # Plot the image matrix
        axs[0].imshow(image_matrix, cmap='viridis', interpolation='none', aspect='auto')
        axs[0].axis("off")  # Turn off axis
        axs[0].set_title("Image Matrix from HDF5")

        # Plot the cpt-like image
        axs[1].imshow(cptlike_image, cmap='viridis', interpolation='none', aspect='auto')
        axs[1].axis("off")  # Turn off axis
        axs[1].set_title("CPT-like Matrix from HDF5")

        # Adjust layout for better appearance
        plt.tight_layout()

        # Show the plot
        plt.show()
# Example usage:
h5_file_path = r"D:\GeoSchemaGen\tests\typeB_RFTrue_20241212\typeB_2.h5"
load_and_plot_h5(h5_file_path, scale_factor=2)  # Increase size by factor of 2
