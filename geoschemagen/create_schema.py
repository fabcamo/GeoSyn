import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geoschemagen.create_rfs import generate_rf_group
from geoschemagen.create_layer_boundaries import layer_boundary

def create_schema(output_folder: str, counter: int, z_max: int, x_max: int, seed: int = 20220412):
    """
    Generate synthetic data with given parameters and save results in the specified output folder.

    Args:
        output_folder (str): The folder to save the synthetic data.
        counter (int): Current realization number.
        z_max (int): Depth of the model.
        x_max (int): Length of the model.
        seed (int): Seed for random number generation.
    Returns:
        None
    """

    # Define the geometry for the synthetic data generation
    x_coord = np.arange(0, x_max, 1)  # Array of x coordinates
    z_coord = np.arange(0, z_max, 1)  # Array of z coordinates
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")  # 2D mesh of coordinates x, z

    # Generate random field models and shuffle them
    layers = generate_rf_group(seed)  # Store the random field models inside layers
    np.random.shuffle(layers)  # Shuffle the layers

    # Set up the matrix geometry
    matrix = np.zeros((z_max, x_max))  # Create the matrix of size {rows, cols}
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T  # Store the grid coordinates in a variable
    values = np.zeros(coords_to_list.shape[0])  # Create a matrix same as coords but with zeros

    # Generate new y value for each plot and sort them to avoid stacking
    y1 = layer_boundary(x_coord, z_max)
    y2 = layer_boundary(x_coord, z_max)
    y3 = layer_boundary(x_coord, z_max)
    y4 = layer_boundary(x_coord, z_max)
    boundaries = [y1, y2, y3, y4]  # Store the boundaries in a list
    boundaries = sorted(boundaries, key=lambda x: x[0])  # Sort the list to avoid stacking on top of each other

    # Create containers for each layer
    area_1, area_2, area_3, area_4, area_5 = [], [], [], [], []

    # Assign grid cells to each layer based on the boundaries
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row <= boundaries[0][col]:
                area_1.append([col, row])
            elif row <= boundaries[1][col]:
                area_2.append([col, row])
            elif row <= boundaries[2][col]:
                area_3.append([col, row])
            elif row <= boundaries[3][col]:
                area_4.append([col, row])
            else:
                area_5.append([col, row])

    # Apply the random field models to the layers
    all_layers = [area_1, area_2, area_3, area_4, area_5]
    for i, lst in enumerate(all_layers):
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]
        layer_IC = layers[i](layer_coordinates.T)
        values[mask] = layer_IC

    # Store the results in a dataframe
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    # Plot and save the results
    plt.clf()  # Clear the current figure
    df_pivot = df.pivot(index="z", columns="x", values="IC")
    fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
    ax.set_position([0, 0, 1, 1])
    ax.imshow(df_pivot)
    plt.axis("off")
    filename = f"cs_{counter}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    df.to_csv(csv_path)
    plt.close()
