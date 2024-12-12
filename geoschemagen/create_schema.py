import os
import h5py
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geoschemagen.create_rfs import generate_rf_group
from geoschemagen.create_layer_boundaries import layer_boundary, layer_boundary_horizA, layer_boundary_irregular
from geoschemagen.create_layer_boundaries import layer_boundary_subhorizB, layer_boundary_lensC, layer_boundary_subhorizD_vert, layer_boundary_irregularE
from utils.create_cptlike import from_schema_to_cptlike, create_cptlike_array


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
    filename = f"cs_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    df.to_csv(csv_path)
    plt.close()


def create_schema_noRF(output_folder: str, counter: int, z_max: int, x_max: int, seed: int = 20220412):
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

    # Set up the matrix geometry
    matrix = np.zeros((z_max, x_max))  # Create the matrix of size {rows, cols}
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T  # Store the grid coordinates in a variable
    values = np.zeros(coords_to_list.shape[0])  # Create a matrix same as coords but with zeros

    # Generate new y value for each plot and sort them to avoid stacking
    y1 = layer_boundary_horizA(x_coord, z_max)
    y2 = layer_boundary_horizA(x_coord, z_max)
    y3 = layer_boundary_horizA(x_coord, z_max)
    y4 = layer_boundary_horizA(x_coord, z_max)
    boundaries = [y1, y2, y3, y4]  # Store the boundaries in a list
    boundaries = sorted(boundaries, key=lambda x: x[0])  # Sort the list to avoid stacking on top of each other

    # # Generate new y value for each plot and sort them to avoid stacking
    # y1 = layer_boundary_irregular(x_coord, z_max)
    # y2 = layer_boundary_irregular(x_coord, z_max)
    # y3 = layer_boundary_irregular(x_coord, z_max)
    # y4 = layer_boundary_irregular(x_coord, z_max)
    # boundaries = [y1, y2, y3, y4]  # Store the boundaries in a list
    # boundaries = sorted(boundaries, key=lambda x: x[0])  # Sort the list to avoid stacking on top of each other

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
        # Create a mask to select the grid cells for each layer
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]

        # # ORIGINAL METHOD: Apply the random value to each layer
        # # Generate a random value for the layer from 1 to 5
        # random_value = np.random.randint(1, 6)
        # # Apply the random value to the mask
        # values[mask] = random_value

        # COMPLETELY FIX ORDER: Apply the fixed value to each layer always
        # Get the i-layer value from an user defined list
        user_layer_values = [5, 4, 3, 2, 1]
        # Apply the user defined values to the mask
        values[mask] = user_layer_values[i]

        # TODO: Here you can also fix some of the vales and assign a random value to some positions
        # # HYBRID METHOD: Some values are fixed and some random
        # random_value = np.random.randint(3, 5)
        # # Create the user defined layer order
        # user_layer_values = [6, 6, random_value, random_value, 1]
        # # Apply the user defined values to the mask
        # values[mask] = user_layer_values[i]



    # Store the results in a dataframe
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    # Plot and save the results
    plt.clf()  # Clear the current figure
    df_pivot = df.pivot(index="z", columns="x", values="IC")
    fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
    ax.set_position([0, 0, 1, 1])
    ax.imshow(df_pivot)
    plt.axis("off")
    filename = f"cs_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    #df.to_csv(csv_path)
    plt.close()


def create_schema_one_layer(output_folder: str, counter: int, z_max: int, x_max: int, seed: int = 20220412):
    x_coord = np.arange(0, x_max, 1)
    z_coord = np.arange(0, z_max, 1)
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")

    layers = generate_rf_group(seed)
    np.random.shuffle(layers)

    matrix = np.zeros((z_max, x_max))
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
    values = np.zeros(coords_to_list.shape[0])

    area_1 = [[col, row] for row in range(matrix.shape[0]) for col in range(matrix.shape[1])]

    all_layers = [area_1]
    for i, lst in enumerate(all_layers):
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]
        layer_IC = layers[i](layer_coordinates.T)
        values[mask] = layer_IC

    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    plt.clf()
    df_pivot = df.pivot(index="z", columns="x", values="IC")
    fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
    ax.set_position([0, 0, 1, 1])
    ax.imshow(df_pivot)
    plt.axis("off")
    filename = f"cs_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    df.to_csv(csv_path)
    plt.close()


def create_schema_six_layers(output_folder: str, counter: int, z_max: int, x_max: int, seed: int = 20220412):
    x_coord = np.arange(0, x_max, 1)
    z_coord = np.arange(0, z_max, 1)
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")

    layers = generate_rf_group(seed)
    np.random.shuffle(layers)

    matrix = np.zeros((z_max, x_max))
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
    values = np.zeros(coords_to_list.shape[0])

    y1 = layer_boundary(x_coord, z_max)
    y2 = layer_boundary(x_coord, z_max)
    y3 = layer_boundary(x_coord, z_max)
    y4 = layer_boundary(x_coord, z_max)
    y5 = layer_boundary(x_coord, z_max)
    boundaries = [y1, y2, y3, y4, y5]
    boundaries = sorted(boundaries, key=lambda x: x[0])

    area_1, area_2, area_3, area_4, area_5, area_6 = [], [], [], [], [], []

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
            elif row <= boundaries[4][col]:
                area_5.append([col, row])
            else:
                area_6.append([col, row])

    all_layers = [area_1, area_2, area_3, area_4, area_5, area_6]
    for i, lst in enumerate(all_layers):
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]
        layer_IC = layers[i](layer_coordinates.T)
        values[mask] = layer_IC

    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    plt.clf()
    df_pivot = df.pivot(index="z", columns="x", values="IC")
    fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
    ax.set_position([0, 0, 1, 1])
    ax.imshow(df_pivot)
    plt.axis("off")
    filename = f"cs_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    df.to_csv(csv_path)
    plt.close()


def create_schema_seven_layers(output_folder: str, counter: int, z_max: int, x_max: int, seed: int = 20220412):
    x_coord = np.arange(0, x_max, 1)
    z_coord = np.arange(0, z_max, 1)
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")

    layers = generate_rf_group(seed)
    np.random.shuffle(layers)

    matrix = np.zeros((z_max, x_max))
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
    values = np.zeros(coords_to_list.shape[0])

    y1 = layer_boundary(x_coord, z_max)
    y2 = layer_boundary(x_coord, z_max)
    y3 = layer_boundary(x_coord, z_max)
    y4 = layer_boundary(x_coord, z_max)
    y5 = layer_boundary(x_coord, z_max)
    y6 = layer_boundary(x_coord, z_max)
    boundaries = [y1, y2, y3, y4, y5, y6]
    boundaries = sorted(boundaries, key=lambda x: x[0])

    area_1, area_2, area_3, area_4, area_5, area_6, area_7 = [], [], [], [], [], [], []

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
            elif row <= boundaries[4][col]:
                area_5.append([col, row])
            elif row <= boundaries[5][col]:
                area_6.append([col, row])
            else:
                area_7.append([col, row])

    all_layers = [area_1, area_2, area_3, area_4, area_5, area_6, area_7]
    for i, lst in enumerate(all_layers):
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]
        layer_IC = layers[i](layer_coordinates.T)
        values[mask] = layer_IC

    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    plt.clf()
    df_pivot = df.pivot(index="z", columns="x", values="IC")
    fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
    ax.set_position([0, 0, 1, 1])
    ax.imshow(df_pivot)
    plt.axis("off")
    filename = f"cs_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    df.to_csv(csv_path)
    plt.close()


def create_schema_eight_layers(output_folder: str, counter: int, z_max: int, x_max: int, seed: int = 20220412):
    x_coord = np.arange(0, x_max, 1)
    z_coord = np.arange(0, z_max, 1)
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")

    layers = generate_rf_group(seed)
    np.random.shuffle(layers)

    matrix = np.zeros((z_max, x_max))
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
    values = np.zeros(coords_to_list.shape[0])

    y1 = layer_boundary(x_coord, z_max)
    y2 = layer_boundary(x_coord, z_max)
    y3 = layer_boundary(x_coord, z_max)
    y4 = layer_boundary(x_coord, z_max)
    y5 = layer_boundary(x_coord, z_max)
    y6 = layer_boundary(x_coord, z_max)
    y7 = layer_boundary(x_coord, z_max)
    boundaries = [y1, y2, y3, y4, y5, y6, y7]
    boundaries = sorted(boundaries, key=lambda x: x[0])

    area_1, area_2, area_3, area_4, area_5, area_6, area_7, area_8 = [], [], [], [], [], [], [], []

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
            elif row <= boundaries[4][col]:
                area_5.append([col, row])
            elif row <= boundaries[5][col]:
                area_6.append([col, row])
            elif row <= boundaries[6][col]:
                area_7.append([col, row])
            else:
                area_8.append([col, row])

    all_layers = [area_1, area_2, area_3, area_4, area_5, area_6, area_7, area_8]
    for i, lst in enumerate(all_layers):
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]
        layer_IC = layers[i](layer_coordinates.T)
        values[mask] = layer_IC

    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    plt.clf()
    df_pivot = df.pivot(index="z", columns="x", values="IC")
    fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
    ax.set_position([0, 0, 1, 1])
    ax.imshow(df_pivot)
    plt.axis("off")
    filename = f"cs_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    df.to_csv(csv_path)
    plt.close()


def create_schema_eight_layers_noRF(output_folder: str, counter: int, z_max: int, x_max: int, seed: int = 20220412):
    x_coord = np.arange(0, x_max, 1)
    z_coord = np.arange(0, z_max, 1)
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")

    layers = generate_rf_group(seed)
    np.random.shuffle(layers)

    matrix = np.zeros((z_max, x_max))
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
    values = np.zeros(coords_to_list.shape[0])

    y1 = layer_boundary(x_coord, z_max)
    y2 = layer_boundary(x_coord, z_max)
    y3 = layer_boundary(x_coord, z_max)
    y4 = layer_boundary(x_coord, z_max)
    y5 = layer_boundary(x_coord, z_max)
    y6 = layer_boundary(x_coord, z_max)
    y7 = layer_boundary(x_coord, z_max)
    boundaries = [y1, y2, y3, y4, y5, y6, y7]
    boundaries = sorted(boundaries, key=lambda x: x[0])

    area_1, area_2, area_3, area_4, area_5, area_6, area_7, area_8 = [], [], [], [], [], [], [], []

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
            elif row <= boundaries[4][col]:
                area_5.append([col, row])
            elif row <= boundaries[5][col]:
                area_6.append([col, row])
            elif row <= boundaries[6][col]:
                area_7.append([col, row])
            else:
                area_8.append([col, row])

    all_layers = [area_1, area_2, area_3, area_4, area_5, area_6, area_7, area_8]
    for i, lst in enumerate(all_layers):
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]
        # Generate a random value for the layer from 1 to 5
        random_value = np.random.randint(1, 9)
        values[mask] = random_value

    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    plt.clf()
    df_pivot = df.pivot(index="z", columns="x", values="IC")
    fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
    ax.set_position([0, 0, 1, 1])
    ax.imshow(df_pivot)
    plt.axis("off")
    filename = f"cs_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    df.to_csv(csv_path)
    plt.close()


def create_schema_typeA_OLD(output_folder: str, counter: int, z_max: int, x_max: int, trigo_type: int, seed: int, RF: bool = False) -> None:
    """
    Generate synthetic data with given parameters and save results in the specified output folder.
    Type A:
    - Up to 4 layers
    - Subhorizontal layers
    - No indentations (only cos or sine)
    - Fixed bottom layer and upper layer

    Args:
        output_folder (str): The folder to save the synthetic data.
        counter (int): Current realization number.
        z_max (int): Depth of the model.
        x_max (int): Length of the model.
        trigo_type (int): Type of trigonometric function to use, with 1 for sine and 2 for cosine.
        seed (int): Seed for random number generation.
        RF (bool): Whether to use Random Fields. Default is False.

    Returns:
        None
    """

    # Define the geometry for the synthetic data generation
    x_coord = np.arange(0, x_max, 1)  # Array of x coordinates
    z_coord = np.arange(0, z_max, 1)  # Array of z coordinates
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")  # 2D mesh of coordinates x, z

    # Set up the matrix geometry
    matrix = np.zeros((z_max, x_max))  # Create the matrix of size {rows, cols}
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T  # Store the grid coordinates in a variable
    values = np.zeros(coords_to_list.shape[0])  # Create a matrix same as coords but with zeros

    # Generate new y value for each plot and sort them to avoid stacking
    y1 = layer_boundary_horizA(x_coord, z_max, trigo_type)
    y2 = layer_boundary_horizA(x_coord, z_max, trigo_type)
    y3 = layer_boundary_horizA(x_coord, z_max, trigo_type)
    boundaries = [y1, y2, y3]  # Store the boundaries in a list
    boundaries = sorted(boundaries, key=lambda x: x[0])  # Sort the list to avoid stacking on top of each other


    # Create containers for each layer
    area_1, area_2, area_3, area_4, = [], [], [], []

    # Assign grid cells to each layer based on the boundaries
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row <= boundaries[0][col]:
                area_1.append([col, row])
            elif row <= boundaries[1][col]:
                area_2.append([col, row])
            elif row <= boundaries[2][col]:
                area_3.append([col, row])
            else:
                area_4.append([col, row])

    # Fill the layers with the corresponding values
    if RF == True:
        #TODO: Think if you want to fix some layers like in the No RF case
        # Generate random field models and shuffle them
        layers = generate_rf_group(seed)  # Store the random field models inside layers
        np.random.shuffle(layers)  # Shuffle the layers

        # Apply the random field models to the layers
        all_layers = [area_1, area_2, area_3, area_4]
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            layer_coordinates = coords_to_list[mask]
            layer_IC = layers[i](layer_coordinates.T)
            values[mask] = layer_IC

    elif RF == False:
        # Apply the random field models to the layers
        all_layers = [area_1, area_2, area_3, area_4]
        for i, lst in enumerate(all_layers):
            # Create a mask to select the grid cells for each layer
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            # Choose random value from 2, 3, 4, 5 with equal probability
            random_value = np.random.choice([2, 3])
            # Get the i-layer value from an user defined list
            user_layer_values = [4, 3, random_value, 1]
            # Apply the user defined values to the mask
            values[mask] = user_layer_values[i]


    # Store the results in a dataframe
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    # Plot and save the results
    plt.clf()  # Clear the current figure
    df_pivot = df.pivot(index="z", columns="x", values="IC")
    fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
    ax.set_position([0, 0, 1, 1])
    ax.imshow(df_pivot)
    plt.axis("off")
    filename = f"typeA_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    #df.to_csv(csv_path)
    plt.close()

########################################################################################################################
# NEW SCHEMAS FOR v.2.0
########################################################################################################################

def create_schema_typeA(output_folder: str,
                        counter: int,
                        z_max: int,
                        x_max: int,
                        trigo_type: int,
                        seed: int,
                        RF: bool = False,
                        create_cptlike: bool = False,
                        save_image: bool = False,
                        save_cptlike_image: bool = False,
                        save_csv: bool = False) -> None:
    """
    Generate synthetic data with given parameters, save results in an HDF5 file, and optionally save the image.

    Args:
        output_folder (str): The folder to save the synthetic data.
        counter (int): Current realization number.
        z_max (int): Depth of the model.
        x_max (int): Length of the model.
        trigo_type (int): Type of trigonometric function to use, with 1 for sine and 2 for cosine.
        seed (int): Seed for random number generation.
        RF (bool): Whether to use Random Fields. Default is False.
        create_cptlike (bool): Whether to create a CPT-like model. Default is False.
        save_image (bool): Whether to save the PNG image. Default is False.
        save_cptlike_image (bool): Whether to save the CPT-like PNG image. Default is False.
        save_csv (bool): Whether to save the CSV file. Default is False.

    Returns:
        None
    """

    # Define the geometry for the synthetic data generation
    x_coord = np.arange(0, x_max, 1)  # Array of x coordinates
    z_coord = np.arange(0, z_max, 1)  # Array of z coordinates
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")  # 2D mesh of coordinates x, z

    # Set up the matrix geometry
    matrix = np.zeros((z_max, x_max))  # Create the matrix of size {rows, cols}
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T  # Store the grid coordinates in a variable
    values = np.zeros(coords_to_list.shape[0])  # Create a matrix same as coords but with zeros

    # Generate new y value for each plot and sort them to avoid stacking
    y1 = layer_boundary_horizA(x_coord, z_max, trigo_type)
    y2 = layer_boundary_horizA(x_coord, z_max, trigo_type)
    y3 = layer_boundary_horizA(x_coord, z_max, trigo_type)
    boundaries = [y1, y2, y3]  # Store the boundaries in a list
    boundaries = sorted(boundaries, key=lambda x: x[0])  # Sort the list to avoid stacking on top of each other

    # Create containers for each layer
    area_1, area_2, area_3, area_4 = [], [], [], []

    # Assign grid cells to each layer based on the boundaries
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row <= boundaries[0][col]:
                area_1.append([col, row])
            elif row <= boundaries[1][col]:
                area_2.append([col, row])
            elif row <= boundaries[2][col]:
                area_3.append([col, row])
            else:
                area_4.append([col, row])

    # Fill the layers with the corresponding values
    if RF:
        # Generate random field models and shuffle them
        layers_with_names = generate_rf_group(seed)  # Store the random field models and names
        # Define the order of the layers
        random_choice = np.random.choice([2, 3]) # Choose random value from 2, 3 with equal probability
        if random_choice == 2:
            random_value = layers_with_names[2]
        elif random_choice == 3:
            random_value = layers_with_names[3]
        else:
            raise ValueError("Invalid random choice in RF model B")

        # Combine the random and fixed layers
        #TODO: Check if this is the order that I want
        my_layers = [layers_with_names[4],
                     layers_with_names[3],
                     random_value,
                     layers_with_names[1]]

        materials_list = [] # Create a list to store the material names
        # Apply the random field models to the layers
        all_layers = [area_1, area_2, area_3, area_4]
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            layer_coordinates = coords_to_list[mask]

            # Extract the random field and material name
            layer_rf, material_name = my_layers[i]
            layer_IC = layer_rf(layer_coordinates.T)
            values[mask] = layer_IC
            # Append the material name to the materials list
            materials_list.append(material_name)

    else:
        # Apply the discrete values to the layers
        all_layers = [area_1, area_2, area_3, area_4]
        random_value = np.random.choice([2, 3])  # Choose random value from 2, 3 with equal probability
        user_layer_values = [4, 3, random_value, 1]  # Get the i-layer value from an user defined list
        # Append the value used in each layer to a list
        materials_list = user_layer_values
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            values[mask] = user_layer_values[i]

    # Create the cptlike data that accompanies the synthetic data if create_cptlike is True
    if create_cptlike:
        cpt_like_image = create_cptlike_array(image_matrix=values, x_max=x_max, z_max=z_max)


    # Store the results in a DataFrame (for plotting image)
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    # Save to HDF5
    h5_filename = f"typeA_{counter + 1}.h5"
    h5_path = os.path.join(output_folder, h5_filename)
    with h5py.File(h5_path, "w") as f:
        # Save the 2D array (image matrix) as a dataset
        # Make sure to save the matrix with the correct orientation
        f.create_dataset("ICvalues_matrix", data=values.reshape(x_max, z_max).T)  # Correctly reshape for z, x
        f.create_dataset("cptlike_matrix", data=cpt_like_image)  # Save the cptlike data

        # Save metadata as attributes
        f.attrs["model_type"] = "A"
        f.attrs["matrix_shape"] = values.reshape(x_max, z_max).T.shape
        #TODO: Add a description that makes sense for the model
        f.attrs["description"] = "Deltaic area with subhorizontal layers and the pleistocene sand as base layer at 30 m depth"
        f.attrs["date"] = str(datetime.datetime.now())
        f.attrs["seed"] = seed
        f.attrs["randomfield"] = RF
        f.attrs["materials"] = materials_list

    print(f"Data saved as {h5_filename}")

    # Optionally, save the image as a PNG file
    if save_image:
        plt.clf()  # Clear the current figure
        df_pivot = df.pivot(index="z", columns="x", values="IC")
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(df_pivot, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")

    # Optionally, save the CSV file
    if save_csv:
        csv_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.csv')}")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved as {csv_path}")

    # Optionally, save the CPTlike image as a PNG file
    if save_cptlike_image:
        plt.clf()  # Clear the current figure
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(cpt_like_image, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"cptlike_{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")



def create_schema_typeB(output_folder: str,
                        counter: int,
                        z_max: int,
                        x_max: int,
                        trigo_type: int,
                        seed: int,
                        RF: bool = False,
                        create_cptlike: bool = False,
                        save_image: bool = False,
                        save_cptlike_image: bool = False,
                        save_csv: bool = False) -> None:
    """
    Generate synthetic data with given parameters and save results in the specified output folder.
    Type B:
    - Up to 5 layers
    - Subhorizontal layers
    - Indentations (combination of cos and sine) possible
    - Fixed bottom layer

    Args:
        output_folder (str): The folder to save the synthetic data.
        counter (int): Current realization number.
        z_max (int): Depth of the model.
        x_max (int): Length of the model.
        seed (int): Seed for random number generation.
        RF (bool): Whether to use Random Fields. Default is False.
        create_cptlike (bool): Whether to create a CPT-like data. Default is False.
        save_image (bool): Whether to save the PNG image. Default is False.
        save_cptlike_image (bool): Whether to save the CPT-like PNG image. Default is False.
        save_csv (bool): Whether to save the CSV file. Default is False.

    Returns:
        None
    """

    # Define the geometry for the synthetic data generation
    x_coord = np.arange(0, x_max, 1)  # Array of x coordinates
    z_coord = np.arange(0, z_max, 1)  # Array of z coordinates
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")  # 2D mesh of coordinates x, z

    # Set up the matrix geometry
    matrix = np.zeros((z_max, x_max))  # Create the matrix of size {rows, cols}
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T  # Store the grid coordinates in a variable
    values = np.zeros(coords_to_list.shape[0])  # Create a matrix same as coords but with zeros

    # Generate new y value for each plot and sort them to avoid stacking
    y1 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 0, 3)
    y2 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 4, 6)
    y3 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 9, 11)
    y4 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 14, 15)
    y5 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 17, 21)
    boundaries = [y1, y2, y3, y4, y5]  # Store the boundaries in a list
    boundaries = sorted(boundaries, key=lambda x: x[0])  # Sort the list to avoid stacking on top of each other


    # Create containers for each layer
    area_1, area_2, area_3, area_4, area_5, area_6 = [], [], [], [], [], []

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
            elif row <= boundaries[4][col]:
                area_5.append([col, row])
            else:
                area_6.append([col, row])

    # Fill the layers with the corresponding values
    if RF:
        # TODO: Think if you want to fix some layers like in the No RF case
        # Generate random field models and shuffle them
        layers_with_names = generate_rf_group(seed)  # Store the random field models inside layers
        # Define the order of the layers
        random_choice = np.random.choice([4, 1])  # Choose random value from 2, 3 with equal probability
        if random_choice == 4:
            random_value = layers_with_names[4]
        elif random_choice == 1:
            random_value = layers_with_names[1]
        else:
            raise ValueError("Invalid random choice in RF model A")

        # Then, create my own order of layers with some layers that are assigned randomly
        #TODO: Check if this is the order that I want
        my_layers = [random_value,
                     layers_with_names[1],
                     random_value,
                     layers_with_names[3],
                     layers_with_names[2],
                     layers_with_names[5]]

        # Create a list to store the materials used in each layer
        materials_list = []
        # Apply the random field models to the layers
        all_layers = [area_1, area_2, area_3, area_4, area_5, area_6]
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            layer_coordinates = coords_to_list[mask]
            # Extract the random field and material name
            layer_rf, material_name = my_layers[i]
            layer_IC = layer_rf(layer_coordinates.T)
            values[mask] = layer_IC
            # Append the material name to the materials list
            materials_list.append(material_name)

    else:
        # Apply the discrete values to the layers
        all_layers = [area_1, area_2, area_3, area_4, area_5, area_6]
        random_value = np.random.choice([4, 1]) # Choose random value from 4 or 1 with equal probability
        user_layer_values = [random_value, 1, random_value, 3, 2, 5] # Get the i-layer value from an user defined list
        # Append the value used in each layer to a list
        materials_list = user_layer_values
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            values[mask] = user_layer_values[i]

    # Create the cptlike data that accompanies the synthetic data if create_cptlike is True
    if create_cptlike:
        cpt_like_image = create_cptlike_array(image_matrix=values, x_max=x_max, z_max=z_max)

    # Store the results in a dataframe
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    # Save to HDF5
    h5_filename = f"typeB_{counter + 1}.h5"
    h5_path = os.path.join(output_folder, h5_filename)
    with h5py.File(h5_path, "w") as f:
        # Save the 2D array (image matrix) as a dataset
        # Make sure to save the matrix with the correct orientation
        f.create_dataset("ICvalues_matrix", data=values.reshape(x_max, z_max).T)  # Correctly reshape for z, x
        f.create_dataset("cptlike_matrix", data=cpt_like_image)  # Save the cptlike data

        # Save metadata as attributes
        f.attrs["model_type"] = "B"
        f.attrs["matrix_shape"] = values.reshape(x_max, z_max).T.shape
        #TODO: Add a description that makes sense for the model
        f.attrs["description"] = "Deltaic transition area with subhorizontal layers and complex indentations with the pleistocene sand as base layer at 30 m depth"
        f.attrs["date"] = str(datetime.datetime.now())
        f.attrs["seed"] = seed
        f.attrs["randomfield"] = RF
        f.attrs["materials"] = materials_list

    print(f"Data saved as {h5_filename}")

    # Optionally, save the image as a PNG file
    if save_image:
        plt.clf()  # Clear the current figure
        df_pivot = df.pivot(index="z", columns="x", values="IC")
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(df_pivot, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")

    # Optionally, save the CSV file
    if save_csv:
        csv_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.csv')}")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved as {csv_path}")

    # Optionally, save the CPTlike image as a PNG file
    if save_cptlike_image:
        plt.clf()  # Clear the current figure
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(cpt_like_image, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"cptlike_{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")


def create_schema_typeC(output_folder: str,
                        counter: int,
                        z_max: int,
                        x_max: int,
                        trigo_type: int,
                        seed: int,
                        RF: bool = False,
                        create_cptlike: bool = False,
                        save_image: bool = False,
                        save_cptlike_image: bool = False,
                        save_csv: bool = False) -> None:
    """
    Generate synthetic data with given parameters and save results in the specified output folder.
    Type C:
    - Up to 5 layers
    - Subhorizontal layers with lenses
    - Indentations (combination of cos and sine) possible
    - Fixed bottom layer

    Args:
        output_folder (str): The folder to save the synthetic data.
        counter (int): Current realization number.
        z_max (int): Depth of the model.
        x_max (int): Length of the model.
        seed (int): Seed for random number generation.
        RF (bool): Whether to use Random Fields. Default is False.
        create_cptlike (bool): Whether to create the CPT-like data. Default is False.
        save_image (bool): Whether to save the PNG image. Default is False.
        save_cptlike_image (bool): Whether to save the CPT-like PNG image. Default is False.
        save_csv (bool): Whether to save the CSV file. Default is False.

    Returns:
        None
    """

    # Define the geometry for the synthetic data generation
    x_coord = np.arange(0, x_max, 1)  # Array of x coordinates
    z_coord = np.arange(0, z_max, 1)  # Array of z coordinates
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")  # 2D mesh of coordinates x, z

    # Set up the matrix geometry
    matrix = np.zeros((z_max, x_max))  # Create the matrix of size {rows, cols}
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T  # Store the grid coordinates in a variable
    values = np.zeros(coords_to_list.shape[0])  # Create a matrix same as coords but with zeros

    # Generate new y value for each plot and sort them to avoid stacking
    y1 = layer_boundary_lensC(x_coord, z_max, trigo_type)
    y2 = layer_boundary_subhorizB(x_coord, z_max, trigo_type)
    y3 = layer_boundary_subhorizB(x_coord, z_max, trigo_type)
    boundaries = [y1, y2, y3]  # Store the boundaries in a list
    boundaries = sorted(boundaries, key=lambda x: x[0])  # Sort the list to avoid stacking on top of each other

    # Create containers for each layer
    area_1, area_2, area_3, area_4 = [], [], [], []

    # Assign grid cells to each layer based on the boundaries
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row <= boundaries[0][col]:
                area_1.append([col, row])
            elif row <= boundaries[1][col]:
                area_2.append([col, row])
            elif row <= boundaries[2][col]:
                area_3.append([col, row])
            else:
                area_4.append([col, row])

    # Fill the layers with the corresponding values
    if RF:
        # TODO: Think if you want to fix some layers like in the No RF case
        # Generate random field models and shuffle them
        layers_with_names = generate_rf_group(seed)  # Store the random field models and names
        np.random.shuffle(layers_with_names)  # Shuffle the layers with their names
        # Create a list to store the materials used in each layer
        materials_list = []
        # Apply the random field models to the layers
        all_layers = [area_1, area_2, area_3, area_4]
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            layer_coordinates = coords_to_list[mask]

            # Extract the random field and material name
            layer_rf, material_name = layers_with_names[i]
            layer_IC = layer_rf(layer_coordinates.T)
            values[mask] = layer_IC
            # Append the material name to the materials list
            materials_list.append(material_name)

    else:
        # Apply the discrete values to the layers
        all_layers = [area_1, area_2, area_3, area_4]
        user_layer_values = [2, 4, 1, 3] # Define the values for each layer
        # Append the value used in each layer to a list
        materials_list = user_layer_values
        for i, lst in enumerate(all_layers):
            # Create a mask to select the grid cells for each layer
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            # Apply the user defined values to the mask
            values[mask] = user_layer_values[i]

    # Create the cptlike data that accompanies the synthetic data if create_cptlike is True
    if create_cptlike:
        cpt_like_image = create_cptlike_array(image_matrix=values, x_max=x_max, z_max=z_max)

    # Store the results in a DataFrame (for plotting image)
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    # Save to HDF5
    h5_filename = f"typeC_{counter + 1}.h5"
    h5_path = os.path.join(output_folder, h5_filename)
    with h5py.File(h5_path, "w") as f:
        # Save the 2D array (image matrix) as a dataset
        # Make sure to save the matrix with the correct orientation
        f.create_dataset("ICvalues_matrix", data=values.reshape(x_max, z_max).T)  # Correctly reshape for z, x
        f.create_dataset("cptlike_matrix", data=cpt_like_image)  # Save the cptlike data

        # Save metadata as attributes
        f.attrs["model_type"] = "C"
        f.attrs["matrix_shape"] = values.reshape(x_max, z_max).T.shape
        # TODO: Add a description that makes sense for the model
        f.attrs["description"] = "Deltaic area with subhorizontal layers and the pleistocene sand as base layer at 30 m depth"
        f.attrs["date"] = str(datetime.datetime.now())
        f.attrs["seed"] = seed
        f.attrs["randomfield"] = RF
        f.attrs["materials"] = materials_list

    print(f"Data saved as {h5_filename}")

    # Optionally, save the image as a PNG file
    if save_image:
        plt.clf()  # Clear the current figure
        df_pivot = df.pivot(index="z", columns="x", values="IC")
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(df_pivot, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")

    # Optionally, save the CSV file
    if save_csv:
        csv_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.csv')}")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved as {csv_path}")

    # Optionally, save the CPTlike image as a PNG file
    if save_cptlike_image:
        plt.clf()  # Clear the current figure
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(cpt_like_image, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"cptlike_{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")


def create_schema_typeD(output_folder: str,
                        counter: int,
                        z_max: int,
                        x_max: int,
                        trigo_type: int,
                        seed: int,
                        RF: bool = False,
                        create_cptlike: bool = False,
                        save_image: bool = False,
                        save_cptlike_image: bool = False,
                        save_csv: bool = False) -> None:
    """
    Generate synthetic data with given parameters and save results in the specified output folder.
    Type D:
    - Up to 6 layers
    - Subhorizontal layers intercalations of 2 soil types
    - Indentations (combination of cos and sine) possible
    - Fixed bottom layer

    Args:
        output_folder (str): The folder to save the synthetic data.
        counter (int): Current realization number.
        z_max (int): Depth of the model.
        x_max (int): Length of the model.
        seed (int): Seed for random number generation.
        RF (bool): Whether to use Random Fields. Default is False.
        create_cptlike (bool): Whether to create the CPT-like data. Default is False.
        save_image (bool): Whether to save the PNG image. Default is False.
        save_cptlike_image
        save_csv (bool): Whether to save the CSV file. Default is False.

    Returns:
        None
    """
    # Define the geometry for the synthetic data generation
    x_coord = np.arange(0, x_max, 1)  # Array of x coordinates
    z_coord = np.arange(0, z_max, 1)  # Array of z coordinates
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")  # 2D mesh of coordinates x, z

    # Set up the matrix geometry
    matrix = np.zeros((z_max, x_max))  # Create the matrix of size {rows, cols}
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T  # Store the grid coordinates in a variable
    values = np.zeros(coords_to_list.shape[0])  # Create a matrix same as coords but with zeros

    # Generate new y value for each plot and sort them to avoid stacking
    y1 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 0, 2)
    y2 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 4, 6)
    y3 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 9, 11)
    y4 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 14, 15)
    y5 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 17, 21)
    y6 = layer_boundary_subhorizD_vert(x_coord, z_max, trigo_type, 20, 29)
    boundaries = [y1, y2, y3, y4, y5, y6]  # Store the boundaries in a list
    boundaries = sorted(boundaries, key=lambda x: x[0])  # Sort the list to avoid stacking on top of each other

    # Create containers for each layer
    area_1, area_2, area_3, area_4, area_5, area_6, area_7 = [], [], [], [], [], [], []

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
            elif row <= boundaries[4][col]:
                area_5.append([col, row])
            elif row <= boundaries[5][col]:
                area_6.append([col, row])
            else:
                area_7.append([col, row])

    # Fill the layers with the corresponding values
    if RF:
        # Generate random field models and shuffle them
        layers_with_names = generate_rf_group(seed)  # Store the random field models inside layers
        # Define my own order of layers
        # First, let at random choose between two layers
        random_choice = np.random.choice([0, 1])
        if random_choice == 0:
            random_value = layers_with_names[4]
        elif random_choice == 1:
            random_value = layers_with_names[0]
        else:
            raise ValueError("Invalid random choice in RF model A")
        # Then, create my own order of layers with some layers that are assigned randomly
        my_layers = [layers_with_names[0],
                     random_value,
                     layers_with_names[4],
                     layers_with_names[0],
                     layers_with_names[4],
                     random_value,
                     layers_with_names[3]]

        # Create a list to store the materials used in each layer
        materials_list = []
        # Apply the random field models to the layers
        all_layers = [area_1, area_2, area_3, area_4, area_5, area_6, area_7]
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            layer_coordinates = coords_to_list[mask]
            # Extract the random field and material name
            layer_rf, material_name = my_layers[i]
            layer_IC = layer_rf(layer_coordinates.T)
            values[mask] = layer_IC
            # Append the material name to the materials list
            materials_list.append(material_name)

    else:
        # Apply the discrete values to the layers
        all_layers = [area_1, area_2, area_3, area_4, area_5, area_6, area_7]
        random_value = np.random.choice([5, 6]) # Choose random value from 5, 6 with equal probability
        # Generate random non-repeating values for the layers using NumPy
        user_layer_values = [6, random_value, 5, 6, 5, random_value, 1]
        # Append the value used in each layer to a list
        materials_list = user_layer_values
        for i, lst in enumerate(all_layers):
            # Create a mask to select the grid cells for each layer
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            # Apply the user defined values to the mask
            values[mask] = user_layer_values[i]

    # Create the cptlike data that accompanies the synthetic data if create_cptlike is True
    if create_cptlike:
        cpt_like_image = create_cptlike_array(image_matrix=values, x_max=x_max, z_max=z_max)

    # Store the results in a dataframe
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    # Save to HDF5
    h5_filename = f"typeD_{counter + 1}.h5"
    h5_path = os.path.join(output_folder, h5_filename)
    with h5py.File(h5_path, "w") as f:
        # Save the 2D array (image matrix) as a dataset
        # Make sure to save the matrix with the correct orientation
        f.create_dataset("ICvalues_matrix", data=values.reshape(x_max, z_max).T)  # Correctly reshape for z, x
        f.create_dataset("cptlike_matrix", data=cpt_like_image)  # Save the cptlike data

        # Save metadata as attributes
        f.attrs["model_type"] = "D"
        f.attrs["matrix_shape"] = values.reshape(x_max, z_max).T.shape
        # TODO: Add a description that makes sense for the model
        f.attrs["description"] = "Deltaic area with subhorizontal layers and the pleistocene sand as base layer at 30 m depth"
        f.attrs["date"] = str(datetime.datetime.now())
        f.attrs["seed"] = seed
        f.attrs["randomfield"] = RF
        f.attrs["materials"] = materials_list

    print(f"Data saved as {h5_filename}")

    # Optionally, save the image as a PNG file
    if save_image:
        plt.clf()  # Clear the current figure
        df_pivot = df.pivot(index="z", columns="x", values="IC")
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(df_pivot, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")

    # Optionally, save the CSV file
    if save_csv:
        csv_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.csv')}")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved as {csv_path}")

    # Optionally, save the CPTlike image as a PNG file
    if save_cptlike_image:
        plt.clf()  # Clear the current figure
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(cpt_like_image, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"cptlike_{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")


def create_schema_typeE(output_folder: str,
                        counter: int,
                        z_max: int,
                        x_max: int,
                        trigo_type: bool,
                        seed: int,
                        RF: bool = False,
                        create_cptlike: bool = False,
                        save_image: bool = False,
                        save_cptlike_image: bool = False,
                        save_csv: bool = False) -> None:
    """
    Generate synthetic data with given parameters and save results in the specified output folder.
    Type A:
    - Up to X layers
    - Inclined layers
    - No indentations (only cos or sine)
    - Fixed bottom layer and upper layer

    Args:
        output_folder (str): The folder to save the synthetic data.
        counter (int): Current realization number.
        z_max (int): Depth of the model.
        x_max (int): Length of the model.
        combine_trigo (bool): Type of trigonometric function to use.
        seed (int): Seed for random number generation.
        RF (bool): Whether to use Random Fields. Default is False.
        create_cptlike (bool): Whether to create the CPT-like data. Default is False.
        save_image (bool): Whether to save the PNG image. Default is False.
        save_cptlike_image (bool): Whether to save the CPT-like PNG image. Default is False.
        save_csv (bool): Whether to save the CSV file. Default is False.

    Returns:
        None
    """

    # Define the geometry for the synthetic data generation
    x_coord = np.arange(0, x_max, 1)  # Array of x coordinates
    z_coord = np.arange(0, z_max, 1)  # Array of z coordinates
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")  # 2D mesh of coordinates x, z

    # Set up the matrix geometry
    matrix = np.zeros((z_max, x_max))  # Create the matrix of size {rows, cols}
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T  # Store the grid coordinates in a variable
    values = np.zeros(coords_to_list.shape[0])  # Create a matrix same as coords but with zeros

    # Check if combine_trigo is True, if it is, choose at random between sine and cosine, else, stick with one for all layers
    if trigo_type == True:
        trigo_type = 0
    else:
        trigo_type = np.random.choice([1, 2])

    # Generate new y value for each plot and sort them to avoid stacking
    y1 = layer_boundary_irregularE(x_coord, z_max, trigo_type)
    y2 = layer_boundary_irregularE(x_coord, z_max, trigo_type)
    y3 = layer_boundary_irregularE(x_coord, z_max, trigo_type)
    y4 = layer_boundary_irregularE(x_coord, z_max, trigo_type)
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

    # Fill the layers with the corresponding values
    if RF:
        # TODO: Think if you want to fix some layers like in the No RF case
        # Generate random field models and shuffle them
        layers_with_names = generate_rf_group(seed)  # Store the random field models and names
        np.random.shuffle(layers_with_names)  # Shuffle the layers with their names
        # Create a list to store the materials used in each layer
        materials_list = []

        # Apply the random field models to the layers
        all_layers = [area_1, area_2, area_3, area_4, area_5]
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            layer_coordinates = coords_to_list[mask]

            # Extract the random field and material name
            layer_rf, material_name = layers_with_names[i]
            layer_IC = layer_rf(layer_coordinates.T)
            values[mask] = layer_IC
            # Append the material name to the materials list
            materials_list.append(material_name)


    else:
        # Apply the discrete values to the layers
        all_layers = [area_1, area_2, area_3, area_4, area_5]
        # Choose random value from 2, 4 with equal probability
        random_value = np.random.choice([2, 4])
        # Get the i-layer value from an user defined list
        user_layer_values = [5, random_value, 3, random_value, 1]

        # Append the value used in each layer to a list
        materials_list = user_layer_values
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            values[mask] = user_layer_values[i]

    # Create the cptlike data that accompanies the synthetic data if create_cptlike is True
    if create_cptlike:
        cpt_like_image = create_cptlike_array(image_matrix=values, x_max=x_max, z_max=z_max)

    # Store the results in a dataframe
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    # Save to HDF5
    h5_filename = f"typeE_{counter + 1}.h5"
    h5_path = os.path.join(output_folder, h5_filename)
    with h5py.File(h5_path, "w") as f:
        # Save the 2D array (image matrix) as a dataset
        # Make sure to save the matrix with the correct orientation
        f.create_dataset("ICvalues_matrix", data=values.reshape(x_max, z_max).T)  # Correctly reshape for z, x
        f.create_dataset("cptlike_matrix", data=cpt_like_image)  # Save the cptlike data

        # Save metadata as attributes
        f.attrs["model_type"] = "E"
        f.attrs["matrix_shape"] = values.reshape(x_max, z_max).T.shape
        # TODO: Add a description that makes sense for the model
        f.attrs["description"] = "Deltaic area with subhorizontal layers and the pleistocene sand as base layer at 30 m depth"
        f.attrs["date"] = str(datetime.datetime.now())
        f.attrs["seed"] = seed
        f.attrs["randomfield"] = RF
        f.attrs["materials"] = materials_list

    print(f"Data saved as {h5_filename}")

    # Optionally, save the image as a PNG file
    if save_image:
        plt.clf()  # Clear the current figure
        df_pivot = df.pivot(index="z", columns="x", values="IC")
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(df_pivot, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")

    # Optionally, save the CSV file
    if save_csv:
        csv_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.csv')}")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved as {csv_path}")

    # Optionally, save the CPTlike image as a PNG file
    if save_cptlike_image:
        plt.clf()  # Clear the current figure
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(cpt_like_image, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"cptlike_{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")


def create_schema_typeF(output_folder: str,
                        counter: int,
                        z_max: int,
                        x_max: int,
                        seed: int,
                        RF: bool = False,
                        create_cptlike: bool = False,
                        save_image: bool = False,
                        save_cptlike_image: bool = False,
                        save_csv: bool = False) -> None:
    """
    Generate synthetic data with given parameters and save results in the specified output folder.
    Type A:
    - Up to X layers
    - Inclined layers
    - No indentations (only cos or sine)
    - Fixed bottom layer and upper layer

    Args:
        output_folder (str): The folder to save the synthetic data.
        counter (int): Current realization number.
        z_max (int): Depth of the model.
        x_max (int): Length of the model.
        seed (int): Seed for random number generation.
        RF (bool): Whether to use Random Fields. Default is False.
        create_cptlike (bool): Whether to create the CPT-like data. Default is False.
        save_image (bool): Whether to save the PNG image. Default is False.
        save_cptlike_image (bool): Whether to save the CPT-like PNG image. Default is False.
        save_csv (bool): Whether to save the CSV file. Default is False.

    Returns:
        None
    """

    # Define the geometry for the synthetic data generation
    x_coord = np.arange(0, x_max, 1)  # Array of x coordinates
    z_coord = np.arange(0, z_max, 1)  # Array of z coordinates
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")  # 2D mesh of coordinates x, z

    # Set up the matrix geometry
    matrix = np.zeros((z_max, x_max))  # Create the matrix of size {rows, cols}
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T  # Store the grid coordinates in a variable
    values = np.zeros(coords_to_list.shape[0])  # Create a matrix same as coords but with zeros


    # Generate new y value for each plot and sort them to avoid stacking
    y1 = layer_boundary_irregular(x_coord, z_max)
    y2 = layer_boundary_irregular(x_coord, z_max)
    y3 = layer_boundary_irregular(x_coord, z_max)
    y4 = layer_boundary_irregular(x_coord, z_max)
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

    # Fill the layers with the corresponding values
    if RF:
        # TODO: Think if you want to fix some layers like in the No RF case
        # Generate random field models and shuffle them
        layers_with_names = generate_rf_group(seed)  # Store the random field models inside layers
        np.random.shuffle(layers_with_names)  # Shuffle the layers
        # Create a list to store the materials used in each layer
        materials_list = []
        # Apply the random field models to the layers
        all_layers = [area_1, area_2, area_3, area_4, area_5]
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            layer_coordinates = coords_to_list[mask]

            # Extract the random field and material name
            layer_rf, material_name = layers_with_names[i]
            layer_IC = layer_rf(layer_coordinates.T)
            values[mask] = layer_IC
            # Append the material name to the materials list
            materials_list.append(material_name)

    else:
        # Apply the random field models to the layers
        all_layers = [area_1, area_2, area_3, area_4, area_5]
        random_value = np.random.choice([2, 4]) # Choose random value from 2, 4 with equal probability
        # Get the i-layer value from an user defined list
        user_layer_values = [5, random_value, 3, random_value, 1]

        # Append the value used in each layer to a list
        materials_list = user_layer_values
        for i, lst in enumerate(all_layers):
            mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
            values[mask] = user_layer_values[i]

    # Create the cptlike data that accompanies the synthetic data if create_cptlike is True
    if create_cptlike:
        cpt_like_image = create_cptlike_array(image_matrix=values, x_max=x_max, z_max=z_max)

    # Store the results in a dataframe
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    # Save to HDF5
    h5_filename = f"typeF_{counter + 1}.h5"
    h5_path = os.path.join(output_folder, h5_filename)
    with h5py.File(h5_path, "w") as f:
        # Save the 2D array (image matrix) as a dataset
        # Make sure to save the matrix with the correct orientation
        f.create_dataset("ICvalues_matrix", data=values.reshape(x_max, z_max).T)  # Correctly reshape for z, x
        f.create_dataset("cptlike_matrix", data=cpt_like_image)  # Save the cptlike data

        # Save metadata as attributes
        f.attrs["model_type"] = "F"
        f.attrs["matrix_shape"] = values.reshape(x_max, z_max).T.shape
        # TODO: Add a description that makes sense for the model
        f.attrs["description"] = "Deltaic area with subhorizontal layers and the pleistocene sand as base layer at 30 m depth"
        f.attrs["date"] = str(datetime.datetime.now())
        f.attrs["seed"] = seed
        f.attrs["randomfield"] = RF
        f.attrs["materials"] = materials_list

    print(f"Data saved as {h5_filename}")

    # Optionally, save the image as a PNG file
    if save_image:
        plt.clf()  # Clear the current figure
        df_pivot = df.pivot(index="z", columns="x", values="IC")
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(df_pivot, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")

    # Optionally, save the CSV file
    if save_csv:
        csv_path = os.path.join(output_folder, f"{h5_filename.replace('.h5', '.csv')}")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved as {csv_path}")

    # Optionally, save the CPTlike image as a PNG file
    if save_cptlike_image:
        plt.clf()  # Clear the current figure
        fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(cpt_like_image, interpolation='none', aspect='auto')
        plt.axis("off")

        fig_path = os.path.join(output_folder, f"cptlike_{h5_filename.replace('.h5', '.png')}")
        plt.savefig(fig_path)
        plt.close()
        print(f"Image saved as {fig_path}")


