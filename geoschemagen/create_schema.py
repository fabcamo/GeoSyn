import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geoschemagen.create_rfs import generate_rf_group
from geoschemagen.create_layer_boundaries import layer_boundary, layer_boundary_horizA, layer_boundary_irregular
from geoschemagen.create_layer_boundaries import layer_boundary_subhorizB, layer_boundary_horizB, layer_boundary_lensC, layer_boundary_subhorizD_vert, layer_boundary_irregularE


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


def create_schema_typeA(output_folder: str, counter: int, z_max: int, x_max: int, trigo_type: int, seed: int, RF: bool = False):
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
            layer_coordinates = coords_to_list[mask]

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


def create_schema_typeB(output_folder: str, counter: int, z_max: int, x_max: int, trigo_type: int, seed: int = 20220412):
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

    # Apply the random field models to the layers
    all_layers = [area_1, area_2, area_3, area_4, area_5, area_6]

    # Choose random value from 2, 3, 4, 5 with equal probability
    random_value = np.random.choice([4, 1])
    # Get the i-layer value from an user defined list
    user_layer_values = [random_value, 1, random_value, 3, 2, 5]

    for i, lst in enumerate(all_layers):
        # Create a mask to select the grid cells for each layer
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]

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
    filename = f"typeB_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    #df.to_csv(csv_path)
    plt.close()


def create_schema_typeC(output_folder: str, counter: int, z_max: int, x_max: int, trigo_type: int, seed: int = 20220412):
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

    # Apply the random field models to the layers
    all_layers = [area_1, area_2, area_3, area_4]

    # Generate random non-repeating values for the layers using NumPy
    #user_layer_values = np.random.permutation(np.arange(1, 5))
    #random_value = np.random.choice([2, 4])
    user_layer_values = [2, 4, 1, 3]

    for i, lst in enumerate(all_layers):
        # Create a mask to select the grid cells for each layer
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]

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
    filename = f"typeC_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    #df.to_csv(csv_path)
    plt.close()


def create_schema_typeD(output_folder: str, counter: int, z_max: int, x_max: int, trigo_type: int, seed: int = 20220412):
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

    # Apply the random field models to the layers
    all_layers = [area_1, area_2, area_3, area_4, area_5, area_6, area_7]

    random_value = np.random.choice([5, 6])
    # Generate random non-repeating values for the layers using NumPy
    user_layer_values = [6, random_value, 5, 6, 5, random_value, 1]

    for i, lst in enumerate(all_layers):
        # Create a mask to select the grid cells for each layer
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]

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
    filename = f"typeC_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    #df.to_csv(csv_path)
    plt.close()



def create_schema_typeE(output_folder: str, counter: int, z_max: int, x_max: int, combine_trigo: bool, seed: int = 20220412):
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
    if combine_trigo == True:
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

    # Apply the random field models to the layers
    all_layers = [area_1, area_2, area_3, area_4, area_5]

    # Choose random value from 2, 3, 4, 5 with equal probability
    random_value = np.random.choice([2, 4])
    # Get the i-layer value from an user defined list
    user_layer_values = [5, random_value, 3, random_value, 1]

    for i, lst in enumerate(all_layers):
        # Create a mask to select the grid cells for each layer
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]

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
    filename = f"typeE_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    #df.to_csv(csv_path)
    plt.close()



def create_schema_typeF(output_folder: str, counter: int, z_max: int, x_max: int, seed: int = 20220412):
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

    # Apply the random field models to the layers
    all_layers = [area_1, area_2, area_3, area_4, area_5]

    # Choose random value from 2, 3, 4, 5 with equal probability
    random_value = np.random.choice([2, 4])
    # Get the i-layer value from an user defined list
    user_layer_values = [5, random_value, 3, random_value, 1]

    for i, lst in enumerate(all_layers):
        # Create a mask to select the grid cells for each layer
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]

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
    filename = f"typeF_{counter+1}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    #df.to_csv(csv_path)
    plt.close()