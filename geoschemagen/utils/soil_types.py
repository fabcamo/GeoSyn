import numpy as np

def soil_behaviour_clay():
    """
    Return RF-IC values for clay.

    Args:
        None
    Return:
        Tuple: Tuple containing std_value, mean, aniso_x, aniso_z, and angles.
    """
    # Set the minimum and maximum values for the RF-IC
    min_IC = 2.95
    max_IC = 3.6
    # Calculate the mean value
    mean = (min_IC + max_IC) / 2
    # Calculate the standard deviation value
    std_value = (max_IC - min_IC) / 6
    # Generate anisotropy in X
    aniso_x = np.random.triangular(3, 47, 80)
    # Generate anisotropy in Z
    aniso_z = np.random.uniform(0.2, 3)  # anisotropy in Z
    # Generate the angle of rotation
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor

    return std_value, mean, aniso_x, aniso_z, angles



def soil_behaviour_siltmix():
    """
    Return RF-IC values for clayey silt to silty clay.

    Args:
        None
    Return:
        Tuple: Tuple containing std_value, mean, aniso_x, aniso_z, and angles.
    """
    # Set the minimum and maximum values for the RF-IC
    min_IC = 2.6
    max_IC = 2.95
    # Calculate the mean value
    mean = (min_IC + max_IC) / 2
    # Calculate the standard deviation value
    std_value = (max_IC - min_IC) / 6
    # Generate anisotropy in X
    aniso_x = np.random.triangular(3, 47, 80)
    # Generate anisotropy in Z
    aniso_z = np.random.uniform(0.2, 3)
    # Generate the angle of rotation
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor

    return std_value, mean, aniso_x, aniso_z, angles



def soil_behaviour_sandmix():
    """
    Return RF-IC values for silty sand to sandy silt.

    Args:
        None
    Return:
        Tuple: Tuple containing std_value, mean, aniso_x, aniso_z, and angles.
    """
    # Set the minimum and maximum values for the RF-IC
    min_IC = 2.05
    max_IC = 2.6
    # Calculate the mean value
    mean = (min_IC + max_IC) / 2
    # Calculate the standard deviation value
    std_value = (max_IC - min_IC) / 6
    # Generate anisotropy in X
    aniso_x = np.random.triangular(3, 47, 80)
    # Generate anisotropy in Z
    aniso_z = np.random.uniform(0.2, 3)
    # Generate the angle of rotation
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor

    return std_value, mean, aniso_x, aniso_z, angles



def soil_behaviour_sand():
    """
    Return RF-IC values for sand.

    Args:
        None
    Return:
        Tuple: Tuple containing std_value, mean, aniso_x, aniso_z, and angles.
    """
    # Set the minimum and maximum values for the RF-IC
    min_IC = 1.31
    max_IC = 2.05
    # Calculate the mean value
    mean = (min_IC + max_IC) / 2
    # Calculate the standard deviation value
    std_value = (max_IC - min_IC) / 6
    # Generate anisotropy in X
    aniso_x = np.random.triangular(3, 47, 80)
    # Generate anisotropy in Z
    aniso_z = np.random.uniform(0.2, 3)
    # Generate the angle of rotation
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor

    return std_value, mean, aniso_x, aniso_z, angles



def soil_behaviour_organic():
    """
    Return RF-IC values for organic soils.

    Args:
        None
    Return:
        Tuple: Tuple containing std_value, mean, aniso_x, aniso_z, and angles.
    """
    # Set the minimum and maximum values for the RF-IC
    min_IC = 3.6
    max_IC = 4.2
    # Calculate the mean value
    mean = (min_IC + max_IC) / 2
    # Calculate the standard deviation value
    std_value = (max_IC - min_IC) / 6
    # Generate anisotropy in X
    aniso_x = np.random.triangular(3, 47, 80)
    # Generate anisotropy in Z
    aniso_z = np.random.uniform(0.2, 3)
    # Generate the angle of rotation
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor

    return std_value, mean, aniso_x, aniso_z, angles

