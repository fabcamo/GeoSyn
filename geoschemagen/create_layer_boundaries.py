import numpy as np
from utils.distributions import pert

def layer_boundary(x_coord: np.array, z_max:float):
    """
    Generate a sine or cosine line as a layer boundary.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)

    # Generate amplitude using the pert function with specified range
    amplitude = pert(2, 5, z_max)
    # Generate period using the pert function with specified range
    period = pert(x_max, 1000, 10000)
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=0, high=x_max)
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=0, high=z_max)
    # Choose between sine and cosine wave functions
    func = np.random.choice([np.sin, np.cos])
    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y


def layer_boundary_irregular(x_coord: np.array, z_max:float):
    """
    Generate a sine or cosine line as a layer boundary.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)

    # Generate amplitude using the pert function with specified range
    amplitude = np.random.triangular(20, 30, 50)
    # Generate period using the pert function with specified range
    period = np.random.triangular(250, 500, 700)
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=100, high=1000)
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=0, high=z_max)
    # Choose between sine and cosine wave functions
    func = np.random.choice([np.sin, np.sin])
    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y



def layer_boundary_horizA(x_coord: np.array, z_max:float, trigo_type: int):
    """
    Generate a sine or cosine line as a layer boundary. This means parameters focus on a horizontal
    layer boundary. This means very high period and low amplitude.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)

    # Generate amplitude using the pert function with specified range
    amplitude = np.random.triangular(2, 3, 5)
    # Generate period using the pert function with specified range
    period = np.random.triangular(1000, 2000, 3000)
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=0, high=0)
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=3, high=25)

    if trigo_type == 1: # Use sin
        func = np.sin
    elif trigo_type == 2: # Use cos
        func = np.cos
    else: # Use sin or cosine at random
        func = np.random.choice([np.sin, np.cos])

    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y


def layer_boundary_horizB(x_coord: np.array, z_max:float, trigo_type: int):
    """
    Generate a sine or cosine line as a layer boundary. This means parameters focus on a horizontal
    layer boundary. This means very high period and low amplitude.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)

    # Generate amplitude using the pert function with specified range
    amplitude = np.random.triangular(2, 3, 5)
    # Generate period using the pert function with specified range
    period = np.random.triangular(1000, 2000, 3000)
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=0, high=0)
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=5, high=26)

    if trigo_type == 1: # Use sin
        func = np.sin
    elif trigo_type == 2: # Use cos
        func = np.cos
    else: # Use sin or cosine at random
        func = np.random.choice([np.sin, np.cos])

    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y


def layer_boundary_subhorizB(x_coord: np.array, z_max:float, trigo_type: int):
    """
    Generate a sine or cosine line as a layer boundary. This means parameters focus on a subhorizontal
    layer boundary. This means very high period and low amplitude.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)

    # Generate amplitude using the pert function with specified range
    amplitude = np.random.triangular(2, 5, 10)
    # Generate period using the pert function with specified range
    period = np.random.triangular(1000, 2000, 3000)
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=0, high=50)
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=0, high=20)

    if trigo_type == 1: # Use sin
        func = np.sin
    elif trigo_type == 2: # Use cos
        func = np.cos
    else: # Use sin or cosine at random
        func = np.random.choice([np.sin, np.cos])

    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y


def layer_boundary_lensC(x_coord: np.array, z_max:float, trigo_type: int):
    """
    Generate a sine or cosine line as a layer boundary. This means parameters focus on a sinusoidal
    layer boundary. This means high amplitude and moderate period.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """
    # Get the length of the x coordinates
    x_max = len(x_coord)

    # Generate amplitude using the pert function with specified range
    amplitude = np.random.triangular(20, 30, 50)
    # Generate period using the pert function with specified range
    period = np.random.triangular(200, 300, 800)
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=0, high=250)
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=10, high=z_max)

    if trigo_type == 1: # Use sin
        func = np.sin
    elif trigo_type == 2: # Use cos
        func = np.cos
    else: # Use sin or cosine at random
        func = np.random.choice([np.sin, np.cos])

    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y


def layer_boundary_subhorizD_vert(x_coord: np.array, z_max:float, trigo_type: int, vert_low: float, vert_high: float):
    """
    Generate a sine or cosine line as a layer boundary. This means parameters focus on a subhorizontal
    layer boundary. This means very high period and low amplitude. User can fix the vertical shift.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.
        trigo_type (int): Type of trigonometric function to use.
        vert_low (float): Lowest vertical shift value.
        vert_high (float): Highest vertical shift value.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)

    # Generate amplitude using the pert function with specified range
    amplitude = np.random.triangular(2, 5, 8)
    # Generate period using the pert function with specified range
    period = np.random.triangular(1000, 3000, 4000)
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=0, high=0)
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=vert_low, high=vert_high)

    if trigo_type == 1: # Use sin
        func = np.sin
    elif trigo_type == 2: # Use cos
        func = np.cos
    else: # Use sin or cosine at random
        func = np.random.choice([np.sin, np.cos])

    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y