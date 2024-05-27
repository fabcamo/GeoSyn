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
