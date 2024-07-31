import numpy as np
import gstools as gs
from utils.soil_types import soil_behaviour_clay, soil_behaviour_siltmix, soil_behaviour_sandmix, soil_behaviour_sand, soil_behaviour_organic

ndim = 2

def rf_generator(std_value: float, mean: float, aniso_x: float, aniso_z: float, angles: float, seed: int):
    """
    Generate random fields using the Gaussian model.

    Args:
        std_value (float): Standard deviation value.
        mean (float): Mean value.
        aniso_x (float): Anisotropy value in X direction.
        aniso_z (float): Anisotropy value in Z direction.
        angles (float): Angle of rotation.
        seed (int): Random seed.
    Returns:
        gs.SRF: Generated random field model.
    """
    # Set the length scale and variance
    len_scale = np.array([aniso_x, aniso_z])
    # Calculate the variance
    var = std_value**2

    # Generate the Gaussian model and the SRF
    model = gs.Gaussian(dim=ndim, var=var, len_scale=len_scale, angles=angles)
    srf = gs.SRF(model, mean=mean, seed=seed)

    return srf



def generate_rf_group(seed: int):
    """
    Generate random field models for different materials.

    Args:
        seed (int): Random seed.
    Returns:
        list: List of generated random field models.
    """
    # Generate random field models for clay
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_clay()
    srf_clay = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 1)
    # Generate random field models for silt mix
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_siltmix()
    srf_siltmix = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 2)
    # Generate random field models for sand mix
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_sandmix()
    srf_sandmix = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 3)
    # Generate random field models for sand
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_sand()
    srf_sand = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 4)
    # Generate random field models for organic
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_organic()
    srf_organic = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 5)
    # Generate random field models for clay
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_clay()
    srf_clay2 = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 6)
    # Generate random field models for sand
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_sand()
    srf_sand2 = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 7)

    # Store the random field models inside layers
    layers = [srf_clay, srf_siltmix, srf_sandmix, srf_sand, srf_organic, srf_clay2, srf_sand2]

    return layers
