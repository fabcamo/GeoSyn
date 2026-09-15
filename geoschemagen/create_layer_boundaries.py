import numpy as np
from geoschemagen.utils.distributions import pert
from geoschemagen.utils.model_config import resolve_bound, resolve_dist_kwargs

def layer_boundary(x_coord: np.array, z_max:float, params: dict = None):
    """
    Generate a sine or cosine line as a layer boundary.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.
        params (dict, optional): Overrides for amplitude/period/phase_shift/vertical_shift/functions,
            as loaded by geoschemagen.utils.model_config.load_model_params("S"). Defaults to None,
            which uses the original hardcoded bounds below.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)

    if params is None:
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
    else:
        # Resolve each bound, substituting the "x_max"/"z_max" placeholders for their actual values
        amplitude_kwargs = {k: resolve_bound(v, x_max, z_max) for k, v in params["amplitude"].items()}
        period_kwargs = {k: resolve_bound(v, x_max, z_max) for k, v in params["period"].items()}
        phase_kwargs = {k: resolve_bound(v, x_max, z_max) for k, v in params["phase_shift"].items()}
        vertical_kwargs = {k: resolve_bound(v, x_max, z_max) for k, v in params["vertical_shift"].items()}

        amplitude = pert(**amplitude_kwargs)
        period = pert(**period_kwargs)
        phase_shift = np.random.uniform(**phase_kwargs)
        vertical_shift = np.random.uniform(**vertical_kwargs)

        func_map = {"sin": np.sin, "cos": np.cos}
        func_choices = [func_map[name] for name in params.get("functions", ["sin", "cos"])]
        func = np.random.choice(func_choices)

    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y


def layer_boundary_irregular(x_coord: np.array, z_max:float, params: dict = None):
    """
    Generate a sine or cosine line as a layer boundary.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.
        params (dict, optional): Overrides for amplitude/period/phase_shift/vertical_shift,
            as loaded by geoschemagen.utils.model_config.load_model_params("F")["irregular"].
            Defaults to None, which uses the original hardcoded bounds below.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)
    params = params or {}

    amplitude_kwargs = resolve_dist_kwargs(params.get("amplitude", {"low": 5, "peak": 20, "high": 45}), x_max, z_max)
    period_kwargs = resolve_dist_kwargs(params.get("period", {"low": 250, "peak": 1000, "high": 2000}), x_max, z_max)
    phase_kwargs = resolve_dist_kwargs(params.get("phase_shift", {"low": 100, "high": 1000}), x_max, z_max)
    vertical_kwargs = resolve_dist_kwargs(params.get("vertical_shift", {"low": 0, "high": z_max}), x_max, z_max)

    # Generate amplitude using the triangular distribution with specified range
    amplitude = np.random.triangular(amplitude_kwargs["low"], amplitude_kwargs["peak"], amplitude_kwargs["high"])
    # Generate period using the triangular distribution with specified range
    period = np.random.triangular(period_kwargs["low"], period_kwargs["peak"], period_kwargs["high"])
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=phase_kwargs["low"], high=phase_kwargs["high"])
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=vertical_kwargs["low"], high=vertical_kwargs["high"])
    # Choose between sine and cosine wave functions
    func = np.random.choice([np.sin, np.cos])
    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y


def layer_boundary_horizA(x_coord: np.array, z_max:float, trigo_type: int, params: dict = None):
    """
    Generate a sine or cosine line as a layer boundary. This means parameters focus on a horizontal
    layer boundary. This means very high period and low amplitude.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.
        params (dict, optional): Overrides for amplitude/period/phase_shift/vertical_shift,
            as loaded by geoschemagen.utils.model_config.load_model_params("A")["horizA"].
            Defaults to None, which uses the original hardcoded bounds below.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)
    params = params or {}

    amplitude_kwargs = resolve_dist_kwargs(params.get("amplitude", {"low": 2, "peak": 3, "high": 5}), x_max, z_max)
    period_kwargs = resolve_dist_kwargs(params.get("period", {"low": 1000, "peak": 2000, "high": 3000}), x_max, z_max)
    phase_kwargs = resolve_dist_kwargs(params.get("phase_shift", {"low": 0, "high": 0}), x_max, z_max)
    vertical_kwargs = resolve_dist_kwargs(params.get("vertical_shift", {"low": 3, "high": 25}), x_max, z_max)

    # Generate amplitude using the triangular distribution with specified range
    amplitude = np.random.triangular(amplitude_kwargs["low"], amplitude_kwargs["peak"], amplitude_kwargs["high"])
    # Generate period using the triangular distribution with specified range
    period = np.random.triangular(period_kwargs["low"], period_kwargs["peak"], period_kwargs["high"])
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=phase_kwargs["low"], high=phase_kwargs["high"])
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=vertical_kwargs["low"], high=vertical_kwargs["high"])

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


def layer_boundary_subhorizB(x_coord: np.array, z_max:float, trigo_type: int, params: dict = None):
    """
    Generate a sine or cosine line as a layer boundary. This means parameters focus on a subhorizontal
    layer boundary. This means very high period and low amplitude.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.
        params (dict, optional): Overrides for amplitude/period/phase_shift/vertical_shift,
            as loaded by geoschemagen.utils.model_config.load_model_params("C")["subhorizB"].
            Defaults to None, which uses the original hardcoded bounds below.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)
    params = params or {}

    amplitude_kwargs = resolve_dist_kwargs(params.get("amplitude", {"low": 2, "peak": 5, "high": 10}), x_max, z_max)
    period_kwargs = resolve_dist_kwargs(params.get("period", {"low": 1000, "peak": 2000, "high": 3000}), x_max, z_max)
    phase_kwargs = resolve_dist_kwargs(params.get("phase_shift", {"low": 0, "high": 50}), x_max, z_max)
    vertical_kwargs = resolve_dist_kwargs(params.get("vertical_shift", {"low": 0, "high": 20}), x_max, z_max)

    # Generate amplitude using the triangular distribution with specified range
    amplitude = np.random.triangular(amplitude_kwargs["low"], amplitude_kwargs["peak"], amplitude_kwargs["high"])
    # Generate period using the triangular distribution with specified range
    period = np.random.triangular(period_kwargs["low"], period_kwargs["peak"], period_kwargs["high"])
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=phase_kwargs["low"], high=phase_kwargs["high"])
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=vertical_kwargs["low"], high=vertical_kwargs["high"])

    if trigo_type == 1: # Use sin
        func = np.sin
    elif trigo_type == 2: # Use cos
        func = np.cos
    else: # Use sin or cosine at random
        func = np.random.choice([np.sin, np.cos])

    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y


def layer_boundary_lensC(x_coord: np.array, z_max:float, trigo_type: int, params: dict = None):
    """
    Generate a sine or cosine line as a layer boundary. This means parameters focus on a sinusoidal
    layer boundary. This means high amplitude and moderate period.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.
        params (dict, optional): Overrides for amplitude/period/phase_shift/vertical_shift,
            as loaded by geoschemagen.utils.model_config.load_model_params("C")["lensC"].
            Defaults to None, which uses the original hardcoded bounds below.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """
    # Get the length of the x coordinates
    x_max = len(x_coord)
    params = params or {}

    amplitude_kwargs = resolve_dist_kwargs(params.get("amplitude", {"low": 20, "peak": 30, "high": 50}), x_max, z_max)
    period_kwargs = resolve_dist_kwargs(params.get("period", {"low": 200, "peak": 300, "high": 800}), x_max, z_max)
    phase_kwargs = resolve_dist_kwargs(params.get("phase_shift", {"low": 0, "high": 250}), x_max, z_max)
    vertical_kwargs = resolve_dist_kwargs(params.get("vertical_shift", {"low": 10, "high": z_max}), x_max, z_max)

    # Generate amplitude using the triangular distribution with specified range
    amplitude = np.random.triangular(amplitude_kwargs["low"], amplitude_kwargs["peak"], amplitude_kwargs["high"])
    # Generate period using the triangular distribution with specified range
    period = np.random.triangular(period_kwargs["low"], period_kwargs["peak"], period_kwargs["high"])
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=phase_kwargs["low"], high=phase_kwargs["high"])
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=vertical_kwargs["low"], high=vertical_kwargs["high"])

    if trigo_type == 1: # Use sin
        func = np.sin
    elif trigo_type == 2: # Use cos
        func = np.cos
    else: # Use sin or cosine at random
        func = np.random.choice([np.sin, np.cos])

    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y


def layer_boundary_subhorizD_vert(x_coord: np.array, z_max:float, trigo_type: int, vert_low: float, vert_high: float, params: dict = None):
    """
    Generate a sine or cosine line as a layer boundary. This means parameters focus on a subhorizontal
    layer boundary. This means very high period and low amplitude. User can fix the vertical shift.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.
        trigo_type (int): Type of trigonometric function to use.
        vert_low (float): Lowest vertical shift value.
        vert_high (float): Highest vertical shift value.
        params (dict, optional): Overrides for amplitude/period/phase_shift, as loaded by
            geoschemagen.utils.model_config.load_model_params("B"/"D")["subhorizD_vert"]. The
            vertical shift is controlled by vert_low/vert_high (see the model's "bands" config
            entry), not by this dict. Defaults to None, which uses the original hardcoded bounds.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)
    params = params or {}

    amplitude_kwargs = resolve_dist_kwargs(params.get("amplitude", {"low": 2, "peak": 5, "high": 8}), x_max, z_max)
    period_kwargs = resolve_dist_kwargs(params.get("period", {"low": 1000, "peak": 3000, "high": 4000}), x_max, z_max)
    phase_kwargs = resolve_dist_kwargs(params.get("phase_shift", {"low": 0, "high": 0}), x_max, z_max)

    # Generate amplitude using the triangular distribution with specified range
    amplitude = np.random.triangular(amplitude_kwargs["low"], amplitude_kwargs["peak"], amplitude_kwargs["high"])
    # Generate period using the triangular distribution with specified range
    period = np.random.triangular(period_kwargs["low"], period_kwargs["peak"], period_kwargs["high"])
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=phase_kwargs["low"], high=phase_kwargs["high"])
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


def layer_boundary_irregularE(x_coord: np.array, z_max:float, trigo_type: int, params: dict = None):
    """
    Generate a sine or cosine line as a layer boundary. This means parameters focus on a horizontal
    layer boundary. This means very high period and low amplitude.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.
        params (dict, optional): Overrides for period/trigo_1/trigo_2, as loaded by
            geoschemagen.utils.model_config.load_model_params("E"). Defaults to None, which
            uses the original hardcoded bounds below.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    # Get the length of the x coordinates
    x_max = len(x_coord)
    params = params or {}

    #trigo_type = 1

    if trigo_type == 1: # Use sin
        func = np.sin
        trigo_cfg = params.get("trigo_1", {})
        amplitude_kwargs = resolve_dist_kwargs(trigo_cfg.get("amplitude", {"low": 35, "peak": 70, "high": 100}), x_max, z_max)
        phase_kwargs = resolve_dist_kwargs(trigo_cfg.get("phase_shift", {"low": -1500, "high": -1000}), x_max, z_max)
        vertical_kwargs = resolve_dist_kwargs(trigo_cfg.get("vertical_shift", {"low": -28, "high": 0}), x_max, z_max)
        amplitude = np.random.triangular(amplitude_kwargs["low"], amplitude_kwargs["peak"], amplitude_kwargs["high"])
        phase_shift = np.random.uniform(low=phase_kwargs["low"], high=phase_kwargs["high"])
        vertical_shift = np.random.uniform(low=vertical_kwargs["low"], high=vertical_kwargs["high"])
    elif trigo_type == 2: # Use cos
        func = np.cos
        trigo_cfg = params.get("trigo_2", {})
        amplitude_kwargs = resolve_dist_kwargs(trigo_cfg.get("amplitude", {"low": 20, "peak": 40, "high": 50}), x_max, z_max)
        phase_kwargs = resolve_dist_kwargs(trigo_cfg.get("phase_shift", {"low": 1000, "high": 1500}), x_max, z_max)
        vertical_kwargs = resolve_dist_kwargs(trigo_cfg.get("vertical_shift", {"low": 0, "high": 28}), x_max, z_max)
        amplitude = np.random.triangular(amplitude_kwargs["low"], amplitude_kwargs["peak"], amplitude_kwargs["high"])
        phase_shift = np.random.uniform(low=phase_kwargs["low"], high=phase_kwargs["high"])
        vertical_shift = np.random.uniform(low=vertical_kwargs["low"], high=vertical_kwargs["high"])
    else: # Use sin or cosine at random
        func = np.random.choice([np.sin, np.cos])

    period_kwargs = resolve_dist_kwargs(params.get("period", {"low": 2000, "peak": 3500, "high": 5000}), x_max, z_max)
    # Generate period using the triangular distribution with specified range
    period = np.random.triangular(period_kwargs["low"], period_kwargs["peak"], period_kwargs["high"])

    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y

