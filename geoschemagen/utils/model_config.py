import os
import json

# Bundled defaults with tunable boundary parameters for every model type (A-F, S)
DEFAULT_MODEL_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model_params.json")


def load_model_params(model_type: str, config_path: str = None) -> dict:
    """
    Load the tunable boundary parameters for a given model type from a JSON file.

    Args:
        model_type (str): The model type/letter to load parameters for (e.g. "A", "B", ..., "S").
        config_path (str): Path to the JSON config file. Defaults to the bundled model_params.json.
    Returns:
        dict: The parsed parameters for the requested model type. Empty dict if the type
            has no entry in the config file (callers fall back to hardcoded defaults in that case).
    """
    path = config_path or DEFAULT_MODEL_CONFIG
    with open(path, "r") as f:
        all_params = json.load(f)
    return all_params.get(model_type, {})


def resolve_bound(value, x_max: float, z_max: float):
    """
    Resolve a config value that may be a fixed number, or the special strings "x_max"/"z_max".

    Args:
        value: A number, "x_max", or "z_max".
        x_max (float): Length of the model.
        z_max (float): Depth of the model.
    Returns:
        float: The resolved numeric value.
    """
    if value == "x_max":
        return x_max
    if value == "z_max":
        return z_max
    return value


def resolve_dist_kwargs(cfg: dict, x_max: float, z_max: float) -> dict:
    """
    Resolve every "x_max"/"z_max" placeholder in a distribution config dict
    (e.g. {"low": 0, "high": "z_max"}) to actual numbers.

    Args:
        cfg (dict): Distribution config, e.g. {"low": ..., "peak": ..., "high": ...}.
        x_max (float): Length of the model.
        z_max (float): Depth of the model.
    Returns:
        dict: Same keys, with placeholder strings resolved to numbers.
    """
    return {k: resolve_bound(v, x_max, z_max) for k, v in cfg.items()}
