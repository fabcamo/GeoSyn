from random import betavariate

def pert(low, peak, high, *, lamb=10):
    """
    Generate a value using a Beta-Pert distribution.

    Args:
        low (float): Minimum value.
        peak (float): Most likely value.
        high (float): Maximum value.
        lamb (float, optional): Lambda parameter for the distribution. Defaults to 10.

    Returns:
        float: Generated value.
    """
    r = high - low

    # Calculate alpha and beta parameters for the Beta distribution
    alpha = 1 + lamb * (peak - low) / r
    beta = 1 + lamb * (high - peak) / r
    # Generate a value using the Beta distribution

    return low + betavariate(alpha, beta) * r