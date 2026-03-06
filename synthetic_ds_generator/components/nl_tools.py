import numpy as np


def wrap_func_based_on_x(f, limit_start=[-1.0, 1.0], hard_limit=1.1):
    """
    Returns a new function that wraps f and replaces any value outside of the limit with  a wrapper.
    We use a tan function to smoothly wrap the original function.
    The limit is smooth by wrapping with a tan function that begins from the limit start
        Parameters

    #TODO: Should consider the direction of the graph at the border to avoid hard cuts.
    -------
    f: function to wrap
    limit_start: [float, float]
        x range in which no filter is applied.
    hard_limit: float
        the maximum value that the function can take when the limit is applied
    -------
    """

    max_val = f(limit_start[1])
    min_val = f(limit_start[0])

    sizetop = (max_val * hard_limit) - max_val
    sizebottom = (min_val * hard_limit) - min_val

    def wrapped_func(
        x,
        f=f,
        limit_start=limit_start,
        sizetop=sizetop,
        sizebottom=sizebottom,
        max_val=max_val,
        min_val=min_val,
    ):
        default = f(x)
        if (x >= limit_start[0]) and (x <= limit_start[1]):
            return default
        elif x < limit_start[0]:
            return np.tanh(limit_start[0] - x) * sizebottom + min_val
        elif x > limit_start[1]:
            return np.tanh(x - limit_start[1]) * sizetop + max_val
        else:
            print(x, default, "ISSUE")

    return wrapped_func


def wrap_func_based_on_y(f, limit_start=[-1.0, 1.0], hard_limit=[-1.1, 1.1]):
    """
    Returns a new function that wraps f and prevent values that are over the hard limit.
    We use a tan function to smoothly wrap the original function.
        Parameters
    ----------
    f : function
        The function to wrap.
    limit_start : the value from which the wrapping is applied instead of the original function
    hard_limit : array_like
        The hard limits of the function.

    Returns
    -------
    function
        A new function that wraps f and prevents values that are over
    """

    sizetop = np.abs((hard_limit[1] - limit_start[1]))
    sizebottom = np.abs((hard_limit[0] - limit_start[0]))

    def wrapped_func(
        x, f=f, limit_start=limit_start, sizetop=sizetop, sizebottom=sizebottom
    ):
        default = f(x)
        if (default >= limit_start[0]) and (default <= limit_start[1]):
            return default
        elif default < limit_start[0]:
            return np.tanh(default - limit_start[0]) * sizebottom + limit_start[0]
        elif default > limit_start[1]:
            return np.tanh(default - limit_start[1]) * sizetop + limit_start[1]

    return wrapped_func
