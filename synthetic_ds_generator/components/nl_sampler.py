import numpy as np
import functools
import copy
import math
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy import interpolate
from scipy.interpolate import make_interp_spline


class NL_function_generator:
    """
    NL_function_generator can be used to sample nonlinear functions of various classes that we use in the structural lagged causal process,
    instantanous effects and exogenous parameters. Notably
    It supports multiple modes for generating nonlinear relationships, including static sets, power functions, splines, radial basis functions (RBF), and symbolic compositions.
    This class is designed to flexibly sample and assign nonlinear transformations to links in a causal structure, supporting both random and deterministic assignment,
    and provides mechanisms to control the complexity and range of the generated functions.

        rng (np.random.Generator): Random number generator for reproducibility.
        nl_mode (str): Mode of nonlinear function generation. Options include 'power_set', 'splines', 'rbf', 'symbolic', and 'dummy_linear'.
        limit_startx (list): Range for input values where no output limiting is applied.
        hard_limitx (float): Hard limit for output values beyond the input range.
        limit_starty (list): Range for output values where no output limiting is applied.
        hard_limity (list): Hard limits for output values.
        power_dist (list): Parameters for sampling exponents in power functions.
        which_power_dist (str): Specifies the distribution type for power functions.
        rbf_length_scale (float): Length scale parameter for RBF kernel.
        spline_samples (int): Number of spline control points.
        n_nl_stacking (int): Number of nonlinear functions to stack in symbolic mode.
        n_nl_operators (int): Number of operators to combine nonlinear functions in symbolic mode.
        n_options (int): Number of nonlinear function options to sample from.
        nl_options (dict): Dictionary of available nonlinear functions.
        operator_options (dict): Dictionary of available binary operators for symbolic mode.

        __init__(...): Initializes the NL_function_generator with specified parameters.
        nl_opts(): Returns a dictionary of available nonlinear function options.
        operator_options(): Returns a dictionary of available binary operator functions.
        get_element(t): Retrieves a nonlinear function by index.
        sample_nl_relationships(links, nl_proba, nl_mask): Samples and assigns nonlinear functions to links based on probability and mask.
        dummy_functions(nl_choice): Assigns simple linear functions for debugging.
        spline_functions(nl_choice): Assigns random spline-based nonlinear functions.
        symbolic_functions(nl_choice): Assigns complex symbolic compositions of nonlinear functions and operators.
        rbf_functions(nl_choice): Assigns functions sampled from a Gaussian Process with RBF kernel.
        power_functions(nl_choice): Assigns power-based nonlinear functions with sampled exponents.
        wrap_func_based_on_x(f): Wraps a function to limit its output smoothly outside a specified input range.
        wrap_func_based_on_y(f): Wraps a function to limit its output smoothly outside a specified output range.
    """

    def __init__(
        self,
        # Linear params
        rng: int = 42,
        nl_mode: str = "power_set",
        # Function limiters:
        limit_startx: list = [-1.0, 1.0],
        hard_limitx: float = 1.1,
        limit_starty: list = [-1.0, 1.0],
        hard_limity: list = [-1.1, 1.1],
        # power dist params
        power_dist: list = [0.5, 0.6, 1.9, 2],
        which_power_dist: str = "all",
        # RBF
        rbf_length_scale: float = 1,
        # Splines
        spline_samples: int = 4,
        # symbolic dist params
        n_nl_stacking: int = 2,
        n_nl_operators: int = 2,
        n_options: int = 10,
    ) -> None:
        
        self.rng = np.random.default_rng(rng)
        self.rbf_rng = rng
        self.n_nl_operators = n_nl_operators
        self.nl_mode = nl_mode
        self.n_nl_stacking = n_nl_stacking
        # power dist_params
        self.power_dist = power_dist
        self.which_power_dist = which_power_dist
        self.limit_startx = limit_startx
        self.hard_limitx = hard_limitx
        self.limit_starty = limit_starty
        self.hard_limity = hard_limity
        self.spline_samples = spline_samples

        self.vf = np.vectorize(self.get_element)

        self.nl_options = self.nl_opts()
        self.operator_options = self.operator_options()
        self.n_options = n_options  # SET properly
        self.n_operators = 2

        if nl_mode == "rbf":
            kernel = RBF(length_scale=rbf_length_scale)
            self.gpr = GaussianProcessRegressor(kernel=kernel, random_state=rng)

    def nl_opts(self):
        mono = {
            # monotonic
            0: np.cbrt,
            1: np.tanh,
            2: np.arcsinh,
            3: lambda a: np.max(a, 0),
            4: lambda a: a,
            # unimodal
            5: np.square,
            6: np.abs,
            7: np.cosh,
            8: lambda a: np.power(a, 2),
            # sinusoidal
            9: np.sin,
            10: np.cos,
            11: np.tan,
            12: np.sinc,
            # exp
            13: np.exp,
            14: np.sinh,
            15: lambda a: np.power(a, 3),
        }
        return mono

    def operator_options(self):
        operator_options = {
            0: lambda a, b: a + b,
            1: lambda a, b: a - b,
            # 2:lambda a,b: a*b, # Too many issues needs further control
            # 3:lambda a,b: a/b,
        }
        return operator_options

    def get_element(self, t):
        return self.nl_options[t]

    def sample_nl_relationships(self, links, nl_proba, nl_mask=None):
        # whether to put a function on the index
        nl_choice = np.zeros(links.shape)
        nl_choice[self.rng.random(links.shape) < nl_proba] = 1
        # Use the mask to fix cert<ain garantueed links.
        if isinstance(nl_mask, np.ndarray):
            if nl_mask.dtype == np.bool_:
                nl_choice[nl_mask] = 1
            else:
                assert False, "NOT IMPLEMENTED"
        nl_choice = nl_choice.astype(bool)
        if self.nl_mode == "dummy_linear":
            return self.dummy_functions(nl_choice)
        elif self.nl_mode == "power_set":
            return self.power_functions(nl_choice)
        elif self.nl_mode == "splines":
            return self.spline_functions(nl_choice)
        elif self.nl_mode == "rbf":
            return self.rbf_functions(nl_choice)
        elif self.nl_mode == "symbolic":
            return self.symbolic_functions(nl_choice)
        else:
            raise ValueError(
                "nl_mode must be one of 'dummy_linear', 'exp_set','splines', 'rbf','symbolic'"
            )

    def dummy_functions(self, nl_choice):
        """
        returns linear functions for debugging.
        """

        nl_stack = np.zeros(nl_choice.shape).astype(object)
        nl_naming = np.zeros(nl_choice.shape).astype(str)
        nl_naming[:, :, :] = "Nothing"

        nl_stack[nl_choice] = lambda a: a * 0.5
        nl_stack[~nl_choice] = lambda a: a

        nl_naming[nl_choice] = "linear_dummy_(*0.5)"
        return nl_stack, nl_naming

    def spline_functions(self, nl_choice):
        nl_stack = np.zeros(nl_choice.shape).astype(object)
        nl_naming = np.zeros(nl_choice.shape).astype(str)
        nl_naming[:, :, :] = "Nothing"
        nl_naming[nl_choice] = "Spline random"
        x_range = (-1, 1)
        x_nodes = np.linspace(x_range[0], x_range[1], self.spline_samples)

        for v in range(nl_choice.shape[0]):
            for w in range(nl_choice.shape[1]):
                for z in range(nl_choice.shape[2]):
                    if nl_choice[v, w, z]:
                        y_nodes = np.zeros_like(x_nodes)
                        y_nodes[0] = self.rng.uniform(0, 1)
                        for i in range(1, len(y_nodes)):
                            y_nodes[i] = y_nodes[i - 1] + self.rng.uniform(0, 10)
                        y_nodes = (
                            2
                            * (y_nodes - np.min(y_nodes))
                            / (np.max(y_nodes) - np.min(y_nodes))
                            - 1
                        )
                        nl_stack[v, w, z] = self.wrap_func_based_on_x(
                            make_interp_spline(x_nodes, y_nodes, k=3)
                        )
                    else:
                        nl_stack[v, w, z] = lambda a: a
        return nl_stack, nl_naming

    def symbolic_functions(self, nl_choice):
        """
        Samples nonlinear relationships from the specified options.

        This method generates a tensor with shape `links` that specifies the nonlinear function
        to apply at each position (lambda function). The nonlinear function is chosen from the options specified
        in `self.nl_options`.

        The method first generates a mask `nl_choice` that indicates which positions should
        have a nonlinear function applied. The mask is generated based on the probability
        `nl_proba` and the mask `nl_mask`.

        Then, the method generates a tensor `nl_stack` that specifies the nonlinear function
        to apply at each position. The tensor is generated by iterating over the last dimension
        of the `selection` tensor, which contains the indices of the nonlinear functions to be
        applied. For each position, the method combines the corresponding nonlinear functions
        using the `np.mean` function.

        Finally, the method returns the `nl_stack` tensor and the `selection` tensor, which
        contains the indices of the nonlinear functions that were actually applied.

        Args:
            links (np.ndarray): Matrix of link coefficients.
            nl_proba (float): Probability of a nonlinear relationship being included.
            nl_mask (np.ndarray): Mask to specify which nonlinear relationships to include.

        Returns:
            tuple: Tuple containing the sampled nonlinear relationships and the corresponding
                indices.
        """

        nl_stack = np.zeros(nl_choice.shape).astype(object)
        nl_naming = np.zeros(nl_choice.shape).astype(str)
        nl_naming[:, :, :] = "Nothing"

        selection = self.rng.integers(
            self.n_options,
            size=list(nl_choice.shape) + [self.n_nl_operators, self.n_nl_stacking],
        )

        operator_selection = self.rng.integers(
            self.n_operators, size=list(nl_choice.shape) + [self.n_nl_operators - 1]
        )

        nl_stack = np.zeros(nl_choice.shape).astype(object)

        for v in range(selection.shape[0]):
            for w in range(selection.shape[1]):
                for z in range(selection.shape[2]):
                    if nl_choice[v, w, z]:
                        components = []
                        ops = [
                            self.operator_options[f]
                            for f in operator_selection[v, w, z]
                        ]
                        for n in range(selection.shape[3]):
                            components.append(
                                [self.nl_options[f] for f in selection[v, w, z, n]]
                            )

                        def combine_functional_elements(
                            a, comps=components, operators=ops
                        ):
                            out_stack = []
                            for com in comps:
                                res = a
                                for func in com:
                                    res = func(res)
                                out_stack.append(res)

                            out = out_stack[0]
                            for n, x in enumerate(operators):
                                out = x(out, out_stack[n + 1])
                            return out

                        nl_stack[v, w, z] = self.wrap_func_based_on_y(
                            combine_functional_elements
                        )
                    else:
                        nl_stack[v, w, z] = lambda a: a

        return nl_stack, nl_naming

    def rbf_functions(self, nl_choice):
        """
        Compute Radial Basis Function (RBF) functions for each non-linearity choice.

        Parameters
        ----------
        nl_choice (numpy array): A 3D array of boolean values indicating whether this elements needs a sampled function.
        If not, lamda a:a is returned.

        Returns
        -------
        nl_stack : array-like
            A 3D array of functions, where each element is a function that can be called.
            If `nl_choice` is `True` at a particular position, the corresponding element in `nl_stack` is a
            Sampled function from an RBF kernel else its lamdbda a: a
        nl_naming : array-like
            A 3D array of string labels, where each element shows which indices were drawn from an RBF kernel

        Notes
        -----
        The RBF functions are computed by sampling the Gaussian Process Regression (GPR) model at a grid of
        points and then interpolating the results using Radial Basis Functions.
        With an increase in sample points, we can reduce the complexity of the function via rbf_length_scale at initialization
        """

        # TODO make hps flexible if necessary.
        nl_stack = np.zeros(nl_choice.shape).astype(object)
        nl_naming = np.zeros(nl_choice.shape).astype(str)
        nl_naming[:, :, :] = "Nothing"
        nl_naming[nl_choice] = "RBF random"
        space = np.arange(-20.1, 20.1, 0.1)
        randomness=self.rng.integers(0,100000) # to draw different samples every time the fuction is called.
        y_samples = self.gpr.sample_y(
            np.expand_dims(space, 1), len(nl_stack.flatten()),random_state=randomness
        )
        counter = 0
        for v in range(nl_choice.shape[0]):
            for w in range(nl_choice.shape[1]):
                for z in range(nl_choice.shape[2]):
                    if nl_choice[v, w, z]:
                        nl_stack[v, w, z] = self.wrap_func_based_on_x(
                            interpolate.interp1d(space, y_samples[:, counter])
                        )
                    else:
                        nl_stack[v, w, z] = lambda a: a
                    counter += 1
        return nl_stack, nl_naming

    def power_functions(self, nl_choice):

        """
        Generate power functions for each element in the nl_choice array.

        Parameters:
        nl_choice (numpy array): A 3D array of boolean values indicating whether this elements needs a sampled function.
        If not, lamda a:a is returned.

        # Distributions:


        # Importantly: By selecting the distribution for the power, we can scale the nonlinearity of the function.

        Returns:
        nl_stack (numpy array): A 3D array of functions, where each function corresponds to a specific function with a random exponent.
        nl_naming (numpy array): A 3D array of strings, where each string corresponds to the exponent value used for a power function.

        Notes:
        The `which_power_dist` attribute determines whether to use the "saddle", "no_saddle", "no_saddle_reversed" or "all" exponential based distribution.
        """

        nl_stack = np.zeros(nl_choice.shape).astype(object)
        nl_naming = np.zeros(nl_choice.shape).astype(str)
        nl_naming[:, :, :] = "Nothing"

        # Randomly select from nl_power1 and nl_power2 for each element
        nl_power1 = self.rng.uniform(
            self.power_dist[0], self.power_dist[1], size=nl_choice.shape
        )
        nl_power2 = self.rng.uniform(
            self.power_dist[2], self.power_dist[3], size=nl_choice.shape
        )
        choose_first = self.rng.integers(0, 2, size=nl_choice.shape).astype(bool)
        nl_power = np.where(choose_first, nl_power1, nl_power2)
        # name them
        nl_naming[nl_choice] = nl_power[nl_choice].astype(str)

        # iterate through all indices and asigns a specific function
        for v in range(nl_choice.shape[0]):
            for w in range(nl_choice.shape[1]):
                for z in range(nl_choice.shape[2]):
                    # We use three simple distributions that are described above.
                    def power_dist(x, exp=nl_power[v, w, z]):
                        if x >= 0:
                            return np.power(np.abs(x), exp)
                        else:
                            return -np.power(np.abs(x), exp)

                    def power_dist2(x, exp=nl_power[v, w, z]):
                        scale = np.abs((x + 1) * 0.5)
                        return np.power(scale, exp) / 0.5 - 1

                    def power_dist3(x, exp=nl_power[v, w, z]):
                        scale = np.abs((x - 1) / 2)
                        return -np.power(np.abs(scale), exp) / 0.5 + 1

                    # Assign draw the specific function and assign to index.
                    if nl_choice[v, w, z]:
                        if self.which_power_dist == "saddle":
                            nl_stack[v, w, z] = self.wrap_func_based_on_y(power_dist)
                        elif self.which_power_dist == "no_saddle":
                            draw = self.rng.integers(0, 2)
                            if draw == 0:
                                nl_stack[v, w, z] = self.wrap_func_based_on_y(
                                    power_dist2
                                )
                            elif draw == 1:
                                nl_stack[v, w, z] = self.wrap_func_based_on_x(
                                    power_dist3
                                )
                            nl_stack[v, w, z] = self.wrap_func_based_on_x(power_dist2)
                        elif self.which_power_dist == "no_saddle_reversed":
                            nl_stack[v, w, z] = self.wrap_func_based_on_x(power_dist3)
                        elif self.which_power_dist == "all":
                            draw = self.rng.integers(0, 3)
                            if draw == 0:
                                nl_stack[v, w, z] = self.wrap_func_based_on_y(
                                    power_dist
                                )
                            elif draw == 1:
                                nl_stack[v, w, z] = self.wrap_func_based_on_x(
                                    power_dist2
                                )
                            elif draw == 2:
                                nl_stack[v, w, z] = self.wrap_func_based_on_x(
                                    power_dist3
                                )
                        else:
                            raise ValueError("unknown power_dist")
                    else:
                        nl_stack[v, w, z] = lambda a: a
        return nl_stack, nl_naming

    def wrap_func_based_on_x(self, f):
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

        max_val = f(self.limit_startx[1])
        min_val = f(self.limit_startx[0])

        sizetop = (max_val * self.hard_limitx) - max_val
        sizebottom = (min_val * self.hard_limitx) - min_val

        def wrapped_func(
            x,
            f=f,
            limit_start=self.limit_startx,
            sizetop=sizetop,
            sizebottom=sizebottom,
            max_val=max_val,
            min_val=min_val,
        ):
            if (x >= limit_start[0]) and (x <= limit_start[1]):
                return f(x)
            elif x < limit_start[0]:
                return np.tanh(limit_start[0] - x, dtype=np.float32) * sizebottom + min_val
            elif x > limit_start[1]:
                return np.tanh(x - limit_start[1], dtype=np.float32 ) * sizetop + max_val
            else:
                print(x, f(x), "ISSUE")

        return wrapped_func

    def wrap_func_based_on_y(self, f):
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
        sizetop = np.abs((self.hard_limity[1] - self.limit_starty[1]))
        sizebottom = np.abs((self.hard_limity[0] - self.limit_starty[0]))

        def wrapped_func(
            x,
            f=f,
            limit_starty=self.limit_starty,
            sizetop=sizetop,
            sizebottom=sizebottom,
        ):
            default = f(x)
            if np.isnan(default):  # QUick fix case where f diverged completely.
                return sizetop
            if (default >= limit_starty[0]) and (default <= limit_starty[1]):
                return default
            elif default < limit_starty[0]:
                return np.tanh(default - limit_starty[0]) * sizebottom + limit_starty[0]
            elif default > limit_starty[1]:
                return np.tanh(default - limit_starty[1]) * sizetop + limit_starty[1]

        return wrapped_func
