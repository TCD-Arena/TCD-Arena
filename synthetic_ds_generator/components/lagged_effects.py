import numpy as np
from typing import List, Callable


class LaggedEffects:
    """
    Structural_Equation
    
    A synthetic data generator for causal graphs and time series (TS) samples, supporting both linear and nonlinear relationships.
    This class enables the creation of synthetic datasets for causal inference and time series analysis by simulating structural 
    equations with configurable parameters. It supports the specification of the number of variables, lags, link probabilities, 
    parameter ranges, and nonlinearities. The generator can enforce stability, handle nonstationarity, and apply custom masks 
    to control the structure of the generated process.
    Key Features:
        rng (np.random.Generator): Random number generator for reproducibility.
        n_vars (int): Number of variables (nodes) in the system.
        max_lags (int): Maximum lag order for the VAR process.
        link_proba (float): Probability that a link (edge) exists between variables.
        param_range (List[float]): Range [min, max] for linear link coefficients.
        mirror_range (bool): If True, randomly assigns negative signs to coefficients.
        nonlinear_proba (float): Probability that a nonlinear relationship is present.
        nl_sampler (object): Sampler object for generating nonlinear functions.
        nonstationary_change (float): Magnitude of nonstationary changes in coefficients.
        stability_tolerance (int): Maximum attempts to find a stable VAR process.
        alternative_coeff_ts (int): Number of variables with alternative coefficient ranges.
        alternativ_parameter_range (List[float]): Range for alternative coefficients.
        alternative_link_proba (float): Link probability for alternative coefficients.
        links (np.ndarray): Coefficient tensor of shape (n_vars, n_vars, max_lags).
        nl_func (np.ndarray): Tensor of nonlinear functions applied to data.
        nl_naming (np.ndarray): Names or identifiers for nonlinear functions.
        vectorize_nl_apply (np.vectorize): Vectorized function for applying nonlinearities.
        check_var_stability(coefficients):
            Checks the stability of a VAR(p) process given coefficient matrices.
            Returns (is_stable: bool, max_eigenvalue: float).
        random_links(link_mask=None):
            Generates a random coefficient tensor for the VAR process, optionally applying a mask.
        restricted_links(current_links):
            Modifies existing links to introduce nonstationarity, respecting parameter bounds.
            Prepares and reshapes time series data for causal effect computation.
        init_random_process(link_mask=None, nl_mask=None, empty_struc=False, mask_restriction=None):
            Initializes the process by generating stable links and nonlinear relationships.
            Computes the next causal effect given the current time series window.
    Usage:
        Instantiate the class with desired parameters, call `init_random_process()` to initialize,
        and use `get_step()` to generate new time series samples step by step.
    """
   

    def __init__(
        self,
        # Linear params
        rng: int = 42,
        verbose: int = 0,
        n_vars: int = 5,
        max_lags: int = 3,
        link_proba: float = 0.15,
        param_range: List[float] = [0.3, 0.5],
        mirror_range: bool = True,
        # nonlinear components
        nonlinear_proba: float = 0.0,
        nl_sampler=None,
        nonstationary_change = 0.0,
        test_for_var_stability: bool = True,
        stability_tolerance: int = 100,
        alternative_coeff_ts: int = 0,
        alternativ_parameter_range: List[float] = [0.3, 0.5],
        alternative_link_proba: float = 0.15
    ) -> None:

        self.rng = np.random.default_rng(rng)
        self.verbose = verbose
        self.n_vars = n_vars
        self.max_lags = max_lags
        self.link_proba = link_proba
        self.param_range = param_range
        self.mirror_range = mirror_range
        self.nonlinear_proba = nonlinear_proba
        self.nonstationary_change = nonstationary_change
        self.stability_tolerance = stability_tolerance
        self.test_for_var_stability = test_for_var_stability
        
        
        # Confounder ts is currently substracted from the time series n_vars.
        # However the coefficients can be
        # initialized to a different range than the main process.
        self.alternative_coeff_ts = alternative_coeff_ts
        self.alternativ_parameter_range = alternativ_parameter_range
        self.alternative_link_proba = alternative_link_proba

        self.links = None
        self.nl_func = None
        self.nl_naming = None
        self.nl_sampler = nl_sampler
        self.vectorize_nl_apply = np.vectorize(lambda a,b: a(b))
        

    def check_var_stability(self,coefficients):
        """
        Checks the stability of a VAR(p) process.
        Args:
            A_matrices (np.ndarray): A 3D numpy array of shape (p, k, k)
                                    containing the coefficient matrices A_1, ..., A_p.
                                    p = number of lags
                                    k = number of time series
        Returns:
            tuple: A tuple containing:
                - bool: True if the process is stable, False otherwise.
                - float: The maximum eigenvalue modulus.
        """
        A_matrices = coefficients.T
        if A_matrices.ndim != 3 or A_matrices.shape[1] != A_matrices.shape[2]:
            raise ValueError("A_matrices must be a 3D array of shape (p, k, k).")

        p, k, _ = A_matrices.shape
        # The total dimension of the companion matrix
        kp = k * p
        # Construct the companion matrix F
        companion_matrix = np.zeros((kp, kp))
        # Place A_1, ..., A_p in the first k rows
        companion_matrix[0:k, :] = np.hstack(A_matrices)
        # Place the identity matrix in the lower-left block
        if p > 1:
            identity_block = np.eye(k * (p - 1))
            companion_matrix[k:, 0:k*(p-1)] = identity_block
        # Calculate the eigenvalues
        eigenvalues = np.linalg.eigvals(companion_matrix)
        # Calculate the modulus (absolute value) of the eigenvalues
        eigenvalue_moduli = np.abs(eigenvalues)
        # Find the maximum modulus
        max_modulus = np.max(eigenvalue_moduli)

        return max_modulus < 1.0, max_modulus


    def random_links(self, link_mask=None):
        """
        Generates a matrix of shape (n_vars, n_vars, max_lags),
          ((Effect, Cause, Lag) that determines the link coefficients.
            0 specifies no relationship
        """
        # draw a random set of params based on the specified range
        
        if self.alternative_coeff_ts == 0: 
            
            potential = self.rng.uniform(
                self.param_range[0],
                self.param_range[1],
                size=(self.n_vars, self.n_vars, self.max_lags),
            )
        else:
            potential1 = self.rng.uniform(
                self.param_range[0],
                self.param_range[1],
                size=(self.n_vars, self.n_vars - self.alternative_coeff_ts, self.max_lags),
            )
            
            potential2 = self.rng.uniform(
                self.alternativ_parameter_range[0],
                self.alternativ_parameter_range[1],
                size=(self.n_vars, self.alternative_coeff_ts, self.max_lags),
            )
            # Create a (n_vars, n_vars, max_lags) zero array
            potential = np.zeros((self.n_vars, self.n_vars, self.max_lags))
            # Fill the top-left block with potential1
            potential[:, :self.n_vars-self.alternative_coeff_ts, :] = potential1
            # Fill the bottom-right block with potential2
            potential[:, self.n_vars-self.alternative_coeff_ts:, :] = potential2

        # we copy and remove.
        add = potential.copy()
        if self.mirror_range:
            # reverse on average half of the coefficients to be negative
            select = self.rng.random(add.shape) > 0.5
            add[select] = add[select] * -1
        # remove everything that where a random sample is under the threshold.
        if self.alternative_coeff_ts == 0:
            add[self.rng.random(potential.shape) > self.link_proba] = 0
            
        else:
            # Remove elements according to link_proba for the first block
            mask1 = self.rng.random(potential1.shape) > self.link_proba
            add[:, :self.n_vars-self.alternative_coeff_ts, :][mask1] = 0
            # Remove elements according to alternative_link_proba for the second block
            mask2 = self.rng.random(potential2.shape) > self.alternative_link_proba
        
            add[:, self.n_vars-self.alternative_coeff_ts:, :][mask2] = 0
        
        # Either adds a random param according to boolean mask or
        #  adds the specific parameters of float mask
        if isinstance(link_mask, np.ndarray):
            if link_mask.dtype == np.bool_:
                add[link_mask] = potential[link_mask]
            else:
                add[np.abs(link_mask) > 0] = link_mask[
                    np.abs(link_mask) > 0
                ]
        return add
    
   

    def restricted_links(self, current_links): 

        if self.alternative_coeff_ts != 0: 
            raise ValueError("Combining link restrictions with alternative coeff ts is not implemented yet.")   
        
        if self.nonstationary_change == 0: 
            potential = np.zeros((self.n_vars, self.n_vars, self.max_lags))
        else:
            potential = self.rng.uniform(
                -self.nonstationary_change,
                self.nonstationary_change,
                size=(self.n_vars, self.n_vars, self.max_lags),
            )
        new_links = current_links.copy()
        mask = np.abs(current_links) > 0
        
        # Note the current implementation prioritizes the prevention of higher coefficients over arbitrary low coefficients.
         # Calculate the proposed new values
        proposed = new_links + potential
        # Check if proposed values exceed param_range (likely causing divergence)
        over_max = proposed > self.param_range[1]
        under_min = proposed < -self.param_range[1]
        change = mask & (over_max | under_min)

        potential[change] = potential[change] * -1  # flip sign of potential
        updated_proposal = new_links + potential
       
        # Apply the (possibly flipped) potential
        new_links[mask] = updated_proposal[mask]
        return new_links
    
        

    def select_relevant_horizon_and_transform(self, data):
        # select the relevant horizon for the next step
        # Flip the time dimension as lags are ordered from left to right.
        data = np.flip(data, axis=1)
        # Repeat for each variable so it matches the parameter matrix
        data = np.stack([data for x in range(self.n_vars)], axis=0)
        return data

    def init_random_process(self, link_mask=None, nl_mask=None, empty_struc=False, mask_restriction=None):
        if not empty_struc:
                        
            for x in range(self.stability_tolerance):
                
                if isinstance(mask_restriction, np.ndarray):
                    self.links = self.restricted_links(current_links=mask_restriction)
                else:
                    self.links = self.random_links(link_mask=link_mask)
                    
                if self.check_var_stability(self.links)[0] or (self.test_for_var_stability is False):
                    break
                else:
                    if self.verbose > 0:
                        print(
                            f"Attempt {x + 1}: The VAR process is not stable. "
                            "Reinitializing with new random links."
                        )
            if x == self.stability_tolerance - 1:
                print(
                    "Warning: Could not find a stable VAR process after "
                    f"{self.stability_tolerance} attempts. "
                    "Consider increasing the stability tolerance."
                )
                    
        else:
            self.links = np.zeros((self.n_vars, self.n_vars, self.max_lags))
            
        if (self.nonlinear_proba) > 0 or (isinstance(nl_mask, np.ndarray)):
            self.nl_func, self.nl_naming = self.nl_sampler.sample_nl_relationships(
                self.links, self.nonlinear_proba, nl_mask
            )
        

    def get_step(self, current_ts):
        assert isinstance(self.links, np.ndarray), "Initialize First"

        # remove if more ts is available than needed.
        data = current_ts[:, -self.max_lags :]
        # stack ts and reverse
        data = self.select_relevant_horizon_and_transform(data)
        assert data.shape == self.links.shape, "TS does not match the links"
        # apply nonlinear transformation to data according to nl_tensor
        if isinstance(self.nl_func, np.ndarray):
            data = self.vectorize_nl_apply(self.nl_func, data)

        # multiply each element of data with the corresponding coefficient and sum the effect
        causal_effect = (
            (data * self.links).sum(axis=1).sum(axis=1)
        )  # over lags and vars
        return np.expand_dims(causal_effect, axis=1)
