import numpy as np
from typing import List, Callable



class ExogenousInfluences:
    """
    This class provides mechanisms to:
    - Generate random exogenous variables and their instantaneous effects on system variables.
    - Support both linear and nonlinear exogenous influences, with configurable probability and parameter ranges.
    - Integrate custom nonlinear relationship samplers.
    - Sample coefficient matrices representing the influence structure, and sample exogenous values for use in synthetic data generation.
    Attributes:
        rng (np.random.Generator): Random number generator for reproducibility.
        n_vars (int): Number of system variables influenced by exogenous factors.
        n_exogs (int): Number of exogenous variables.
        link_proba (float): Probability of a link between an exogenous variable and a system variable.
        param_range (List[float]): Range for the magnitude of influence coefficients.
        mirror_range (bool): Whether to allow negative coefficients by mirroring the range.
        nonlinear_proba (float): Probability of a nonlinear relationship.
        nl_sampler (callable): Sampler for nonlinear relationships.
    Methods:
        init_exogs(link_mask=None, nl_mask=None):
            Initializes the exogenous influence structure, including both linear and nonlinear components.
        get_exogs_influence():
            Samples exogenous variable values and computes their instantaneous influence on system variables, applying any nonlinear transformations as specified.
    Notes:
        - Exogenous influences are modeled separately from innovation noise for clarity.
        - The structure supports instantaneous hidden confounding by allowing shared exogenous influences.
        - Nonlinear relationships require a compatible nl_sampler to be provided.

´    """

    def __init__(
        self,
        rng: int = 42,
        n_vars : int = 3,
        n_exogs: int = 2,
        link_proba: float = 0.0,
        param_range: List[float] = [0.3, 0.5],
        mirror_range = True,
        nonlinear_proba = 0.0,
        nl_sampler = None,

    ) -> None:

        self.rng =  np.random.default_rng(rng)
        self.n_vars = n_vars
        self.n_exogs = n_exogs
        self.link_proba = link_proba
        self.mirror_range = mirror_range
        self.param_range = param_range
        self.nonlinear_proba = nonlinear_proba
        
        self.links = None
        self.nl_func = None
        self.nl_naming = None
        self.nl_sampler = nl_sampler
        
        self.vectorize_nl_apply = np.vectorize(lambda a,b: a(b))


    def random_exogs(self, link_mask=None):
        """
        random exogs are coded as n random variables that have arbitrary links to X_t
        Generates a n_exogs * n_vars matrix where each elements specifies an instantanous effect
        Returns: A coefficient matrix of n_exogs x n_vars 
            that specifies the influences on all variables of X_t
        """
 
        params = np.zeros((self.n_exogs, self.n_vars))
        potential  = self.rng.uniform(
            self.param_range[0], self.param_range[1], size=params.shape)
        # set random links
        if self.mirror_range:
            # reverse on average half of the coefficients to be negative
            select = self.rng.random(potential.shape) > 0.5
            potential[select] = potential[select] * -1
        selection = self.rng.random(params.shape) < self.link_proba
        params[selection] = potential[selection]
        if isinstance(link_mask,np.ndarray) :
            if link_mask.dtype == np.bool_:  
                params[link_mask] = potential[link_mask]
            else: 
                params[np.abs(link_mask) != 0 ] = link_mask[np.abs(link_mask) != 0]        
        return params
        
    def init_exogs(self , link_mask=None, nl_mask=None,):
        self.links = self.random_exogs(link_mask)
        if (self.nonlinear_proba) > 0 or (isinstance(nl_mask, np.ndarray)):
            # NL gen expects 3 dim array according to struc. exand dim here and remove again.
            self.nl_func, self.nl_naming = self.nl_sampler.sample_nl_relationships(
                np.expand_dims(self.links,2), self.nonlinear_proba, nl_mask
            )
            self.nl_func = self.nl_func[:,:,0]
            self.nl_naming = self.nl_naming[:,:,0]

        
    def get_exogs_influence(self):

        """
        additive instantanous influences. These might be shared (Hidden Confounding)
        """
        exog_base = self.rng.normal(0,1, (self.n_exogs))
        exog_values = np.stack([exog_base]*self.n_vars, axis=1)
        if isinstance(self.nl_func, np.ndarray):
            exog_values = self.vectorize_nl_apply(self.nl_func, exog_values)
        exog_influence = (exog_values * self.links).sum(axis=0)
        return np.expand_dims(exog_influence,axis=1), exog_base