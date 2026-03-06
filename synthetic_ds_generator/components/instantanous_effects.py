import numpy as np
from typing import List, Callable
from scipy.linalg import expm
import networkx as nx


class InstantanousEffects:
    """
    InstantanousEffects
    ====================
    This class models the generation and application of instantaneous (contemporaneous) effects in a synthetic dataset generator, particularly for time series or causal structure simulations.
    It is designed to flexibly create random or masked acyclic effect matrices (adjacency matrices) that encode how variables instantaneously influence each other within the same time step, supporting both linear and nonlinear relationships.
    Key Features:
    -------------
    - **Acyclic Structure Enforcement:** Ensures that the generated instantaneous effect matrix is acyclic, preventing feedback loops within the same time step, as required by many causal discovery algorithms.
    - **Randomized Link Generation:** Randomly samples the presence and strength of instantaneous effects between variables, with configurable probability and parameter ranges.
    - **Masking and Customization:** Supports user-defined masks to enforce or restrict specific links or nonlinearities, allowing for fine-grained control over the generated structure.
    - **Nonlinear Effects:** Integrates nonlinear relationships between variables via a user-supplied sampler, with configurable probability and masking.
    - **Nonstationarity:** Allows for controlled, random perturbations of the effect matrix to simulate nonstationary environments.
    - **Topological Ordering:** Computes a topological order of variables based on the effect structure, ensuring correct sequential application of effects.
    Parameters:
    -----------
    - rng : np.random.Generator
        Random number generator for reproducibility.
    - n_vars : int
        Number of variables in the system.
    - link_proba : float
        Probability of an instantaneous effect (edge) between any two variables.
    - param_range : List[float]
        Range for the magnitude of effect coefficients.
    - mirror_range : bool
        If True, randomly assigns negative signs to half of the coefficients.
    - sample_tries : int
        Number of attempts to sample an acyclic effect matrix.
    - nonlinear_proba : float
        Probability of a nonlinear relationship between variables.
    - nl_sampler : object
        Sampler object for generating nonlinear functions.
    - nonstationary_change : float
        Maximum magnitude of random perturbations for nonstationarity.
    Methods:
    --------
    - random_links(link_mask=None)
        Generates a random acyclic instantaneous effect matrix, optionally applying a mask.
    - restricted_links(current_links)
        Perturbs an existing effect matrix within specified bounds to simulate nonstationarity.
    - init_instantanous_influence(link_mask=None, nl_mask=None, empty_struc=False, mask_restriction=None)
        Initializes the instantaneous effect structure, including nonlinearities and topological order.
    - get_instantanous_effect(current_ts)
        Applies the instantaneous effects (linear and nonlinear) to a given time series vector in topological order.
    Usage:
    ------
    This class is intended for use in synthetic data generation pipelines where the instantaneous (within-time-step) causal structure must be controlled, randomized, or perturbed.
    It is suitable for benchmarking causal discovery algorithms, simulating interventions, or studying the impact of nonstationarity and nonlinearities in time series models.

    ´"""

    def __init__(
        self,
        rng: int = 42,
        n_vars: int = 5,
        link_proba: float = 0.0,
        param_range: List[float] = [0.3, 0.5],
        mirror_range=True,
        sample_tries: int = 50,
        nonlinear_proba: float = 0.0,
        nl_sampler=None,
        nonstationary_change=0.1,
    ) -> None:
        self.rng = np.random.default_rng(rng)
        self.n_vars = n_vars
        self.link_proba = link_proba
        self.mirror_range = mirror_range
        self.param_range = param_range
        self.nonlinear_proba = nonlinear_proba
        self.sample_tries = sample_tries
        self.nl_sampler = nl_sampler
        self.nonstationary_change = nonstationary_change

        self.vectorize_nl_apply = np.vectorize(lambda a, b: a(b))

        self.links = None  # needs init
        self.nl_func = None
        self.nl_naming = None

    def random_links(self, link_mask=None):
        """
        Instantanous effects are codes as a two dimensional tensor (EffectxCause)
        Importantly. We sample ONLY noncyclic effects. by controling the matrix.

        Returns: n_vars * n_vars matrix where each elements specifies an instantanous effect
        """

        for x in range(self.sample_tries):
            links = self.rng.random((self.n_vars, self.n_vars)) < self.link_proba
            # Remove diagonal as it MUST be 0
            np.fill_diagonal(links, 0)
            if isinstance(link_mask, np.ndarray):
                links[link_mask != 0] = 1
            # Check wether resulting matrix is acyclic via tr(e^A) (no-tears paper)
            if self.n_vars == np.matrix.trace(expm(links)):
                break  # if this breaks we have a acyclic matrix
        if x == self.sample_tries - 1:
            raise ValueError(
                "Could not sample an acyclic matrix in {} tries. Please check your link_proba and masks.".format(
                    self.sample_tries
                )
            )
        # now determined the coefficients.
        potential = self.rng.uniform(
            self.param_range[0], self.param_range[1], size=links.shape
        )
        # set random links
        if self.mirror_range:
            # reverse on average half of the coefficients to be negative
            select = self.rng.random(potential.shape) > 0.5
            potential[select] = potential[select] * -1

        params = np.zeros(links.shape)
        # Set all vars
        params[links] = potential[links]

        if isinstance(link_mask, np.ndarray):
            if link_mask.dtype == np.bool_:
                params[link_mask] = potential[link_mask]
            else:
                params[np.abs(link_mask) > 0] = link_mask[np.abs(link_mask) > 0]

            if self.n_vars != np.matrix.trace(expm(links)):
                raise ValueError(
                    "Cyclic matrix created. Likely the mask enforced loops."
                )
        return params

    def restricted_links(self, current_links):
        if self.nonstationary_change == 0:
            potential = np.zeros(current_links.shape)

        potential = self.rng.uniform(
            -self.nonstationary_change,
            self.nonstationary_change,
            size=current_links.shape,
        )
        new_links = current_links.copy()
        mask = np.abs(current_links) > 0
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

    def init_instantanous_influence(
        self, link_mask=None, nl_mask=None, empty_struc=False, mask_restriction=None
    ):
        if not empty_struc:
            if mask_restriction is not None:
                self.links = self.restricted_links(current_links=mask_restriction)
            else:
                self.links = self.random_links(link_mask=link_mask)
        else:
            self.links = np.zeros((self.n_vars, self.n_vars))

        if (self.nonlinear_proba) > 0 or (isinstance(nl_mask, np.ndarray)):
            # NL gen expects 3 dim array according to struc. expand dim here.
            self.nl_func, self.nl_naming = self.nl_sampler.sample_nl_relationships(
                np.expand_dims(self.links, 2), self.nonlinear_proba, nl_mask
            )
            self.nl_func = self.nl_func[:, :, 0]
            self.nl_naming = self.nl_naming[:, :, 0]

        # get the topological order of the links
        G = nx.DiGraph()
        n_vars = self.links.shape[0]
        for i in range(n_vars):
            for j in range(n_vars):
                if self.links[i, j] != 0:
                    G.add_edge(j, i)  # cause -> effect
        self.topo_order = list(nx.topological_sort(G))

    def get_instantanous_effect(self, ts):
        current_ts = ts.copy()
        assert isinstance(self.links, np.ndarray) or self.link_proba == 0, (
            "Initialize First"
        )
        assert current_ts.shape == (self.n_vars, 1), "TS does not match the links"
        # stack ts and reverse
        if isinstance(self.links, np.ndarray):
            for item in self.topo_order:
                if isinstance(self.nl_func, np.ndarray):
                    temporary_transform = self.vectorize_nl_apply(
                        self.nl_func[item], current_ts[:, 0]
                    )
                    current_ts[item] += (
                        temporary_transform * self.links[item]
                    ).sum()  # over lags and vars
                else:
                    # multiply each element of data with the corresponding coefficient and sum the effect
                    current_ts[item] += (
                        current_ts[:, 0] * self.links[item]
                    ).sum()  # over lags and vars

            return current_ts
        else:
            return current_ts
