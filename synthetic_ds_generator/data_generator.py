import numpy as np
from typing import List, Callable
from components.tools import (
    calc_steps_according_to_change_points,
    normalize_ts,
    check_divergence,
)
import pandas as pd
from scipy.optimize import root_scalar



class SyntheticDataGenerator:
    """
    SyntheticDataGenerator is a wrapper class for generating synthetic time series data based on multiple modular components.
    This generator supports a variety of configurations, including random or constrained graph sampling,
    regime changes (change points), and both linear and nonlinear relationships. It is designed to facilitate
    the creation of datasets for causal inference and time series analysis, with options to enforce specific
    graph structures, introduce exogenous influences, and simulate nonstationary processes.
    Key Features:
    - Modular design: Accepts custom components for noise, observation, exogenous variables, structural and instantaneous effects.
    - Change points: Supports regime changes in the data generation process, allowing for abrupt or gradual transitions.
    - Masking: Allows for the enforcement of specific links or nonlinearities in the causal graph via masks.
    - Normalization: Supports min-max and standardization normalization of generated data.
    - Missing data: Can randomly introduce missing values and interpolate them.
    - Divergence filtering: Recursively resamples if the generated process diverges, but does not guarantee stationarity.
    - Exogenous components: Handles external influences and their memory.
    - Confounding: Optionally removes variables to simulate confounding effects.
    Parameters:
        ino_n (Callable): Innovation noise generator.
        obs_n (Callable): Observation noise generator.
        exog (Callable): Exogenous variable generator.
        struc (Callable): Structural causal model generator.
        instant (Callable): Instantaneous effect generator.
        rng (int, optional): Random number generator. Defaults to np.random.default_rng(42).
        normalize (str, optional): Normalization method ('minmax', 'standardization', or None).
        time_series_n (int): Number of time steps to generate.
        standardization_factor (float): Weight for standardization blending if this is used (1 by default meaning full standardization).
        change_points (List[int]): List of change point indices for regime changes.
        drop_struc_for_window (bool): Whether to drop the structure for certain windows.
        nonstationary (bool): If True, allows for nonstationary regime changes.
        interpolate (float): Fraction of values to randomly mask missing vakues and interpolate.
        remove_n_variables_for_confounding (int): Number of variables to remove for confounding simulation.
    Methods:
        # Use this function to generate a sample of synthetic data according to hyperparameters.
        get_sample(...): Generates a synthetic dataset, handling change points, normalization, and missing data.


    Notes:
    - The class does not guarantee stationarity of the generated processes (We however test var stability in the struc class).
    - Divergence is checked empirically; if divergence is detected, the process is resampled.
    - Care should be taken when specifying masks and sampling parameters to avoid excessive resampling or degenerate outputs.

    """

    def __init__(
        self,
        # Modular components
        inno_n: Callable = None,
        obs_n: Callable = None,
        exog: Callable = None,
        lagged: Callable = None,
        instant: Callable = None,
        # General params
        rng: int = 42,
        verbose: int = 0,
        normalize: str = None,
        time_series_n: int = 600,
        standardization_factor: float = 1,
        change_points: List[int] = [],
        drop_struc_for_window: bool = False,
        nonstationary: bool = False,
        interpolate: float = 0.0,
        remove_n_variables_for_confounding: int = 0,
        section_tolerance: int = 50,
        sample_attempts: int = 10,
        missingness_type: str = "MCAR",
        missingness_base_path = "tools_and_examples/semi_synthetic_bases/rivers_ts_flood.csv"
    ) -> None:
        # Modular components.
        self.inno_n = inno_n
        self.obs_n = obs_n
        self.exog = exog
        self.lagged = lagged
        self.instant = instant

        self.rng = np.random.default_rng(rng)
        self.verbose = verbose
        self.normalize = normalize
        self.standardization_factor = standardization_factor
        self.nonstationary = nonstationary
        self.time_series_n = time_series_n
        self.change_points = calc_steps_according_to_change_points(
            change_points, time_series_n
        )
        self.interpolate = interpolate
        self.missingness_type = missingness_type
        self.drop_struc_for_window = drop_struc_for_window
        self.divergence_test = check_divergence
        self.section_tolerance = section_tolerance
        self.sample_attempts = sample_attempts
        self.missingness_base_path = missingness_base_path
        self.remove_n_variables_for_confounding = remove_n_variables_for_confounding
        # Load real world data for MAR (dependent on the real world time series) missingness if necessary.
        if self.missingness_type == "MAR":
            self.river_samples = pd.read_csv(self.missingness_base_path, index_col=0)
            self.river_samples = (self.river_samples - self.river_samples.min()) / (self.river_samples.max() - self.river_samples.min())
            self.river_samples = self.river_samples.interpolate().bfill().T

    def attempt_sample_generation(
        self, link_mask=None, nl_mask=None, instant_link_mask=None, instant_nl_mask=None
    ):
        (
            ts_stack,
            link_stack,
            instant_links_stack,
            exog_links_stack,
            exog_memory_stack,
            nl_namings_stack,
            instant_nl_namings_stack,
            exog_nl_namings_stack,
        ) = ([], [], [], [], [], [], [], [])
        if (len(self.change_points)) > 1 and (self.verbose > 0):
            print("Change points:", self.change_points)

        for n, step in enumerate(self.change_points):
            # select old links from mask according to specifications:
            if len(link_stack) == 0:
                old_links = None
                old_instant_links = None
            else:  # Dropping causal graph somehwere requires not the last but the last valid one.
                # this is only valid if we start with a proper section (which we currently do.)
                if self.drop_struc_for_window:
                    old_links = link_stack[-2] if len(link_stack) > 1 else None
                    old_instant_links = (
                        instant_links_stack[-2]
                        if len(instant_links_stack) > 1
                        else None
                    )
                else:
                    old_links = link_stack[-1]
                    old_instant_links = instant_links_stack[-1]

            for attempt in range(
                self.section_tolerance
            ):  # try to generate a valid sample several times.
                if attempt > 0:
                    print(f"  Resampling section {n} (attempt {attempt + 1})")

                (
                    out,
                    links,
                    instant_links,
                    exog_links,
                    exog_memory,
                    nl_namings,
                    instant_nl_namings,
                    exog_nl_namings,
                ) = self.generate_section(
                    init_data=(
                        None if n == 0 else ts_stack[-1][:, -self.lagged.max_lags :]
                    ),
                    n_steps=step,
                    section=n,
                    link_mask=link_mask,
                    nl_mask=nl_mask,
                    instant_link_mask=instant_link_mask,
                    instant_nl_mask=instant_nl_mask,
                    old_links=old_links,
                    old_instant_links=old_instant_links,
                )

                if isinstance(out, np.ndarray):
                    break

            if attempt == self.section_tolerance - 1:
                print(
                    "Failed to generate a valid section after several attempts, returning None."
                )
                return False, None, None, None, None, None, None, None, attempt

            ts_stack.append(out)
            link_stack.append(links)
            instant_links_stack.append(instant_links)
            exog_links_stack.append(exog_links)
            exog_memory_stack.append(exog_memory)
            nl_namings_stack.append(nl_namings)
            instant_nl_namings_stack.append(instant_nl_namings)
            exog_nl_namings_stack.append(exog_nl_namings)

        combined_ts = np.concatenate(ts_stack, axis=1)
        combined_exog = np.concatenate(exog_memory_stack, axis=0).T

        return (
            combined_ts,
            np.array(link_stack),
            np.array(instant_links_stack),
            np.array(exog_links_stack) if self.exog.link_proba > 0 else None,
            combined_exog if self.exog.link_proba > 0 else None,
            np.array(nl_namings_stack) if self.lagged.nonlinear_proba > 0 else None,
            np.array(instant_nl_namings_stack)
            if self.instant.nonlinear_proba > 0
            else None,
            np.array(exog_nl_namings_stack) if self.exog.nonlinear_proba > 0 else None,
            attempt,
        )

    def get_sample(
        self, link_mask=None, nl_mask=None, instant_link_mask=None, instant_nl_mask=None
    ):
        """
        Main function that handels change points in the generation process.
        If the generation process is fixed then this basically just wraps "generate section"
        SHould always be called to generate a sample.
        """

        if self.inno_n is None:
            raise ValueError("inno_n must be specified")
        if self.obs_n is None:
            raise ValueError("obs_n must be specified")
        if self.exog is None:
            raise ValueError("exog must be specified")
        if self.lagged is None:
            raise ValueError("struc must be specified")
        if self.instant is None:
            raise ValueError("instant must be specified")

        for sample_attempt in range(self.sample_attempts):
            (
                combined_ts,
                link_stack,
                instant_links_stack,
                exog_links_stack,
                combined_exog,
                nl_namings_stack,
                instant_nl_namings_stack,
                exog_nl_namings_stack,
                section_attempts,
            ) = self.attempt_sample_generation(
                link_mask=link_mask,
                nl_mask=nl_mask,
                instant_link_mask=instant_link_mask,
                instant_nl_mask=instant_nl_mask,
            )

            if isinstance(combined_ts, np.ndarray):
                break
            if sample_attempt > 0:
                print(f"Resampling entire sample (attempt {sample_attempt + 1})")
                
            if sample_attempt == self.sample_attempts - 1:
                print(
                    "Failed to generate a valid sample after several attempts, returning None."
                )
                raise ValueError("Could not generate a valid sample. Check config")

        if self.normalize is not None:
            # Normalize the time series and exogenous data if specified.
            combined_ts, combined_exog = self.data_scaling(combined_ts, combined_exog)

        if self.interpolate > 0.0:
            if self.missingness_type == "MCAR":
                # Generate missing values and interpolates.
                combined_ts = self.mcar_holes_and_interpolate(combined_ts)
            elif self.missingness_type == "MNAR":
                combined_ts = self.mnar_holes_and_interpolate(combined_ts)
            elif self.missingness_type == "MAR":
                combined_ts = self.mar_real_world_holes_and_interpolate(combined_ts)

            else: 
                raise NotImplementedError("Missingness type not implemented yet.")

        if self.remove_n_variables_for_confounding > 0:
            combined_ts, link_stack, instant_links_stack = self.remove_for_confounding(
                combined_ts, link_stack, instant_links_stack
            )

        return (
            np.array(combined_ts),
            np.array(link_stack),
            np.array(instant_links_stack),
            np.array(exog_links_stack) if self.exog.link_proba > 0 else None,
            np.array(combined_exog) if self.exog.link_proba > 0 else None,
            np.array(nl_namings_stack) if self.lagged.nonlinear_proba > 0 else None,
            np.array(instant_nl_namings_stack) if self.instant.nonlinear_proba > 0 else None,
            np.array(exog_nl_namings_stack) if self.exog.nonlinear_proba > 0 else None,
            [sample_attempt, section_attempts]
        )

    def generate_section(
        self,
        init_data=None,
        n_steps=0,
        section=0,
        link_mask=None,
        nl_mask=None,
        instant_link_mask=None,
        instant_nl_mask=None,
        old_links=None,
        old_instant_links=None,
        exog_mask=None,
    ):
        self.inno_n.reset()  # Currently only used to reset heteroskedasticity.
        self.obs_n.reset()
        self.exog.init_exogs() # TODO I am not providing the mask here yet and this doesnt support conditional sampling. Fix if necessary.
        self.lagged.init_random_process(
            link_mask,
            nl_mask=nl_mask,
            # we set activate empty struc every second window of the change points.
            empty_struc=(
                True if (self.drop_struc_for_window and ((section % 2) > 0)) else False
            ),
            mask_restriction=old_links if self.nonstationary else None,
        )

        self.instant.init_instantanous_influence(
            instant_link_mask,
            instant_nl_mask,
            empty_struc=(
                True if (self.drop_struc_for_window and ((section % 2) > 0)) else False
            ),
            mask_restriction=old_instant_links if self.nonstationary else None,
        )
        out = self.init_time_series(self.lagged.links, n_steps, init_data)
        # now we generates the ts step by step
        exog_memory = []
        for x in range(self.lagged.max_lags, n_steps + self.lagged.max_lags):
            # Kill process if divergence likely.
            if self.divergence_test:
                if self.divergence_test(out):  # give current ts.
                    print("break!")
                    out = None
                    break
                
                # lagged causal process
                if self.lagged.link_proba != 0 or isinstance(link_mask, np.ndarray):
                    step = self.lagged.get_step(out[:, x - self.lagged.max_lags : x])
                else:
                    step = np.zeros((self.lagged.n_vars, 1))

                if self.exog.link_proba != 0:
                    # exogenous influences
                    exog_effect, exog_base = self.exog.get_exogs_influence()
                    step += exog_effect.astype(float)
                    exog_memory.append(exog_base)

                # innovation noise cannot be turned off currently. If determinism is needed it should be added.
                step += self.inno_n.get_noise(step)
                if self.instant.link_proba != 0  or isinstance(instant_link_mask, np.ndarray):
                    # Instantanous effects.
                    step = self.instant.get_instantanous_effect(step)
                # assign
                out[:, x : x + 1] = step

        if out is not None:  # we can skip if the generation failed.
            out = out[:, self.lagged.max_lags :]  # remove init.
            if self.obs_n.snr:  # None for no noise.
                out += self.obs_n.get_noise(out)

        return (
            out,
            self.lagged.links,  # Lagged links. Add this later.
            self.instant.links,  # Instant links. Add this later.
            self.exog.links if self.exog.link_proba > 0 else None,  # Exog links.
            np.array(exog_memory),
            self.lagged.nl_naming,
            self.instant.nl_naming,
            self.exog.nl_naming,
        )

    def init_time_series(self, links, steps, init_data):
        # init the time series if necessary.
        # If no init_data is given, we initialize with random guassian noise
        out = np.zeros((links.shape[0], steps + self.lagged.max_lags))
        if isinstance(init_data, np.ndarray):
            out[:, : self.lagged.max_lags] = init_data
        elif links.sum() == 0:  # no lagged structure:
            out[:, :1] = self.rng.normal(0, 1, size=(links.shape[0], 1))
        else:
            out[:, : self.lagged.max_lags] = self.rng.normal(0, 1, size=links.shape[1:])
        return out

    def data_scaling(self, combined_ts, combined_exog):
        if self.normalize == "minmax":
            combined_ts = normalize_ts(combined_ts)
            if combined_exog:
                combined_exog = normalize_ts(combined_exog)
        elif self.normalize == "standardization":
            standardized_ts = (
                combined_ts - np.mean(combined_ts, axis=1, keepdims=True)
            ) / np.std(combined_ts, axis=1, keepdims=True)
            combined_ts = (
                standardized_ts * self.standardization_factor
                + (1 - self.standardization_factor) * combined_ts
            )
            if combined_exog:
                standardized_exog = (
                    combined_exog - np.mean(combined_exog, axis=1, keepdims=True)
                ) / np.std(combined_exog, axis=1, keepdims=True)
                combined_exog = (
                    standardized_exog * self.standardization_factor
                    + (1 - self.standardization_factor) * combined_exog
                )

        return combined_ts, combined_exog
    
    
    
    def mar_real_world_holes_and_interpolate(self, combined_ts, b =3.5):
        """
        # Samples a random time series for each synthetic data. 
        # Then uses its values to determine missingness in the synthetic data.
        # The probabilities are then scales similar to the mnar case
        """
        
        # Sample random real world time series for missingness basis.
        random_indices = self.rng.integers(0, self.river_samples.shape[0], size=combined_ts.shape[0])
        basis_ts = self.river_samples.values[random_indices,: ]
        # randomly select a subset of the rivers that match the timesteps in combined ts:
        cut = self.rng.integers(0,self.river_samples.shape[1]-combined_ts.shape[1])
        basis_ts = basis_ts[:,cut:cut+combined_ts.shape[1]]

        # Now we find the 'a' intercept to achieve the desired missingness rate.
        def objective_function(a_candidate):
            linear_term = a_candidate + (b * basis_ts)
            # Using a numerically stable sigmoid
            probs = 1 / (1 + np.exp(-linear_term))
            current_mean = np.mean(probs)
            return current_mean - self.interpolate 

        sol = root_scalar(objective_function, bracket=[-100, 100], method='brentq')
        optimal_a = sol.root
        # 3. Calculate final probabilities with the found 'a'
        probabilities = 1 / (1 + np.exp(-(optimal_a + b * basis_ts)))
        # Mask each value with the specified probability
        mask = self.rng.uniform(0, 1, combined_ts.shape) < probabilities
        combined_ts[mask] = np.nan

        # NOw interpolate the missing values.
        combined_ts = (
            pd.DataFrame(combined_ts.T)
            .interpolate(method="linear")
            .bfill()
            .values.T)
        return combined_ts
    
    
    
    def mnar_holes_and_interpolate(self,combined_ts, b=3.5):
        """
        Finds the intercept 'a' required to achieve a specific expected 
        missing rate given data 'x' and slope 'b'.
        Calculates the probability of a value being missing based on the 
        logistic function: p(R=1) = 1 / (1 + e^(-(a + b*x)))
        Parameters:
        -----------
        x : np.ndarray
            The input data signal.
        b : float
            The slope hyperparameter (controls dependency on x).
            b = 0 implies MCAR (Missing Completely At Random).
            b != 0 implies MNAR (Missing Not At Random).
        target_rate : float
            Desired fraction of missing data (e.g., 0.2 for 20%).
            
        Returns:
        --------
        a : float
            The calculated intercept.
        probs : np.ndarray
            The resulting probabilities matching the target rate.
        """
        
        # 1. Define the function to optimize (Find 'a' where result is 0)
        # Target: mean( 1 / (1+e^-(a+bx)) ) == target_rate
        def objective_function(a_candidate):
            linear_term = a_candidate + (b * combined_ts)
            # Using a numerically stable sigmoid
            probs = 1 / (1 + np.exp(-linear_term))
            current_mean = np.mean(probs)
            return current_mean - self.interpolate 

        # 2. Use a root finding algorithm (Brent's method is reliable here)
        # We search for 'a' in a wide range of logits [-100, 100]
        try:
            sol = root_scalar(objective_function, bracket=[-100, 100], method='brentq')
        except ValueError:
            print("Error: Could not converge. Target rate might be reachable only with extreme 'a'.")
            return None, None

        optimal_a = sol.root
        
        # 3. Calculate final probabilities with the found 'a'
        probabilities = 1 / (1 + np.exp(-(optimal_a + b * combined_ts)))
        
        # Mask each value with the specified probability
        mask = self.rng.uniform(0, 1, combined_ts.shape) < probabilities
        combined_ts[mask] = np.nan

        # NOw interpolate the missing values.
        combined_ts = (
            pd.DataFrame(combined_ts.T)
            .interpolate(method="linear")
            .bfill()
            .values.T)
        return combined_ts


    def mcar_holes_and_interpolate(self, combined_ts):
        """
        Removes random values COMPLETELY AT RANDOM"
        """

        # Mask x % of values with nan:
        mask = self.rng.uniform(0, 1, combined_ts.shape) < self.interpolate
        combined_ts[mask] = np.nan
        # NOw interpolate the missing values.
        combined_ts = (
            pd.DataFrame(combined_ts.T)
            .interpolate(method="linear")
            .bfill()
            .values.T)

        return combined_ts
    
    


    def remove_for_confounding(self, combined_ts, link_stack, instant_links_stack):
        # remove some variables for confounding.
        # We remove from the beginning for simplicity.
        if self.remove_n_variables_for_confounding > 0:
            combined_ts = combined_ts[: -self.remove_n_variables_for_confounding, :]
            link_stack = link_stack[
                :,
                : -self.remove_n_variables_for_confounding,
                : -self.remove_n_variables_for_confounding,
                :,
            ]

            instant_links_stack = instant_links_stack[
                :,
                : -self.remove_n_variables_for_confounding,
                : -self.remove_n_variables_for_confounding,
            ]
        return combined_ts, link_stack, instant_links_stack
