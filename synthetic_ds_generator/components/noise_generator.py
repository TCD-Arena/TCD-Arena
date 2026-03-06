import numpy as np
from typing import Callable
from components.tools import sinus_year_cycle, scale_up_through_time,mean_var_shift
import pandas as pd

class NoiseGenerator:
    
    """
     flexible synthetic noise generator for time series and causal graph data.
    This class supports the generation of various types of noise
    Key Features:
    - Supports multiple noise types: additive, multiplicative, time-dependent, autoregressive, shock, and common innovation.
    - Allows mixing of noise types with configurable proportions.
    - Supports non-Gaussian additive noise (Weibull, exponential, uniform, or truncated/inverse normal).
    - Handles non-equal variance across variables.
    - Allows time-dependent scaling of noise via user-defined or built-in cycles.
    - Can scale noise to achieve a target signal-to-noise ratio (SNR).
    - Maintains state for autoregressive and time-dependent noise.
    
    Parameters:
        rng (np.random.Generator): Random number generator for reproducibility.
        modus (str): Mode of operation, either "obs" (observational noise) or "inno" (innovation noise).
        additive (bool): Whether to include additive noise.
        multiplicative (bool): Whether to include multiplicative noise.
        time_dependent (bool): Whether to include time-dependent noise.
        autoregressive (bool): Whether to include autoregressive noise.
        common (bool): Whether to include common innovation noise (confounding).
        shock (bool): Whether to include shock noise.
        shock_proba (float): Probability of a shock occurring at each time step.
        time_dependent_cycle (Callable or str): Function or string identifier for time-dependent scaling (e.g., "annual_sin").
        shock_size (float): Amplitude of shocks, relative to the mean of recent time steps.
        snr (float): Desired signal-to-noise ratio (used for scaling noise in "obs" mode).
        non_additive_noise_proba (float): Probability of using non-additive noise types in "inno" mode.
        autoregressive_innovation (float): Autoregressive coefficient for autoregressive noise.
        non_gaussian_additive (float): Proportion of non-Gaussian additive noise.
        non_equal_variance_range (list or bool): Range [low, high] for variable-specific noise variance.
        scale_for_time_dependent (float): Scaling factor for time-dependent noise.
        which_non_gaussian (str): Type of non-Gaussian distribution ("weibull", "exponential", "uniform", "inv_normal").
    Methods:
        reset(): Resets internal state (e.g., for autoregressive and time-dependent noise).
        get_noise(ts_in): Generates noise for the given input, combining enabled noise types and scaling as needed.
    Usage:
        Instantiate the class with desired noise configuration, then call get_noise(ts_in) with a 2D array (n_vars, ts_length) to generate noise matching the specified characteristics.
    Raises:
        AssertionError: If the sum of enabled noise types exceeds allowed limits for the selected mode.
        NotImplementedError: If an unsupported mode is specified.
        
    ´"""

    def __init__(
        self,
        rng: int = 42,
        modus: str = "obs",
        additive: bool = True,
        multiplicative: bool = False,
        time_dependent: bool = False,
        autoregressive: bool = False,
        common: bool = False,
        semi_synthetic: bool = False,   # Only implemented for obs
        shock: bool = False,
        shock_proba: float = 0.05,
        time_dependent_cycle: str = "mean_var_shift",
        shock_size: float = 5,
        snr: float = None,
        non_additive_noise_proba: float = 0,
        autoregressive_innovation: float = 0.5,
        non_gaussian_additive: float = 0,
        non_equal_variance_range: list = False, # [low, high, low, high]
        scale_for_time_dependent: float = 0.005,
        which_non_gaussian: str = "weibull",
        semi_synthetic_noise_path: str = "tools_and_examples/semi_synthetic_bases/"

    ) -> None:
        
        self.rng =  np.random.default_rng(rng)
        self.additive = additive
        self.multiplicative = multiplicative
        self.time_dependent = time_dependent
        self.shock = shock
        self.autoregressive = autoregressive
        self.common = common
        self.semi_synthetic = semi_synthetic

        self.shock_proba = shock_proba
        self.shock_size = shock_size
        self.modus = modus

        self.snr = snr
        self.non_additive_noise_proba = non_additive_noise_proba
        self.autoregressive_innovation = autoregressive_innovation
        self.non_gaussian_additive = non_gaussian_additive
        self.non_equal_variance_range = non_equal_variance_range
        self.which_non_gaussian = which_non_gaussian
        self.scale_for_time_dependent = scale_for_time_dependent
        self.initialized_non_equal_variance = None
        self.cyle_timer = 0
        self.last_noise = None
        self.semi_synthetic_noise_path = semi_synthetic_noise_path
        # I think there is a better way to init this here.
        if time_dependent_cycle == "annual_sin":
            self.time_dependent_cycle = np.vectorize(sinus_year_cycle)
        elif time_dependent_cycle == "increase_per_step":
            self.time_dependent_cycle = np.vectorize(scale_up_through_time)
        elif time_dependent_cycle == "mean_var_shift":
            self.time_dependent_cycle = np.vectorize(mean_var_shift)
        elif time_dependent_cycle:
            self.time_dependent_cycle = np.vectorize(time_dependent_cycle)
        else:
            self.time_dependent_cycle = None
            
        if self.semi_synthetic:
            if self.modus == "obs":
                self.noise_base = [
                    np.load(f"{self.semi_synthetic_noise_path}/s_p.npy"),
                    np.load(f"{self.semi_synthetic_noise_path}/gold.npy"),
                    np.load(f"{self.semi_synthetic_noise_path}/nvidia.npy")
                    ]
            else:
                self.river_samples = pd.read_csv(f"{self.semi_synthetic_noise_path}/rivers_ts_flood.csv", index_col=0)
                self.river_samples = (self.river_samples - self.river_samples.min()) / (self.river_samples.max() - self.river_samples.min())
                self.river_samples = self.river_samples.interpolate().bfill().T.values
                self.semi_synthetic_current_index = self.rng.integers(0, self.river_samples.shape[1]-1, size=1)
                self.selected_river_indices = None

    def reset(self):
        self.cyle_timer = 0
        self.last_noise = None
        self.initialized_non_equal_variance = None
        if self.semi_synthetic and self.modus == "inno":
            self.semi_synthetic_current_index = self.rng.integers(0, self.river_samples.shape[1]-1, size=1)
            self.selected_river_indices = None
            

    def invNormal(self, low, high, mu=0, sd=1, *, size=1, block_size=1024):
        remain = size
        result = []

        mul = -0.5 * sd**-2

        while remain:
            # draw next block of uniform variates within interval
            x = self.rng.uniform(low, high, size=min((remain + 5) * 2, block_size))
            # reject proportional to normal density
            x = x[self.rng.exp(mul * (x - mu) ** 2) < self.rng.rand(*x.shape)]
            # make sure we don't add too much
            if remain < len(x):
                x = x[:remain]

            result.append(x)
            remain -= len(x)

        return np.concatenate(result)

    def draw_ng_sample(self, n):
        """
        Draws a sample consisting of 2 weighted distributions and scales it to (0,1)

        """
        a = self.rng.normal(0, 1, size=n.shape)

        if self.which_non_gaussian == "weibull":
            b = self.rng.weibull(1.5, n.shape)
            mean_ng, var_ng = 0.9027, 0.3757

        elif self.which_non_gaussian == "uniform":
            b = self.rng.uniform(-1, 1, n.shape)
            mean_ng, var_ng = 0, 4/12
        elif self.which_non_gaussian == "inv_normal":
            b = self.invNormal(low=-3, high=3, size=5)
            mean_ng, var_ng = 0, 4.447105693168032
            raise NotImplementedError("Inverse normal sampling not properly implemented.")

        elif self.which_non_gaussian == "exponential":
            b = self.rng.exponential(1,n.shape )
            mean_ng, var_ng = 1, 1

        p = 1 - self.non_gaussian_additive
        up = (1 - p) * (b - mean_ng) + p * a
        down = np.sqrt(var_ng * (1 - 2 * p) + p**2 * (var_ng + 1))
        return up / down

    def additive_noise(self, current):
        if self.non_equal_variance_range: 
            if not isinstance(self.initialized_non_equal_variance, np.ndarray):
                
                    # Randomly select from nl_power1 and nl_power2 for each element
                    lower = self.rng.uniform(
                    self.non_equal_variance_range[0],
                    self.non_equal_variance_range[1],
                    current.shape[0]
                    )
                    upper = self.rng.uniform(
                    self.non_equal_variance_range[2],
                    self.non_equal_variance_range[3],
                    current.shape[0]
                    )
                    choose_first = self.rng.integers(0, 2, size=current.shape[0]).astype(bool)
                    self.initialized_non_equal_variance = np.where(choose_first, lower, upper)
                
                
            unique_n = []
            for x in range(current.shape[0]):
                unique_n.append(self.rng.normal(0, self.initialized_non_equal_variance[x], current.shape[1]))
            joint = np.stack(unique_n, axis=0)
            return joint
        return self.rng.normal(0, 1, current.shape)

    def multiplicative_noise(self, current):
        return self.rng.normal(0, 1, (current.shape)) * current

    def shock_noise(self, current):
        # TODO: might be better to use the magnitude of the last n ts steps to properly set the magnitude of the shock. Currently its fixed to the hp.
        shock = np.zeros(current.shape)
        shock[
            self.rng.uniform(low=0.0, high=1.0, size=current.shape) < self.shock_proba
        ] = self.shock_size
        return shock

    def time_dependent_noise(self, current):
        if self.time_dependent_cycle:
            base = self.rng.normal(0, 1, current.shape)
            cycle_steps = np.stack(
                [np.arange(self.cyle_timer, self.cyle_timer + current.shape[1])]
                * current.shape[0]
            )
            scaling = self.time_dependent_cycle(cycle_steps,self.scale_for_time_dependent)
            time_dependent = base * scaling
            self.cyle_timer += 1
            return time_dependent
        else:
            return np.zeros(current)

    def autoregressive_noise(self, current):

        # different process for obs and inno.
        if self.modus == "inno": 
            if (
                isinstance(self.last_noise, np.ndarray)
                and self.last_noise.shape == current.shape
            ):
                return self.last_noise
            else:
                return np.zeros(current.shape)
        else:  # obs_n
            out = np.zeros(current.shape)
            out[:,0] = self.rng.normal(0, 1, current.shape[0])
            for x in range(1, out.shape[1]):
                out[:,x] = self.autoregressive_innovation * out[:,x - 1] + (
                    1 - self.autoregressive_innovation
                ) * self.rng.normal(0, 1, current.shape[0])
            return out

    def common_noise(self, current):
        return np.stack(
            [self.rng.normal(0, 1, current.shape[1])] * current.shape[0], axis=0
        )
        
    def semi_synthetic_noise(self, current):
        # Here we use some real world noise previously prepared
        if self.modus == "obs":
            stack = []
            d = current.shape
            for x in range (d[0]):
                # Draw a index from the base noise list:
                ind = self.rng.integers(0, len(self.noise_base))
                start_idx = self.rng.integers(0, len(self.noise_base[ind]) - d[1])
                real_noise = self.noise_base[ind][start_idx:start_idx + d[1]]
                stack.append(real_noise)
            return np.stack(stack, axis=0)
        else:
            # draw from the selected river samples in sequence
            if self.selected_river_indices is None:
                self.selected_river_indices = self.rng.integers(0, self.river_samples.shape[0], size=current.shape[0])
            noise = self.river_samples[self.selected_river_indices, self.semi_synthetic_current_index]
            if (self.semi_synthetic_current_index+1) == self.river_samples.shape[1]:
                self.semi_synthetic_current_index = 0
            else:
                self.semi_synthetic_current_index += 1
            return np.expand_dims(noise, axis=1)

    def get_noise(self, ts_in):
        # TODO Currently only a single nonadditive noise type can be selected. Extend this for mixing in the future.

        """
        Takes in  a 2 dim matrix (n_vars, ts_length)
        """
        # validate specification:
        specsum = sum(
            [
                self.additive,
                self.multiplicative,
                self.time_dependent,
                self.autoregressive,
                self.common,
                self.shock,
            ]
        )
        assert specsum <= (
            1 if self.modus == "obs" else 2
        ), "THIS CASE ISNT IMPLEMENTED: " + str(specsum)
         
        
        if self.modus == "obs":
            if self.additive:
                base_noise = self.additive_noise(ts_in)
            if self.multiplicative:
                base_noise = self.multiplicative_noise(ts_in)
            if self.time_dependent:
                base_noise = self.time_dependent_noise(ts_in)
            if self.autoregressive:
                base_noise = self.autoregressive_noise(ts_in)
            if self.semi_synthetic:
                base_noise = self.semi_synthetic_noise(ts_in)
            if self.common:
                base_noise = self.common_noise(ts_in)
            if self.shock:
                base_noise = self.shock_noise(ts_in)

            # we calculate an alpha to rescale the base noise term to match the required SNR level.
            signal_power = np.mean(ts_in**2)
            target_noise_power = signal_power / self.snr
            base_noise_power = np.mean(base_noise**2)
            rescaling_factor = np.sqrt(target_noise_power / base_noise_power)
            final_noise = rescaling_factor * base_noise


            assert (
                signal_power / np.mean(final_noise**2) - self.snr
            ) < 0.01, "SNR scaling has a high error margin" + str(signal_power / np.mean(final_noise**2) - self.snr)
            self.last_noise = final_noise
            return final_noise

        if self.modus == "inno":
            if self.non_gaussian_additive > 0: 
                out_noise = self.draw_ng_sample(ts_in)
            else:
                out_noise = self.additive_noise(ts_in)
            if self.non_additive_noise_proba > 0:
                if self.multiplicative:
                    add_noise = self.multiplicative_noise(ts_in)
                elif self.time_dependent:
                    add_noise = self.time_dependent_noise(ts_in)
                elif self.autoregressive:
                    add_noise = self.autoregressive_noise(ts_in)
                elif self.common:
                    add_noise = self.common_noise(ts_in)
                elif self.shock:
                    add_noise = self.shock_noise(ts_in)
                elif self.semi_synthetic:
                    add_noise = self.semi_synthetic_noise(ts_in)
                else:
                    raise ValueError("No non-additive noise type selected.")
                    
                out_noise = add_noise * self.non_additive_noise_proba + (1-self.non_additive_noise_proba) * out_noise
                self.last_noise = out_noise
            return out_noise

        else:
            raise NotImplementedError
