#!/usr/bin/env python3
"""
Test Component-Level Seeding Independence

This script tests that each component can be seeded independently, meaning:
1. When only one component's seed changes, only that component's output changes
2. All other components remain deterministic and unchanged
3. The overall system maintains determinism while allowing selective component variation

This is crucial for controlled experiments where you want to vary specific aspects
of the data generation process while keeping others constant.
"""

import sys
import numpy as np
from typing import Any, Dict, List

sys.path.append("../../")  # Adjust path to import from synthetic_ds_generator

# Import all components
from components.noise_generator import NoiseGenerator
from components.lagged_effects import LaggedEffects
from components.instantanous_effects import InstantanousEffects
from components.exog_influences import ExogenousInfluences
from components.nl_sampler import NL_function_generator
from data_generator import SyntheticDataGenerator


class ComponentSeedingTester:
    """Test that components can be seeded independently."""

    def __init__(
        self, base_seed: int = 42, alt_seed: int = 123, tolerance: float = 1e-10
    ):
        """
        Initialize the component seeding tester.

        Args:
            base_seed: Base seed used for all components in baseline
            alt_seed: Alternative seed used when testing specific component changes
            tolerance: Numerical tolerance for floating-point comparisons
        """
        self.base_seed = base_seed
        self.alt_seed = alt_seed
        self.tolerance = tolerance
        self.results = {}

    def create_baseline_generator(self, **kwargs) -> SyntheticDataGenerator:
        """Create a baseline generator with all components using base_seed."""
        default_config = {
            "n_vars": 4,
            "max_lags": 3,
            "n_exogs": 2,
            "time_series_n": 100,
        }
        default_config.update(kwargs)

        # Create components with base seed
        inno_n = NoiseGenerator(
            rng=self.base_seed,
            modus="inno",
            additive=True,
            non_additive_noise_proba=0.0,
        )

        obs_n = NoiseGenerator(rng=self.base_seed, modus="obs", additive=True, snr=2.0)

        nl_sampler = NL_function_generator(
            rng=self.base_seed, nl_mode="power_set", n_options=5
        )

        lagged = LaggedEffects(
            rng=self.base_seed,
            n_vars=default_config["n_vars"],
            max_lags=default_config["max_lags"],
            link_proba=0.3,
            param_range=[0.2, 0.4],
            nonlinear_proba=0.5,
            nl_sampler=nl_sampler,
        )

        nl_sampler = NL_function_generator(
            rng=self.base_seed, nl_mode="power_set", n_options=5
        )

        instant = InstantanousEffects(
            rng=self.base_seed,
            n_vars=default_config["n_vars"],
            link_proba=0.5,
            param_range=[0.2, 0.4],
            nonlinear_proba=0.5,
            nl_sampler=nl_sampler,
        )

        nl_sampler = NL_function_generator(
            rng=self.base_seed, nl_mode="power_set", n_options=5
        )

        exog = ExogenousInfluences(
            rng=self.base_seed,
            n_vars=default_config["n_vars"],
            n_exogs=default_config["n_exogs"],
            link_proba=0.4,
            param_range=[0.2, 0.4],
            nonlinear_proba=0.5,
            nl_sampler=nl_sampler,
        )

        generator = SyntheticDataGenerator(
            rng=self.base_seed,
            time_series_n=default_config["time_series_n"],
            normalize=None,
            inno_n=inno_n,
            obs_n=obs_n,
            exog=exog,
            lagged=lagged,
            instant=instant,
        )

        return generator

    def create_variant_generator(
        self, component_to_change: str, **kwargs
    ) -> SyntheticDataGenerator:
        """Create a generator with only one component using alternative seed."""
        default_config = {
            "n_vars": 4,
            "max_lags": 3,
            "n_exogs": 2,
            "time_series_n": 100,
        }
        default_config.update(kwargs)

        # Create components, using alt_seed only for the specified component
        inno_seed = self.alt_seed if component_to_change == "inno_n" else self.base_seed
        inno_n = NoiseGenerator(
            rng=inno_seed, modus="inno", additive=True, non_additive_noise_proba=0.0
        )

        obs_seed = self.alt_seed if component_to_change == "obs_n" else self.base_seed
        obs_n = NoiseGenerator(rng=obs_seed, modus="obs", additive=True, snr=2.0)

        nl_sampler_seed = (
            self.alt_seed
            if (component_to_change == "lagged" or component_to_change == "nl_sampler")
            else self.base_seed
        )
        nl_sampler = NL_function_generator(
            rng=nl_sampler_seed, nl_mode="power_set", n_options=5
        )

        lagged_seed = (
            self.alt_seed if component_to_change == "lagged" else self.base_seed
        )
        lagged = LaggedEffects(
            rng=lagged_seed,
            n_vars=default_config["n_vars"],
            max_lags=default_config["max_lags"],
            link_proba=0.3,
            param_range=[0.2, 0.4],
            nonlinear_proba=0.5,
            nl_sampler=nl_sampler,
        )

        nl_sampler_seed = (
            self.alt_seed if (component_to_change == "instant" or component_to_change == "nl_sampler")
            else self.base_seed
        )
        nl_sampler = NL_function_generator(
            rng=nl_sampler_seed, nl_mode="power_set", n_options=5
        )

        instant_seed = (
            self.alt_seed if component_to_change == "instant" else self.base_seed
        )
        instant = InstantanousEffects(
            rng=instant_seed,
            n_vars=default_config["n_vars"],
            link_proba=0.5,
            param_range=[0.2, 0.4],
            nonlinear_proba=0.5,
            nl_sampler=nl_sampler,
        )

        nl_sampler_seed = (
            self.alt_seed if (component_to_change == "exog" or component_to_change == "nl_sampler")
            else self.base_seed
        )
        nl_sampler = NL_function_generator(
            rng=nl_sampler_seed, nl_mode="power_set", n_options=5
        )

        exog_seed = self.alt_seed if component_to_change == "exog" else self.base_seed
        exog = ExogenousInfluences(
            rng=exog_seed,
            n_vars=default_config["n_vars"],
            n_exogs=default_config["n_exogs"],
            link_proba=0.4,
            param_range=[0.2, 0.4],
            nonlinear_proba=0.5,
            nl_sampler=nl_sampler,
        )

        generator = SyntheticDataGenerator(
            rng=self.base_seed,  # Main generator always uses base seed
            time_series_n=default_config["time_series_n"],
            normalize=None,
            inno_n=inno_n,
            obs_n=obs_n,
            exog=exog,
            lagged=lagged,
            instant=instant,
        )

        return generator

    def compare_generator_outputs(
        self, gen1: SyntheticDataGenerator, gen2: SyntheticDataGenerator
    ) -> Dict[str, bool]:
        """Compare outputs from two generators and return which components differ."""
        # Generate data from both generators
        result1 = gen1.get_sample()
        result2 = gen2.get_sample()

        # Extract outputs
        (
            ts1,
            links1,
            instant1,
            exog_links1,
            exog_ts1,
            nl_names1,
            inst_nl_names1,
            exog_nl_names1,
            attempts1,
        ) = result1
        (
            ts2,
            links2,
            instant2,
            exog_links2,
            exog_ts2,
            nl_names2,
            inst_nl_names2,
            exog_nl_names2,
            attempts2,
        ) = result2

        comparison = {}

        # Compare time series (overall output)
        comparison["time_series"] = np.allclose(
            ts1, ts2, rtol=self.tolerance, atol=self.tolerance
        )

        # Compare lagged links (lagged effects)
        comparison["lagged_links"] = np.allclose(
            links1, links2, rtol=self.tolerance, atol=self.tolerance
        )

        # Compare instantaneous links
        comparison["instantaneous_links"] = np.allclose(
            instant1, instant2, rtol=self.tolerance, atol=self.tolerance
        )

        # Compare exogenous links and time series
        if exog_links1 is not None and exog_links2 is not None:
            comparison["exogenous_links"] = np.allclose(
                exog_links1, exog_links2, rtol=self.tolerance, atol=self.tolerance
            )
            comparison["exogenous_series"] = np.allclose(
                exog_ts1, exog_ts2, rtol=self.tolerance, atol=self.tolerance
            )
        else:
            comparison["exogenous_links"] = (exog_links1 is None) and (
                exog_links2 is None
            )
            comparison["exogenous_series"] = (exog_ts1 is None) and (exog_ts2 is None)

        # Compare nonlinear function specifications
        comparison["nl_functions"] = self._compare_nl_specs(nl_names1, nl_names2)
        comparison["instant_nl_functions"] = self._compare_nl_specs(
            inst_nl_names1, inst_nl_names2
        )
        comparison["exog_nl_functions"] = self._compare_nl_specs(
            exog_nl_names1, exog_nl_names2
        )

        return comparison

    def _compare_nl_specs(self, spec1, spec2) -> bool:
        """Compare nonlinear function specifications."""
        if spec1 is None and spec2 is None:
            return True
        if spec1 is None or spec2 is None:
            return False

        # For complex nested structures, convert to strings and compare
        return str(spec1) == str(spec2)

    def test_component_independence(self, component_name: str) -> Dict[str, Any]:
        """
        Test that changing only one component's seed affects only that component.

        Args:
            component_name: Name of component to change ('inno_n', 'obs_n', 'lagged', 'instant', 'exog', 'nl_sampler')

        Returns:
            Dictionary with test results and detailed comparison
        """
        print(f"\n=== Testing {component_name} Independence ===")

        try:
            # Create baseline and variant generators
            baseline_gen = self.create_baseline_generator()
            variant_gen = self.create_variant_generator(component_name)

            # Compare outputs
            comparison = self.compare_generator_outputs(baseline_gen, variant_gen)

            # Analyze results based on which component was changed
            expected_changes = self._get_expected_changes(component_name)

            test_passed = True
            detailed_results = {}

            for output_type, is_same in comparison.items():
                should_be_different = output_type in expected_changes

                if should_be_different:
                    # This output should be different
                    if is_same:
                        print(
                            f"  ❌ {output_type}: Expected difference but outputs are identical"
                        )
                        test_passed = False
                    else:
                        print(f"  ✅ {output_type}: Correctly different")
                else:
                    # This output should be the same
                    if not is_same:
                        print(f"  ❌ {output_type}: Expected same but outputs differ")
                        test_passed = False
                    else:
                        print(f"  ✅ {output_type}: Correctly identical")

                detailed_results[output_type] = {
                    "is_same": is_same,
                    "expected_different": should_be_different,
                    "correct": is_same != should_be_different,
                }

            result = {
                "component": component_name,
                "passed": test_passed,
                "detailed_results": detailed_results,
                "error": None,
            }

            status = "✅ PASS" if test_passed else "❌ FAIL"
            print(f"Result: {status}")

            return result

        except Exception as e:
            error_msg = f"Error testing {component_name}: {str(e)}"
            print(f"Result: ❌ ERROR - {error_msg}")
            return {
                "component": component_name,
                "passed": False,
                "detailed_results": {},
                "error": error_msg,
            }

    def _get_expected_changes(self, component_name: str) -> List[str]:
        """Get list of outputs that should change when a specific component changes."""
        if component_name == "inno_n":
            # Innovation noise affects the time series
            return ["time_series"]
        elif component_name == "obs_n":
            # Observation noise affects the final time series
            return ["time_series"]
        elif component_name == "lagged":
            # Lagged effects affect lagged links and time series
            return ["lagged_links", "time_series", "nl_functions"]
        elif component_name == "instant":
            # Instantaneous effects affect instant links and time series
            return ["instantaneous_links", "time_series", "instant_nl_functions"]
        elif component_name == "exog":
            # Exogenous component affects exog links, exog series, and main time series
            return [
                "exogenous_links",
                "exogenous_series",
                "time_series",
                "exog_nl_functions",
            ]
        elif component_name == "nl_sampler":
            # NL sampler affects nonlinear function specs and potentially time series
            return [
                "nl_functions",
                "instant_nl_functions",
                "exog_nl_functions",
                "time_series",
            ]
        else:
            return []

    def run_all_tests(self) -> Dict[str, Any]:
        """Run independence tests for all components."""
        print("=" * 70)
        print("COMPONENT SEEDING INDEPENDENCE TESTS")
        print("=" * 70)
        print(f"Base seed: {self.base_seed}, Alternative seed: {self.alt_seed}")

        # Test each component
        components_to_test = [
            "inno_n",
            "obs_n",
            "lagged",
            "instant",
            "exog",
            "nl_sampler",
        ]

        all_passed = True
        for component_name in components_to_test:
            result = self.test_component_independence(component_name)
            self.results[component_name] = result
            if not result["passed"]:
                all_passed = False

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        for component_name, result in self.results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"{component_name:20} {status}")
            if result["error"]:
                print(f"{'':20} Error: {result['error']}")

        overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        print(f"\nOverall result: {overall_status}")

        return self.results


def main():
    """Main function to run the component seeding independence tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test component seeding independence")
    parser.add_argument(
        "--base-seed", type=int, default=42, help="Base seed for baseline"
    )
    parser.add_argument(
        "--alt-seed", type=int, default=123, help="Alternative seed for testing"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-10, help="Numerical tolerance"
    )
    parser.add_argument(
        "--component",
        type=str,
        default=None,
        choices=["inno_n", "obs_n", "lagged", "instant", "exog", "nl_sampler"],
        help="Test only specific component (default: test all)",
    )

    args = parser.parse_args()

    tester = ComponentSeedingTester(
        base_seed=args.base_seed, alt_seed=args.alt_seed, tolerance=args.tolerance
    )

    if args.component:
        # Test only specific component
        result = tester.test_component_independence(args.component)
        success = result["passed"]
    else:
        # Test all components
        results = tester.run_all_tests()
        success = all(result["passed"] for result in results.values())

    # Exit with error code if any tests failed
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
