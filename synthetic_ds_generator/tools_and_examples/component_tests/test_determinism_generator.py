#!/usr/bin/env python3
"""
Test Determinism of Synthetic Data Generator

This script tests the determinism of the SyntheticDataGenerator class.
The generator should produce identical results when initialized with the same seed
and component configurations.
"""

import sys
import os
import numpy as np
from typing import Dict, List

sys.path.append("../../")

# Import all required components
from components.noise_generator import NoiseGenerator
from components.lagged_effects import LaggedEffects
from components.instantanous_effects import InstantanousEffects
from components.exog_influences import ExogenousInfluences
from components.nl_sampler import NL_function_generator
from data_generator import SyntheticDataGenerator


class SyntheticDataGeneratorDeterminismTester:
    """Test determinism of the SyntheticDataGenerator."""
    
    def __init__(self, test_seed: int = 42, n_tests: int = 3):
        """
        Initialize the determinism tester for SyntheticDataGenerator.
        
        Args:
            test_seed: Seed to use for testing determinism
            n_tests: Number of independent tests to run
        """
        self.test_seed = test_seed
        self.n_tests = n_tests
        self.results = {}
        
    def create_components(self, seed: int, n_vars: int = 4, n_exogs: int = 3, max_lags: int = 3):
        """
        Create all required components for SyntheticDataGenerator with given seed.
        
        Args:
            seed: Random seed for component initialization
            n_vars: Number of variables
            n_exogs: Number of exogenous variables
            max_lags: Maximum lags for structural equation
            
        Returns:
            Dictionary containing all initialized components
        """
        
        # Create NL sampler
        nl_sampler = NL_function_generator(
            rng=seed,
            nl_mode="power_set",
            power_dist=[0.5, 1.0, 1.5, 2.0],
            which_power_dist="all"
        )
        
        # Create all required components
        inno_n = NoiseGenerator(
            rng=seed,
            modus="inno",
            additive=True,
            autoregressive=True
        )
        
        obs_n = NoiseGenerator(
            rng=seed,
            modus="obs",
            additive=True,
            snr=5.0
        )
        
        lagged = LaggedEffects(
            rng=seed,
            n_vars=n_vars,
            max_lags=max_lags,
            link_proba=0.15,
            nonlinear_proba=0.2,
            nl_sampler=nl_sampler
        )
        
        instant = InstantanousEffects(
            rng=seed,
            n_vars=n_vars,
            link_proba=0.1,
            nonlinear_proba=0.3,
            nl_sampler=nl_sampler
        )
        
        exog = ExogenousInfluences(
            rng=seed,
            n_vars=n_vars,
            n_exogs=n_exogs,
            link_proba=0.0,
            nonlinear_proba=0.1,
            nl_sampler=nl_sampler
        )
        
        return {
            'inno_n': inno_n,
            'obs_n': obs_n,
            'lagged': lagged,
            'instant': instant,
            'exog': exog,
            'rng': seed
        }
    
    def test_basic_determinism(self) -> bool:
        """Test basic determinism without change points."""
        print("\n=== Testing Basic SyntheticDataGenerator Determinism ===")
        
        try:
            generators = []
            
            # Create multiple generator instances with same seed
            for i in range(self.n_tests):
                components = self.create_components(self.test_seed)
                
                generator = SyntheticDataGenerator(
                    inno_n=components['inno_n'],
                    obs_n=components['obs_n'],
                    exog=components['exog'],
                    lagged=components['lagged'],
                    instant=components['instant'],
                    rng=components['rng'],
                    time_series_n=200,
                    normalize=None
                )
                generators.append(generator)
            
            # Generate samples from each instance
            results = []
            for i, generator in enumerate(generators):
                print(f"Generating sample {i+1}/{len(generators)}...")
                sample = generator.get_sample()
                results.append(sample)
            
            # Compare results
            return self._compare_generator_results(results, "Basic generation")
            
        except Exception as e:
            print(f"Error in basic determinism test: {e}")
            return False
    
    def test_determinism_with_change_points(self) -> bool:
        """Test determinism with change points."""
        print("\n=== Testing SyntheticDataGenerator Determinism with Change Points ===")
        
        try:
            generators = []
            change_points = [100, 200, 300]
            
            # Create multiple generator instances with same seed
            for i in range(self.n_tests):
                components = self.create_components(self.test_seed)
                
                generator = SyntheticDataGenerator(
                    inno_n=components['inno_n'],
                    obs_n=components['obs_n'],
                    exog=components['exog'],
                    lagged=components['lagged'],
                    instant=components['instant'],
                    rng=components['rng'],
                    time_series_n=400,
                    change_points=change_points,
                    nonstationary=True,
                    normalize=None
                )
                generators.append(generator)
            
            # Generate samples from each instance
            results = []
            for i, generator in enumerate(generators):
                print(f"Generating sample with change points {i+1}/{len(generators)}...")
                sample = generator.get_sample()
                results.append(sample)
            
            # Compare results
            return self._compare_generator_results(results, "Change points generation")
            
        except Exception as e:
            print(f"Error in change points determinism test: {e}")
            return False
    
    def test_determinism_with_structure_dropping(self) -> bool:
        """Test determinism with structure dropping."""
        print("\n=== Testing SyntheticDataGenerator Determinism with Structure Dropping ===")
        
        try:
            generators = []
            change_points = [150, 300]
            
            # Create multiple generator instances with same seed
            for i in range(self.n_tests):
                components = self.create_components(self.test_seed)
                
                generator = SyntheticDataGenerator(
                    inno_n=components['inno_n'],
                    obs_n=components['obs_n'],
                    exog=components['exog'],
                    lagged=components['lagged'],
                    instant=components['instant'],
                    rng=components['rng'],
                    time_series_n=450,
                    change_points=change_points,
                    drop_struc_for_window=True,
                    nonstationary=True,
                    normalize=None
                )
                generators.append(generator)
            
            # Generate samples from each instance
            results = []
            for i, generator in enumerate(generators):
                print(f"Generating sample with structure dropping {i+1}/{len(generators)}...")
                sample = generator.get_sample()
                results.append(sample)
            
            # Compare results
            return self._compare_generator_results(results, "Structure dropping generation")
            
        except Exception as e:
            print(f"Error in structure dropping determinism test: {e}")
            return False
    
    def test_determinism_with_normalization(self) -> bool:
        """Test determinism with different normalization methods."""
        print("\n=== Testing SyntheticDataGenerator Determinism with Normalization ===")
        
        normalization_methods = [None, "minmax", "standardization"]
        all_passed = True
        
        for norm_method in normalization_methods:
            print(f"\nTesting normalization: {norm_method}")
            
            try:
                generators = []
                
                # Create multiple generator instances with same seed
                for i in range(self.n_tests):
                    components = self.create_components(self.test_seed)
                    
                    generator = SyntheticDataGenerator(
                        inno_n=components['inno_n'],
                        obs_n=components['obs_n'],
                        exog=components['exog'],
                        lagged=components['lagged'],
                        instant=components['instant'],
                        rng=components['rng'],
                        time_series_n=200,
                        normalize=norm_method
                    )
                    generators.append(generator)
                
                # Generate samples from each instance
                results = []
                for i, generator in enumerate(generators):
                    sample = generator.get_sample()
                    results.append(sample)
                
                # Compare results
                passed = self._compare_generator_results(results, f"Normalization ({norm_method})")
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                print(f"Error in normalization determinism test ({norm_method}): {e}")
                all_passed = False
        
        return all_passed
    
    def test_determinism_with_missing_data(self) -> bool:
        """Test determinism with missing data interpolation."""
        print("\n=== Testing SyntheticDataGenerator Determinism with Missing Data ===")
        
        try:
            generators = []
            
            # Create multiple generator instances with same seed
            for i in range(self.n_tests):
                components = self.create_components(self.test_seed)
                
                generator = SyntheticDataGenerator(
                    inno_n=components['inno_n'],
                    obs_n=components['obs_n'],
                    exog=components['exog'],
                    lagged=components['lagged'],
                    instant=components['instant'],
                    rng=components['rng'],
                    time_series_n=200,
                    interpolate=0.1,  # 10% missing data
                    normalize=None
                )
                generators.append(generator)
            
            # Generate samples from each instance
            results = []
            for i, generator in enumerate(generators):
                print(f"Generating sample with missing data {i+1}/{len(generators)}...")
                sample = generator.get_sample()
                results.append(sample)
            
            # Compare results
            return self._compare_generator_results(results, "Missing data interpolation")
            
        except Exception as e:
            print(f"Error in missing data determinism test: {e}")
            return False
    
    def test_determinism_with_confounding(self) -> bool:
        """Test determinism with variable removal for confounding."""
        print("\n=== Testing SyntheticDataGenerator Determinism with Confounding ===")
        
        try:
            generators = []
            
            # Create multiple generator instances with same seed
            for i in range(self.n_tests):
                components = self.create_components(self.test_seed, n_vars=5)  # More variables for confounding
                
                generator = SyntheticDataGenerator(
                    inno_n=components['inno_n'],
                    obs_n=components['obs_n'],
                    exog=components['exog'],
                    lagged=components['lagged'],
                    instant=components['instant'],
                    rng=components['rng'],
                    time_series_n=200,
                    remove_n_variables_for_confounding=1,
                    normalize=None
                )
                generators.append(generator)
            
            # Generate samples from each instance
            results = []
            for i, generator in enumerate(generators):
                print(f"Generating sample with confounding {i+1}/{len(generators)}...")
                sample = generator.get_sample()
                results.append(sample)
            
            # Compare results
            return self._compare_generator_results(results, "Confounding simulation")
            
        except Exception as e:
            print(f"Error in confounding determinism test: {e}")
            return False
    
    def test_different_seeds_produce_different_results(self) -> bool:
        """Test that different seeds produce different results."""
        print("\n=== Testing Different Seeds Produce Different Results ===")
        
        try:
            test_seeds = [42, 123, 456, 789]
            results = []
            
            # Generate samples with different seeds
            for seed in test_seeds:
                components = self.create_components(seed)
                
                generator = SyntheticDataGenerator(
                    inno_n=components['inno_n'],
                    obs_n=components['obs_n'],
                    exog=components['exog'],
                    lagged=components['lagged'],
                    instant=components['instant'],
                    rng=components['rng'],
                    time_series_n=200,
                    normalize=None
                )
                
                print(f"Generating sample with seed {seed}...")
                sample = generator.get_sample()
                results.append((seed, sample))
            
            # Compare results - they should be different
            return self._compare_different_seed_results(results)
            
        except Exception as e:
            print(f"Error in different seeds test: {e}")
            return False
    
    def test_different_seeds_with_change_points(self) -> bool:
        """Test that different seeds produce different results with change points."""
        print("\n=== Testing Different Seeds with Change Points ===")
        
        try:
            test_seeds = [42, 987, 555]
            change_points = [100, 200]
            results = []
            
            # Generate samples with different seeds
            for seed in test_seeds:
                components = self.create_components(seed)
                
                generator = SyntheticDataGenerator(
                    inno_n=components['inno_n'],
                    obs_n=components['obs_n'],
                    exog=components['exog'],
                    lagged=components['lagged'],
                    instant=components['instant'],
                    rng=components['rng'],
                    time_series_n=300,
                    change_points=change_points,
                    nonstationary=True,
                    normalize=None
                )
                
                print(f"Generating sample with seed {seed} and change points...")
                sample = generator.get_sample()
                results.append((seed, sample))
            
            # Compare results - they should be different
            return self._compare_different_seed_results(results)
            
        except Exception as e:
            print(f"Error in different seeds with change points test: {e}")
            return False

    def _compare_generator_results(self, results: List, test_name: str) -> bool:
        """
        Compare results from multiple generator instances.
        
        Args:
            results: List of generator results (tuples)
            test_name: Name of the test for reporting
            
        Returns:
            True if all results are identical, False otherwise
        """
        if len(results) < 2:
            print(f"Not enough results to compare for {test_name}")
            return False
        
        reference = results[0]
        
        for i in range(1, len(results)):
            current = results[i]
            # Compare time series data (first element)
            if not np.allclose(reference[0], current[0], rtol=1e-10, atol=1e-10):
                print(f"✗ {test_name}: Time series data differs between instance 0 and {i}")
                print(f"  Max difference: {np.max(np.abs(reference[0] - current[0]))}")
                return False
            
            # Compare lagged links (second element)
            if not np.allclose(reference[1], current[1], rtol=1e-10, atol=1e-10):
                print(f"✗ {test_name}: Lagged links differ between instance 0 and {i}")
                return False
            
            # Compare instantaneous links (third element)
            if not np.allclose(reference[2], current[2], rtol=1e-10, atol=1e-10):
                print(f"✗ {test_name}: Instantaneous links differ between instance 0 and {i}")
                return False
     
            # Compare nonlinear structures (sixth element)
            if not np.array_equal(reference[5], current[5]):
                print(f"✗ {test_name}: Nonlinear structures differ between instance 0 and {i}")
                return False
        
        print(f"✓ {test_name}: All instances produce identical results")
        return True

    def _compare_different_seed_results(self, results: List) -> bool:
        """
        Compare results from generators with different seeds to ensure they differ.
        
        Args:
            results: List of (seed, generator_result) tuples
            
        Returns:
            True if results are sufficiently different, False otherwise
        """
        if len(results) < 2:
            print("Not enough results to compare for different seeds test")
            return False
        
        # Compare each pair of results
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                seed_i, result_i = results[i]
                seed_j, result_j = results[j]
                
                # Check if time series data is different
                if np.allclose(result_i[0], result_j[0], rtol=1e-6, atol=1e-6):
                    print(f"✗ Different seeds test: Seeds {seed_i} and {seed_j} produce similar time series data")
                    print(f"  Max difference: {np.max(np.abs(result_i[0] - result_j[0]))}")
                    return False
                
                # Check if structural components are different
                if np.allclose(result_i[1], result_j[1], rtol=1e-6, atol=1e-6) and (result_i[1].sum() > 0) :
                    print(f"✗ Different seeds test: Seeds {seed_i} and {seed_j} produce similar lagged structures")
                if np.allclose(result_i[2], result_j[2], rtol=1e-6, atol=1e-6) and (result_i[2].sum() > 0):

                    print(f"✗ Different seeds test: Seeds {seed_i} and {seed_j} produce similar instantaneous structures")
                    return False
        
        print("✓ Different seeds test: All seed pairs produce sufficiently different results")
        return True
            
        # except Exception as e:
        #     print(f"✗ {test_name}: Error during comparison: {e}")
        #     return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all determinism tests for SyntheticDataGenerator."""
        print("=" * 70)
        print("DETERMINISM TESTING FOR SYNTHETIC DATA GENERATOR")
        print("=" * 70)
        print(f"Test seed: {self.test_seed}")
        print(f"Number of test instances per test: {self.n_tests}")
        
        # Define all tests
        tests = [
            ("Basic Generation", self.test_basic_determinism),
            ("Change Points", self.test_determinism_with_change_points),
            ("Structure Dropping", self.test_determinism_with_structure_dropping),
            ("Normalization", self.test_determinism_with_normalization),
            ("Missing Data", self.test_determinism_with_missing_data),
            ("Confounding", self.test_determinism_with_confounding),
            ("Different Seeds", self.test_different_seeds_produce_different_results),
            ("Different Seeds with Change Points", self.test_different_seeds_with_change_points)
        ]
        
        # Run all tests
        all_passed = True
        for test_name, test_func in tests:
            try:
                print(f"\n{'-' * 50}")
                print(f"Running: {test_name}")
                print(f"{'-' * 50}")
                
                passed = test_func()
                self.results[test_name] = {
                    'passed': passed,
                    'error': None
                }
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                error_msg = f"Error in {test_name}: {str(e)}"
                print(f"✗ {error_msg}")
                self.results[test_name] = {
                    'passed': False,
                    'error': error_msg
                }
                all_passed = False
        
        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        for test_name, result in self.results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"{test_name:25} {status}")
            if result['error']:
                print(f"{'':25} Error: {result['error']}")
        
        overall_status = "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
        print(f"\nOverall result: {overall_status}")
        
        return self.results


def main():
    """Main function to run the SyntheticDataGenerator determinism tests."""
    tester = SyntheticDataGeneratorDeterminismTester(test_seed=42, n_tests=3)
    results = tester.run_all_tests()
    
    # Exit with error code if any tests failed
    if not all(result['passed'] for result in results.values()):
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
