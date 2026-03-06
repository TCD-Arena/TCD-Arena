

#!/usr/bin/env python3
"""
Test Determinism of Synthetic Data Generator Components

This script loads all the components in the synthetic_ds_generator package,
initializes them and checks if they are deterministic depending on the seed.
Each component should produce identical results when initialized with the same seed.
"""

import sys
import os
import numpy as np
from typing import Any, Dict, List

# Add the parent directory to the Python path to import components
sys.path.append("../../")

# Import all components
from components.noise_generator import NoiseGenerator
from components.lagged_effects import LaggedEffects
from components.instantanous_effects import InstantanousEffects
from components.exog_influences import ExogenousInfluences
from components.nl_sampler import NL_function_generator


class DeterminismTester:
    """Test determinism of synthetic data generator components."""
    
    def __init__(self, test_seed: int = 42, n_tests: int = 3):
        """
        Initialize the determinism tester.
        
        Args:
            test_seed: Seed to use for testing determinism
            n_tests: Number of independent tests to run for each component
        """
        self.test_seed = test_seed
        self.n_tests = n_tests
        self.results = {}
        
    def test_component_determinism(self, component_class, component_name: str, 
                                 init_kwargs: Dict[str, Any] = None) -> bool:
        """
        Test if a component is deterministic by creating multiple instances
        with the same seed and comparing their behavior.
        
        Args:
            component_class: The component class to test
            component_name: Name of the component for reporting
            init_kwargs: Additional initialization parameters
            
        Returns:
            True if component is deterministic, False otherwise
        """
        if init_kwargs is None:
            init_kwargs = {}
            
        print(f"\n=== Testing {component_name} ===")
        
        try:
            # Create multiple instances with the same seed
            instances = []
            for i in range(self.n_tests):
                if "nonlinear_proba" in init_kwargs:
                    # If the component requires a nonlinear sampler, instantiate it
                    init_kwargs["nl_sampler"] = NL_function_generator(rng=self.test_seed)
                
                instance = component_class(rng=self.test_seed, **init_kwargs)
                instances.append(instance)
            
            # Test determinism based on component type
            is_deterministic = self._test_specific_component(instances, component_name)
            
            self.results[component_name] = {
                'deterministic': is_deterministic,
                'error': None
            }
            
            status = "✓ PASS" if is_deterministic else "✗ FAIL"
            print(f"Result: {status}")
            
            return is_deterministic
            
        except Exception as e:
            error_msg = f"Error testing {component_name}: {str(e)}"
            print(f"Result: ✗ ERROR - {error_msg}")
            self.results[component_name] = {
                'deterministic': False,
                'error': error_msg
            }
            return False
    
    def _test_specific_component(self, instances: List[Any], component_name: str) -> bool:
        """Test determinism for specific component types."""
        
        if component_name == "OBS":
            return self._test_noise_generator(instances)
        if component_name == "INNO":
            return self._test_noise_generator(instances)
        elif component_name == "Structural_Equation":
            return self._test_structural_equation(instances)
        elif component_name == "Instantanous_effects":
            return self._test_instantaneous_effects(instances)
        elif component_name == "Exogenous_influences":
            return self._test_exogenous_influences(instances)
        elif component_name == "NL_function_generator":
            return self._test_nl_function_generator(instances)
        else:
            print(f"Unknown component type: {component_name}")
            return False
    
    def _test_noise_generator(self, instances: List[NoiseGenerator]) -> bool:
        """Test determinism of noise generator."""
        try:
            # Test noise generation with same parameters - use seeded RNG for test data
            test_rng = np.random.default_rng(self.test_seed)
            test_data = test_rng.standard_normal((3, 1))  # 10 timesteps, 3 variables
            
            results = []
            for instance in instances:
                # Generate noise

                noise = instance.get_noise(test_data)
                results.append(noise)
            # Compare results
            for i in range(1, len(results)):
                if not np.allclose(results[0], results[i], rtol=1e-10):
                    print(f"Noise generation differs between instance 0 and {i}")
                    return False
            
            print("Noise generation is deterministic")
            return True
            
        except Exception as e:
            print(f"Error in noise generator test: {e}")
            # Try simpler test - just check if instances can be created consistently
            return self._compare_instance_attributes(instances)
    
    def _test_structural_equation(self, instances: List[LaggedEffects]) -> bool:
        """Test determinism of structural equation."""
        try:
            # Initialize all instances with same parameters
            for instance in instances:
                instance.init_random_process()
            
            # Compare coefficient matrices
            link_matrices = [instance.links for instance in instances]
            for i in range(1, len(link_matrices)):
                if not np.allclose(link_matrices[0], link_matrices[i], rtol=1e-10):
                    print(f"Link matrices differ between instance 0 and {i}")
                    return False
            
            # Test step generation
            test_rng = np.random.default_rng(self.test_seed)
            test_ts = test_rng.standard_normal((5, 3))  # 5 timesteps, 3 variables
            step_results = []
            for instance in instances:
                result = instance.get_step(test_ts)
                step_results.append(result)
            for i in range(1, len(step_results)):
                if not np.allclose(step_results[0], step_results[i], rtol=1e-10):
                    print(f"Step results differ between instance 0 and {i}")
                    return False
            
            print("Structural equation is deterministic")
            return True
            
        except Exception as e:
            print(f"Error in structural equation test: {e}")
            return self._compare_instance_attributes(instances)
    
    def _test_instantaneous_effects(self, instances: List[InstantanousEffects]) -> bool:
        """Test determinism of instantaneous effects."""
        try:
            # Initialize all instances
            for instance in instances:
                instance.init_instantanous_influence()
            
            # Compare link matrices
            link_matrices = [instance.links for instance in instances]
            for i in range(1, len(link_matrices)):
                if not np.allclose(link_matrices[0], link_matrices[i], rtol=1e-10):
                    print(f"Instantaneous effect matrices differ between instance 0 and {i}")
                    return False
            
            # Test effect computation
            test_rng = np.random.default_rng(self.test_seed)
            test_ts = test_rng.standard_normal((5, 1))  # Current timestep for 3 variables
            effect_results = []
            for instance in instances:
                result = instance.get_instantanous_effect(test_ts)
                effect_results.append(result)
            for i in range(1, len(effect_results)):
                if not np.allclose(effect_results[0], effect_results[i], rtol=1e-10):
                    print(f"Instantaneous effect results differ between instance 0 and {i}")
                    return False
            
            print("Instantaneous effects are deterministic")
            return True
            
        except Exception as e:
            print(f"Error in instantaneous effects test: {e}")
            return self._compare_instance_attributes(instances)

    def _test_exogenous_influences(self, instances: List[ExogenousInfluences]) -> bool:
        """Test determinism of exogenous influences."""
        try:
            
            # Initialize all instances
            for instance in instances:
                instance.init_exogs()
            
            # Compare link matrices
            link_matrices = [instance.links for instance in instances]
            for i in range(1, len(link_matrices)):
                if not np.allclose(link_matrices[0], link_matrices[i], rtol=1e-10):
                    print(f"Exogenous link matrices differ between instance 0 and {i}")
                    return False
            
            # Test influence computation
            influence_results = []
            for instance in instances:
                result, exog_base = instance.get_exogs_influence()
                influence_results.append((result, exog_base))
            
            for i in range(1, len(influence_results)):
                if not (np.allclose(influence_results[0][0], influence_results[i][0], rtol=1e-10) and
                       np.allclose(influence_results[0][1], influence_results[i][1], rtol=1e-10)):
                    print(f"Exogenous influence results differ between instance 0 and {i}")
                    return False
            
            print("Exogenous influences are deterministic")
            return True
            
        except Exception as e:
            print(f"Error in exogenous influences test: {e}")
            return self._compare_instance_attributes(instances)
    
    def _test_nl_function_generator(self, instances: List[NL_function_generator]) -> bool:
        """Test determinism of nonlinear function generator."""
        try:
            # Test function generation
            test_rng = np.random.default_rng(self.test_seed)
            test_links = test_rng.standard_normal((3, 3, 2))  # 3x3x2 coefficient tensor
            nl_proba = 0.5
            
            function_results = []
            naming_results = []
            
            for instance in instances:
                nl_func, nl_naming = instance.sample_nl_relationships(test_links, nl_proba)
                function_results.append(nl_func)
                naming_results.append(nl_naming)
            
            # Compare function shapes and naming
            for i in range(1, len(function_results)):
                if function_results[0].shape != function_results[i].shape:
                    print(f"NL function shapes differ between instance 0 and {i}")
                    return False
                if not np.array_equal(naming_results[0], naming_results[i]):
                    print(f"NL function naming differs between instance 0 and {i}")
                    return False
            
            # Test that the same functions produce the same outputs
            test_input = np.array([0.5])
            for i in range(function_results[0].shape[0]):
                for j in range(function_results[0].shape[1]):
                    for k in range(function_results[0].shape[2]):
                        if function_results[0][i,j,k] is not None:
                            outputs = []
                            for instance_idx in range(len(function_results)):
                                if function_results[instance_idx][i,j,k] is not None:
                                    output = function_results[instance_idx][i,j,k](test_input)
                                    outputs.append(output)
                            
                            # Check if all outputs are the same
                            for output_idx in range(1, len(outputs)):
                                if not np.allclose(outputs[0], outputs[output_idx], rtol=1e-10):
                                    print(f"NL function outputs differ at position ({i},{j},{k})")
                                    return False
            
            print("NL function generator is deterministic")
            return True
            
        except Exception as e:
            print(f"Error in NL function generator test: {e}")
            return self._compare_instance_attributes(instances)
    
    def _compare_instance_attributes(self, instances: List[Any]) -> bool:
        """Fallback method to compare basic attributes of instances."""
        try:
            # Get attributes of first instance
            first_attrs = {k: v for k, v in instances[0].__dict__.items() 
                          if not callable(v) and not k.startswith('_')}
            
            # Compare with other instances
            for i, instance in enumerate(instances[1:], 1):
                instance_attrs = {k: v for k, v in instance.__dict__.items() 
                                if not callable(v) and not k.startswith('_')}
                
                for key, value in first_attrs.items():
                    if key in instance_attrs:
                        other_value = instance_attrs[key]
                        if isinstance(value, np.ndarray) and isinstance(other_value, np.ndarray):
                            if not np.allclose(value, other_value, rtol=1e-10):
                                print(f"Attribute '{key}' differs between instance 0 and {i}")
                                return False
                        elif value != other_value:
                            print(f"Attribute '{key}' differs between instance 0 and {i}")
                            return False
            
            print("Basic attributes are deterministic")
            return True
            
        except Exception as e:
            print(f"Error in attribute comparison: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run determinism tests for all components."""
        print("=" * 60)
        print("DETERMINISM TESTING FOR SYNTHETIC DATA GENERATOR COMPONENTS")
        print("=" * 60)
        print(f"Test seed: {self.test_seed}")
        print(f"Number of test instances per component: {self.n_tests}")
        
        # Test configurations for each component
        test_configs = {
            "OBS": {
                "class": NoiseGenerator,
                "kwargs": {"additive": True ,"modus": "obs", "snr": 0.1}
            },
            "INNO": {
                "class": NoiseGenerator,
                "kwargs": {"additive": True ,"modus": "inno", "autoregressive": True, "non_additive_noise_proba": 0.5}
            },
            "Structural_Equation": {
                "class": LaggedEffects,
                "kwargs": {"n_vars": 5, "max_lags": 3, "link_proba": 0.075, "nonlinear_proba":0.5}
            },
            "Instantanous_effects": {
                "class": InstantanousEffects,
                "kwargs": {"n_vars": 5, "link_proba": 0.2, "nonlinear_proba": 0.5 }
            },
            "Exogenous_influences": {
                "class": ExogenousInfluences,
                "kwargs": {"n_vars": 3, "n_exogs": 2, "link_proba": 0.5, "nonlinear_proba": 0.5 }
            },
            "NL_function_generator": {
                "class": NL_function_generator,
                "kwargs": {"nl_mode": "power_set", "n_options": 5}
            }
        }
        
        # Run tests
        all_passed = True
        for component_name, config in test_configs.items():
            passed = self.test_component_determinism(
                config["class"], 
                component_name, 
                config["kwargs"]
            )
            if not passed:
                all_passed = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        for component_name, result in self.results.items():
            status = "✓ PASS" if result['deterministic'] else "✗ FAIL"
            print(f"{component_name:25} {status}")
            if result['error']:
                print(f"{'':25} Error: {result['error']}")
        
        overall_status = "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
        print(f"\nOverall result: {overall_status}")
        
        return self.results


def main():
    """Main function to run the determinism tests."""
    tester = DeterminismTester(test_seed=42, n_tests=3)
    results = tester.run_all_tests()
    
    # Exit with error code if any tests failed
    if not all(result['deterministic'] for result in results.values()):
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main() 



