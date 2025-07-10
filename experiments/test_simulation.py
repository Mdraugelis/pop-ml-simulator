#!/usr/bin/env python3
"""
Quick Simulation Test Script

This script provides a quick verification of the VectorizedTemporalRiskSimulator's
basic functionality without the full configuration overhead.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pop_ml_simulator import VectorizedTemporalRiskSimulator

# Mock expensive ML optimization for all tests
def mock_ml_optimization():
    """Context manager to mock expensive ML optimization."""
    return patch(
        'pop_ml_simulator.ml_simulation.MLPredictionSimulator.optimize_noise_parameters',
        return_value={'correlation': 0.7, 'scale': 0.3}
    )


def test_basic_functionality():
    """Test basic simulator functionality."""
    print("=== Basic Functionality Test ===")
    
    with mock_ml_optimization():
        # Create a small simulator for quick testing
        simulator = VectorizedTemporalRiskSimulator(
            n_patients=10,  # Reduced for speed
            n_timesteps=8,  # Reduced for speed
            annual_incident_rate=0.1,
            intervention_effectiveness=0.3,
            prediction_window=4,  # Reduced to fit within 8 timesteps
            random_seed=42
        )
        
        print(f"✓ Simulator initialized with {simulator.n_patients} patients")
        
        # Test population initialization
        simulator.initialize_population()
        print(f"✓ Population initialized, mean base risk: {np.mean(simulator.results.patient_base_risks):.4f}")
        
        # Test temporal evolution
        simulator.simulate_temporal_evolution()
        print(f"✓ Temporal evolution completed, risk matrix shape: {simulator.results.temporal_risk_matrix.shape}")
        
        # Test ML predictions (with reduced iterations for speed)
        simulator.generate_ml_predictions(
            prediction_times=[0],  # Reduced to one prediction time
            target_sensitivity=0.8,
            target_ppv=0.3,
            n_optimization_iterations=1  # Minimal for speed
        )
        print(f"✓ ML predictions generated for {len(simulator.results.ml_predictions)} time points")
        
        # Test intervention assignment
        simulator.assign_interventions(
            assignment_strategy="ml_threshold",
            threshold=0.5
        )
        print(f"✓ Interventions assigned, total interventions: {simulator.results.intervention_matrix.nnz}")
        
        # Test incident simulation
        simulator.simulate_incidents(generate_counterfactuals=True)
        total_incidents = np.sum(simulator.results.incident_matrix)
        total_counterfactual = np.sum(simulator.results.counterfactual_incidents)
        print(f"✓ Incidents simulated - Actual: {total_incidents}, Counterfactual: {total_counterfactual}")
        
        print("✓ All basic functionality tests passed!")
        return True


def test_full_pipeline():
    """Test the complete simulation pipeline."""
    print("\n=== Full Pipeline Test ===")
    
    with mock_ml_optimization():
        start_time = time.time()
        
        # Create simulator
        simulator = VectorizedTemporalRiskSimulator(
            n_patients=15,  # Reduced for speed
            n_timesteps=8,  # Reduced for speed
            annual_incident_rate=0.08,
            intervention_effectiveness=0.25,
            prediction_window=4,  # Reduced to fit within 8 timesteps
            random_seed=123
        )
        
        # Run full simulation
        results = simulator.run_full_simulation(
            prediction_times=[0],  # Reduced to one prediction time
            target_sensitivity=0.75,
            target_ppv=0.35,
            assignment_strategy="ml_threshold",
            threshold=0.5,
            generate_counterfactuals=True,
            n_optimization_iterations=1,  # Minimal for speed
            concentration=0.5,
            rho=0.9,
            sigma=0.1
        )
        
        duration = time.time() - start_time
        
        # Verify results
        assert results.n_patients == 15
        assert results.n_timesteps == 8
        assert results.temporal_risk_matrix.shape == (15, 8)
        assert results.incident_matrix.shape == (15, 8)
        assert len(results.ml_predictions) == 1
        assert results.counterfactual_incidents is not None
        
        print(f"✓ Full pipeline completed in {duration:.2f} seconds")
        print(f"✓ Results validation passed")
        
        return True


def test_summary_statistics():
    """Test summary statistics generation."""
    print("\n=== Summary Statistics Test ===")
    
    with mock_ml_optimization():
        # Create and run a quick simulation
        simulator = VectorizedTemporalRiskSimulator(
            n_patients=8,  # Reduced for speed
            n_timesteps=6,  # Reduced for speed
            annual_incident_rate=0.1,
            prediction_window=3,  # Reduced to fit within 6 timesteps
            random_seed=456
        )
        
        results = simulator.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )
        
        # Get summary statistics
        stats = simulator.get_summary_statistics()
        
        # Verify statistics structure
        required_fields = [
            'n_patients', 'n_timesteps', 'intervention_effectiveness',
            'total_incidents', 'total_interventions', 'mean_base_risk'
        ]
        
        for field in required_fields:
            assert field in stats, f"Missing field: {field}"
        
        print(f"✓ Summary statistics generated with {len(stats)} fields")
        print(f"  - Mean base risk: {stats['mean_base_risk']:.4f}")
        print(f"  - Total incidents: {stats['total_incidents']}")
        print(f"  - Total interventions: {stats['total_interventions']}")
        
        return True


def test_patient_trajectory():
    """Test patient trajectory retrieval."""
    print("\n=== Patient Trajectory Test ===")
    
    with mock_ml_optimization():
        # Create and run simulation
        simulator = VectorizedTemporalRiskSimulator(
            n_patients=8,  # Reduced for speed
            n_timesteps=6,  # Reduced for speed
            annual_incident_rate=0.1,
            prediction_window=3,  # Reduced to fit within 6 timesteps
            random_seed=789
        )
        
        results = simulator.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )
        
        # Get trajectory for first patient
        trajectory = simulator.get_patient_trajectory(0)
        
        # Verify trajectory structure
        required_keys = ['base_risk', 'temporal_risks', 'incidents', 'interventions']
        for key in required_keys:
            assert key in trajectory, f"Missing trajectory key: {key}"
        
        # Verify trajectory lengths
        assert len(trajectory['temporal_risks']) == 6
        assert len(trajectory['incidents']) == 6
        assert len(trajectory['interventions']) == 6
        
        print(f"✓ Patient trajectory retrieved successfully")
        print(f"  - Base risk: {trajectory['base_risk']:.4f}")
        print(f"  - Risk range: {np.min(trajectory['temporal_risks']):.4f} - {np.max(trajectory['temporal_risks']):.4f}")
        print(f"  - Total incidents: {np.sum(trajectory['incidents'])}")
        print(f"  - Total interventions: {np.sum(trajectory['interventions'])}")
        
        return True


def test_different_assignment_strategies():
    """Test different intervention assignment strategies."""
    print("\n=== Assignment Strategy Test ===")
    
    with mock_ml_optimization():
        strategies = ["ml_threshold", "random", "top_k"]
        
        for strategy in strategies:
            print(f"Testing {strategy} strategy...")
            
            simulator = VectorizedTemporalRiskSimulator(
                n_patients=8,  # Reduced for speed
                n_timesteps=6,  # Reduced for speed
                annual_incident_rate=0.1,
                prediction_window=3,  # Reduced to fit within 6 timesteps
                random_seed=42
            )
            
            simulator.initialize_population()
            simulator.simulate_temporal_evolution()
            simulator.generate_ml_predictions(
                prediction_times=[0],
                n_optimization_iterations=1
            )
            
            # Assign interventions with different strategies
            if strategy == "random" or strategy == "top_k":
                simulator.assign_interventions(
                    assignment_strategy=strategy,
                    treatment_fraction=0.3
                )
            else:
                simulator.assign_interventions(
                    assignment_strategy=strategy,
                    threshold=0.5
                )
            
            total_interventions = simulator.results.intervention_matrix.nnz
            print(f"  ✓ {strategy}: {total_interventions} interventions assigned")
        
        print("✓ All assignment strategies tested successfully")
        return True


def test_reproducibility():
    """Test that results are reproducible with same seed."""
    print("\n=== Reproducibility Test ===")
    
    with mock_ml_optimization():
        seed = 999
        
        # Run simulation twice with same seed
        results1 = None
        results2 = None
        
        for i in range(2):
            simulator = VectorizedTemporalRiskSimulator(
                n_patients=8,  # Reduced for speed
                n_timesteps=6,  # Reduced for speed
                annual_incident_rate=0.1,
                prediction_window=3,  # Reduced to fit within 6 timesteps
                random_seed=seed
            )
            
            results = simulator.run_full_simulation(
                prediction_times=[0],
                n_optimization_iterations=1
            )
            
            if i == 0:
                results1 = results
            else:
                results2 = results
        
        # Compare results
        np.testing.assert_array_equal(results1.patient_base_risks, results2.patient_base_risks)
        np.testing.assert_array_equal(results1.temporal_risk_matrix, results2.temporal_risk_matrix)
        
        print("✓ Reproducibility test passed - results are identical with same seed")
        return True


def run_performance_benchmark():
    """Run a simple performance benchmark."""
    print("\n=== Performance Benchmark ===")

    with mock_ml_optimization():
        sizes = [(50, 6), (100, 8), (200, 10)]  # Dramatically reduced for speed

        for n_patients, n_timesteps in sizes:
            start_time = time.time()
            
            simulator = VectorizedTemporalRiskSimulator(
                n_patients=n_patients,
                n_timesteps=n_timesteps,
                annual_incident_rate=0.08,
                prediction_window=min(4, n_timesteps-2),  # Ensure window fits
                random_seed=42
            )
            
            results = simulator.run_full_simulation(
                prediction_times=[0],
                n_optimization_iterations=1
            )
            
            duration = time.time() - start_time
            
            print(f"  {n_patients:4d} patients × {n_timesteps:2d} timesteps: {duration:6.2f}s")
        
        print("✓ Performance benchmark completed")
        return True


def main():
    """Run all tests."""
    print("VectorizedTemporalRiskSimulator - Quick Verification Tests")
    print("(Optimized for speed with reduced ML optimization iterations)")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_full_pipeline,
        test_summary_statistics,
        test_patient_trajectory,
        test_different_assignment_strategies,
        test_reproducibility,
        run_performance_benchmark
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VectorizedTemporalRiskSimulator is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())