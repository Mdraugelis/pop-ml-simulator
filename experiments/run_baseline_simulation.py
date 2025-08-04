#!/usr/bin/env python3
"""
Baseline Simulation Experiment Runner

This script runs a baseline healthcare AI intervention simulation using the
VectorizedTemporalRiskSimulator with Hydra configuration management.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import hydra
from omegaconf import DictConfig, OmegaConf
from pop_ml_simulator import VectorizedTemporalRiskSimulator, SimulationResults


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging for the simulation."""
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper()),
        format=cfg.logging.log_format,
        handlers=[
            logging.FileHandler(cfg.logging.log_file),
            logging.StreamHandler()
        ]
    )


def create_output_directory(cfg: DictConfig) -> Path:
    """Create output directory for simulation results."""
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def initialize_simulator(cfg: DictConfig) -> VectorizedTemporalRiskSimulator:
    """Initialize the vectorized temporal risk simulator."""
    logging.info("Initializing VectorizedTemporalRiskSimulator...")
    
    simulator = VectorizedTemporalRiskSimulator(
        n_patients=cfg.population.n_patients,
        n_timesteps=cfg.temporal.n_timesteps,
        annual_incident_rate=cfg.population.annual_incident_rate,
        intervention_effectiveness=cfg.intervention.effectiveness,
        timestep_duration=cfg.temporal.timestep_duration,
        prediction_window=cfg.temporal.prediction_window,
        random_seed=cfg.random_seed
    )
    
    logging.info(f"Simulator initialized with {cfg.population.n_patients} patients, "
                f"{cfg.temporal.n_timesteps} timesteps")
    
    return simulator


def run_simulation(simulator: VectorizedTemporalRiskSimulator, cfg: DictConfig) -> SimulationResults:
    """Run the complete simulation pipeline."""
    logging.info("Starting simulation pipeline...")
    start_time = time.time()
    
    # Run full simulation
    results = simulator.run_full_simulation(
        prediction_times=cfg.prediction_schedule.prediction_times,
        target_sensitivity=cfg.ml_model.target_sensitivity,
        target_ppv=cfg.ml_model.target_ppv,
        assignment_strategy=cfg.intervention.assignment_strategy,
        threshold=cfg.intervention.threshold,
        generate_counterfactuals=cfg.simulation.generate_counterfactuals,
        n_optimization_iterations=cfg.ml_model.n_optimization_iterations,
        treatment_fraction=cfg.intervention.treatment_fraction,
        # Population initialization parameters
        concentration=cfg.population.risk_concentration,
        rho=cfg.temporal_dynamics.rho,
        sigma=cfg.temporal_dynamics.sigma,
        temporal_bounds=cfg.temporal_dynamics.temporal_bounds,
        seasonal_amplitude=cfg.temporal_dynamics.seasonal_amplitude,
        seasonal_period=cfg.temporal_dynamics.seasonal_period
    )
    
    duration = time.time() - start_time
    logging.info(f"Simulation completed in {duration:.2f} seconds")
    
    return results


def compute_summary_statistics(results: SimulationResults, cfg: DictConfig) -> Dict[str, Any]:
    """Compute comprehensive summary statistics."""
    logging.info("Computing summary statistics...")
    
    # Basic simulation stats
    basic_stats = {
        'simulation_parameters': {
            'n_patients': results.n_patients,
            'n_timesteps': results.n_timesteps,
            'intervention_effectiveness': results.intervention_effectiveness,
            'prediction_times': results.ml_prediction_times,
            'random_seed': cfg.random_seed
        },
        'population_statistics': {
            'mean_base_risk': float(np.mean(results.patient_base_risks)),
            'std_base_risk': float(np.std(results.patient_base_risks)),
            'min_base_risk': float(np.min(results.patient_base_risks)),
            'max_base_risk': float(np.max(results.patient_base_risks)),
            'median_base_risk': float(np.median(results.patient_base_risks))
        },
        'temporal_dynamics': {
            'mean_temporal_risk': float(np.mean(results.temporal_risk_matrix)),
            'std_temporal_risk': float(np.std(results.temporal_risk_matrix)),
            'temporal_risk_range': [
                float(np.min(results.temporal_risk_matrix)),
                float(np.max(results.temporal_risk_matrix))
            ]
        },
        'intervention_statistics': {
            'total_interventions': int(results.intervention_matrix.nnz),
            'intervention_coverage': float(results.intervention_coverage),
            'unique_patients_treated': int(len(np.unique(results.intervention_matrix.nonzero()[0])))
        },
        'outcome_statistics': {
            'total_incidents': int(np.sum(results.incident_matrix)),
            'incident_rate': float(np.mean(results.incident_matrix)),
            'patients_with_incidents': int(np.sum(np.any(results.incident_matrix, axis=1)))
        }
    }
    
    # Add counterfactual statistics if available
    if results.counterfactual_incidents is not None:
        basic_stats['counterfactual_statistics'] = {
            'counterfactual_incidents': int(np.sum(results.counterfactual_incidents)),
            'incident_reduction': float(results.incident_reduction),
            'counterfactual_incident_rate': float(np.mean(results.counterfactual_incidents))
        }
    
    # Add ML prediction statistics
    if results.ml_predictions:
        all_predictions = np.concatenate(list(results.ml_predictions.values()))
        all_binary_predictions = np.concatenate(list(results.ml_binary_predictions.values()))
        
        basic_stats['ml_prediction_statistics'] = {
            'mean_ml_score': float(np.mean(all_predictions)),
            'std_ml_score': float(np.std(all_predictions)),
            'ml_score_range': [float(np.min(all_predictions)), float(np.max(all_predictions))],
            'positive_prediction_rate': float(np.mean(all_binary_predictions)),
            'total_predictions': len(all_predictions)
        }
    
    return basic_stats


def create_visualizations(results: SimulationResults, cfg: DictConfig, output_dir: Path) -> None:
    """Create visualization plots."""
    if not cfg.output.generate_plots:
        return
    
    logging.info("Creating visualizations...")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Risk Distribution Plot
    if "risk_distribution" in cfg.output.plot_types:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Base Risk Distribution
        plt.subplot(2, 2, 1)
        plt.hist(results.patient_base_risks, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Base Annual Risk')
        plt.ylabel('Number of Patients')
        plt.title('Distribution of Patient Base Risks')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Temporal Risk Range
        plt.subplot(2, 2, 2)
        risk_ranges = np.max(results.temporal_risk_matrix, axis=1) - np.min(results.temporal_risk_matrix, axis=1)
        plt.hist(risk_ranges, bins=50, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Risk Range (Max - Min)')
        plt.ylabel('Number of Patients')
        plt.title('Distribution of Temporal Risk Ranges')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: ML Predictions (if available)
        if results.ml_predictions:
            plt.subplot(2, 2, 3)
            all_predictions = np.concatenate(list(results.ml_predictions.values()))
            plt.hist(all_predictions, bins=50, alpha=0.7, edgecolor='black', color='green')
            plt.xlabel('ML Prediction Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of ML Prediction Scores')
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Intervention Distribution
        plt.subplot(2, 2, 4)
        intervention_counts = np.sum(results.intervention_matrix.toarray(), axis=1)
        plt.hist(intervention_counts, bins=max(1, int(np.max(intervention_counts))), 
                alpha=0.7, edgecolor='black', color='red')
        plt.xlabel('Number of Interventions per Patient')
        plt.ylabel('Number of Patients')
        plt.title('Distribution of Interventions per Patient')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Temporal Trends Plot
    if "temporal_trends" in cfg.output.plot_types:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Population Mean Risk Over Time
        plt.subplot(2, 2, 1)
        mean_risks = np.mean(results.temporal_risk_matrix, axis=0)
        plt.plot(mean_risks, linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Mean Risk')
        plt.title('Population Mean Risk Over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Incident Rate Over Time
        plt.subplot(2, 2, 2)
        incident_rates = np.mean(results.incident_matrix, axis=0)
        plt.plot(incident_rates, linewidth=2, color='red')
        plt.xlabel('Time Step')
        plt.ylabel('Incident Rate')
        plt.title('Incident Rate Over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Sample Patient Trajectories
        plt.subplot(2, 2, 3)
        sample_patients = np.random.choice(results.n_patients, min(20, results.n_patients), replace=False)
        for patient in sample_patients:
            plt.plot(results.temporal_risk_matrix[patient, :], alpha=0.7, linewidth=1)
        plt.xlabel('Time Step')
        plt.ylabel('Risk')
        plt.title('Sample Patient Risk Trajectories')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Intervention Timeline
        plt.subplot(2, 2, 4)
        intervention_counts = np.sum(results.intervention_matrix.toarray(), axis=0)
        plt.bar(range(len(intervention_counts)), intervention_counts, alpha=0.7, color='orange')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Interventions')
        plt.title('Interventions Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Intervention Effects Plot
    if "intervention_effects" in cfg.output.plot_types and results.counterfactual_incidents is not None:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Incident Comparison
        plt.subplot(2, 2, 1)
        actual_incidents = np.sum(results.incident_matrix, axis=1)
        counterfactual_incidents = np.sum(results.counterfactual_incidents, axis=1)
        
        plt.scatter(counterfactual_incidents, actual_incidents, alpha=0.6)
        max_incidents = max(np.max(actual_incidents), np.max(counterfactual_incidents))
        plt.plot([0, max_incidents], [0, max_incidents], 'r--', alpha=0.5)
        plt.xlabel('Counterfactual Incidents')
        plt.ylabel('Actual Incidents')
        plt.title('Actual vs Counterfactual Incidents')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Incident Reduction by Patient
        plt.subplot(2, 2, 2)
        incident_reduction = counterfactual_incidents - actual_incidents
        plt.hist(incident_reduction, bins=50, alpha=0.7, edgecolor='black', color='green')
        plt.xlabel('Incident Reduction')
        plt.ylabel('Number of Patients')
        plt.title('Distribution of Incident Reduction')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Effectiveness by Time
        plt.subplot(2, 2, 3)
        actual_rates = np.mean(results.incident_matrix, axis=0)
        counterfactual_rates = np.mean(results.counterfactual_incidents, axis=0)
        
        plt.plot(actual_rates, label='Actual', linewidth=2)
        plt.plot(counterfactual_rates, label='Counterfactual', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Incident Rate')
        plt.title('Incident Rates Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative Effect
        plt.subplot(2, 2, 4)
        cumulative_actual = np.cumsum(np.sum(results.incident_matrix, axis=0))
        cumulative_counterfactual = np.cumsum(np.sum(results.counterfactual_incidents, axis=0))
        
        plt.plot(cumulative_actual, label='Actual', linewidth=2)
        plt.plot(cumulative_counterfactual, label='Counterfactual', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative Incidents')
        plt.title('Cumulative Incidents Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'intervention_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Visualizations saved to {output_dir}")


def save_results(results: SimulationResults, stats: Dict[str, Any], cfg: DictConfig, output_dir: Path) -> None:
    """Save simulation results to files."""
    if not cfg.output.save_results:
        return
    
    logging.info("Saving simulation results...")
    
    # Save summary statistics
    if cfg.output.save_components.summary_stats:
        stats_file = output_dir / "summary_statistics.json"
        import json
        from omegaconf import ListConfig, DictConfig
        
        def make_json_serializable(obj, path="root"):
            """
            Recursively convert objects to JSON-serializable format.
            Handles AnyNode, Hydra configs, numpy arrays, and other complex objects.
            """
            try:
                # Handle None
                if obj is None:
                    return None
                
                # Handle basic JSON-serializable types
                if isinstance(obj, (str, int, float, bool)):
                    return obj
                
                # Handle Hydra config objects
                if isinstance(obj, (ListConfig, DictConfig)):
                    try:
                        return list(obj) if isinstance(obj, ListConfig) else dict(obj)
                    except Exception:
                        return f"<Hydra config object: {type(obj).__name__}>"
                
                # Handle numpy arrays and scalars
                if hasattr(obj, 'tolist'):
                    try:
                        return obj.tolist()
                    except Exception:
                        return f"<numpy array: shape={getattr(obj, 'shape', 'unknown')}>"
                
                # Handle numpy scalars
                if hasattr(obj, 'item'):
                    try:
                        return obj.item()
                    except Exception:
                        return f"<numpy scalar: {type(obj).__name__}>"
                
                # Handle dictionaries
                if isinstance(obj, dict):
                    result = {}
                    for k, v in obj.items():
                        try:
                            key = make_json_serializable(k, f"{path}.{k}")
                            value = make_json_serializable(v, f"{path}.{k}")
                            result[str(key)] = value
                        except Exception as e:
                            result[str(k)] = f"<serialization error: {str(e)}>"
                    return result
                
                # Handle lists and tuples
                if isinstance(obj, (list, tuple)):
                    result = []
                    for i, item in enumerate(obj):
                        try:
                            result.append(make_json_serializable(item, f"{path}[{i}]"))
                        except Exception as e:
                            result.append(f"<serialization error: {str(e)}>")
                    return result
                
                # Handle sparse matrices
                if hasattr(obj, 'toarray'):
                    try:
                        return obj.toarray().tolist()
                    except Exception:
                        return f"<sparse matrix: {type(obj).__name__}>"
                
                # Handle AnyNode objects specifically
                if type(obj).__name__ == 'AnyNode':
                    try:
                        # Try to extract basic attributes from AnyNode
                        node_data = {}
                        for attr in ['name', 'value', 'children', 'parent']:
                            if hasattr(obj, attr):
                                attr_value = getattr(obj, attr)
                                if attr == 'children':
                                    node_data[attr] = f"<{len(attr_value)} children>" if attr_value else "no children"
                                elif attr == 'parent':
                                    node_data[attr] = f"<parent: {getattr(attr_value, 'name', 'unnamed')}>" if attr_value else "no parent"
                                else:
                                    node_data[attr] = make_json_serializable(attr_value, f"{path}.{attr}")
                        return node_data
                    except Exception as e:
                        return f"<AnyNode object: {str(e)}>"
                
                # Handle other complex objects
                if hasattr(obj, '__dict__'):
                    try:
                        return f"<{type(obj).__name__} object>"
                    except Exception:
                        return f"<unknown object type>"
                
                # Last resort - try to convert to string
                try:
                    return str(obj)
                except Exception:
                    return f"<non-serializable object: {type(obj).__name__}>"
                    
            except Exception as e:
                logging.warning(f"Failed to serialize object at {path}: {e}")
                return f"<serialization failed: {str(e)}>"
        
        # Debug function to find AnyNode objects
        def find_problem_objects(obj, path="root", depth=0):
            """Find and report non-serializable objects, especially AnyNode."""
            if depth > 10:  # Prevent infinite recursion
                return
                
            try:
                obj_type = type(obj).__name__
                
                # Report AnyNode objects
                if obj_type == 'AnyNode':
                    logging.warning(f"Found AnyNode object at {path}")
                    return
                
                # Report other potentially problematic objects
                if hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, list, dict, tuple)):
                    if not hasattr(obj, 'tolist') and not hasattr(obj, 'item'):  # Skip numpy objects
                        logging.info(f"Found complex object {obj_type} at {path}")
                
                # Recurse into containers
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        find_problem_objects(v, f"{path}.{k}", depth + 1)
                elif isinstance(obj, (list, tuple)):
                    for i, item in enumerate(obj):
                        find_problem_objects(item, f"{path}[{i}]", depth + 1)
                        
            except Exception as e:
                logging.warning(f"Error inspecting object at {path}: {e}")
        
        # Debug: find problematic objects before serialization
        logging.info("Scanning for non-serializable objects...")
        find_problem_objects(stats)
        
        # Convert stats to JSON-serializable format
        try:
            serializable_stats = make_json_serializable(stats)
            with open(stats_file, 'w') as f:
                json.dump(serializable_stats, f, indent=2)
            logging.info(f"Summary statistics saved to {stats_file}")
        except Exception as e:
            logging.error(f"Failed to save summary statistics: {e}")
            # Try to save a minimal version
            minimal_stats = {
                'error': f"Failed to serialize full stats: {str(e)}",
                'basic_info': {
                    'n_patients': getattr(results, 'n_patients', 'unknown'),
                    'n_timesteps': getattr(results, 'n_timesteps', 'unknown'),
                    'timestamp': str(pd.Timestamp.now())
                }
            }
            with open(stats_file, 'w') as f:
                json.dump(minimal_stats, f, indent=2)
            logging.info(f"Minimal summary statistics saved to {stats_file}")
    
    # Save patient base risks
    if cfg.output.save_components.patient_risks:
        risk_df = pd.DataFrame({
            'patient_id': range(results.n_patients),
            'base_risk': results.patient_base_risks
        })
        
        if "csv" in cfg.output.save_formats:
            risk_df.to_csv(output_dir / "patient_risks.csv", index=False)
        if "parquet" in cfg.output.save_formats:
            risk_df.to_parquet(output_dir / "patient_risks.parquet", index=False)
    
    # Save temporal risk matrix
    if cfg.output.save_components.temporal_matrix:
        if "csv" in cfg.output.save_formats:
            temporal_df = pd.DataFrame(results.temporal_risk_matrix)
            temporal_df.to_csv(output_dir / "temporal_risk_matrix.csv", index=False)
        if "parquet" in cfg.output.save_formats:
            temporal_df = pd.DataFrame(results.temporal_risk_matrix)
            temporal_df.to_parquet(output_dir / "temporal_risk_matrix.parquet", index=False)
    
    # Save incident matrix
    if cfg.output.save_components.incident_matrix:
        incident_df = pd.DataFrame(results.incident_matrix.astype(int))
        if "csv" in cfg.output.save_formats:
            incident_df.to_csv(output_dir / "incident_matrix.csv", index=False)
        if "parquet" in cfg.output.save_formats:
            incident_df.to_parquet(output_dir / "incident_matrix.parquet", index=False)
    
    # Save ML predictions
    if cfg.output.save_components.ml_predictions and results.ml_predictions:
        ml_data = []
        for time, predictions in results.ml_predictions.items():
            for patient_id, score in enumerate(predictions):
                ml_data.append({
                    'patient_id': patient_id,
                    'prediction_time': time,
                    'ml_score': score,
                    'ml_binary': results.ml_binary_predictions[time][patient_id]
                })
        
        ml_df = pd.DataFrame(ml_data)
        if "csv" in cfg.output.save_formats:
            ml_df.to_csv(output_dir / "ml_predictions.csv", index=False)
        if "parquet" in cfg.output.save_formats:
            ml_df.to_parquet(output_dir / "ml_predictions.parquet", index=False)
    
    # Save intervention matrix
    if cfg.output.save_components.interventions:
        intervention_df = pd.DataFrame(results.intervention_matrix.toarray().astype(int))
        if "csv" in cfg.output.save_formats:
            intervention_df.to_csv(output_dir / "intervention_matrix.csv", index=False)
        if "parquet" in cfg.output.save_formats:
            intervention_df.to_parquet(output_dir / "intervention_matrix.parquet", index=False)
    
    # Save counterfactuals
    if cfg.output.save_components.counterfactuals and results.counterfactual_incidents is not None:
        counterfactual_df = pd.DataFrame(results.counterfactual_incidents.astype(int))
        if "csv" in cfg.output.save_formats:
            counterfactual_df.to_csv(output_dir / "counterfactual_incidents.csv", index=False)
        if "parquet" in cfg.output.save_formats:
            counterfactual_df.to_parquet(output_dir / "counterfactual_incidents.parquet", index=False)
    
    logging.info(f"Results saved to {output_dir}")


def run_validation_checks(results: SimulationResults, cfg: DictConfig) -> Dict[str, bool]:
    """Run validation checks on simulation results."""
    if not cfg.validation.run_validation_checks:
        return {}
    
    logging.info("Running validation checks...")
    
    validation_results = {}
    
    # Check population mean
    if cfg.validation.validate_population_mean:
        actual_mean = np.mean(results.patient_base_risks)
        target_mean = cfg.population.annual_incident_rate
        tolerance = cfg.validation.population_mean_tolerance
        
        validation_results['population_mean'] = (
            abs(actual_mean - target_mean) <= tolerance
        )
        
        if validation_results['population_mean']:
            logging.info(f"✓ Population mean validation passed: {actual_mean:.4f} "
                        f"(target: {target_mean:.4f})")
        else:
            logging.warning(f"✗ Population mean validation failed: {actual_mean:.4f} "
                           f"(target: {target_mean:.4f}, tolerance: {tolerance})")
    
    # Check temporal autocorrelation
    if cfg.validation.validate_temporal_autocorrelation:
        # Calculate autocorrelation for sample of patients
        sample_patients = np.random.choice(results.n_patients, min(100, results.n_patients), replace=False)
        autocorrs = []
        
        for patient in sample_patients:
            trajectory = results.temporal_risk_matrix[patient, :]
            if len(trajectory) > 1:
                autocorr = np.corrcoef(trajectory[:-1], trajectory[1:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)
        
        if autocorrs:
            mean_autocorr = np.mean(autocorrs)
            min_autocorr = cfg.validation.temporal_autocorr_min
            
            validation_results['temporal_autocorrelation'] = (
                mean_autocorr >= min_autocorr
            )
            
            if validation_results['temporal_autocorrelation']:
                logging.info(f"✓ Temporal autocorrelation validation passed: {mean_autocorr:.3f} "
                            f"(min: {min_autocorr:.3f})")
            else:
                logging.warning(f"✗ Temporal autocorrelation validation failed: {mean_autocorr:.3f} "
                               f"(min: {min_autocorr:.3f})")
    
    # Check intervention coverage
    if cfg.validation.validate_intervention_coverage:
        coverage = results.intervention_coverage
        coverage_range = cfg.validation.intervention_coverage_range
        
        validation_results['intervention_coverage'] = (
            coverage_range[0] <= coverage <= coverage_range[1]
        )
        
        if validation_results['intervention_coverage']:
            logging.info(f"✓ Intervention coverage validation passed: {coverage:.3f} "
                        f"(range: {coverage_range})")
        else:
            logging.warning(f"✗ Intervention coverage validation failed: {coverage:.3f} "
                           f"(range: {coverage_range})")
    
    # Check for no re-enrollment
    if cfg.validation.validate_no_re_enrollment:
        validation_results['no_re_enrollment'] = simulator.validate_no_re_enrollment()
        if validation_results['no_re_enrollment']:
            logging.info("✓ No re-enrollment validation passed")
        else:
            logging.warning("✗ No re-enrollment validation failed")

    # Summary
    passed = sum(validation_results.values())
    total = len(validation_results)
    logging.info(f"Validation summary: {passed}/{total} checks passed")
    
    return validation_results


@hydra.main(version_base=None, config_path="../configs", config_name="baseline_simulation")
def main(cfg: DictConfig) -> None:
    """Main experiment runner."""
    # Setup
    setup_logging(cfg)
    logging.info("Starting baseline simulation experiment")
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directory
    output_dir = create_output_directory(cfg)
    logging.info(f"Output directory: {output_dir}")
    
    # Initialize simulator
    simulator = initialize_simulator(cfg)
    
    # Run simulation
    results = run_simulation(simulator, cfg)
    
    # Compute statistics
    stats = compute_summary_statistics(results, cfg)
    
    # Run validation checks
    validation_results = run_validation_checks(results, cfg)
    stats['validation_results'] = validation_results
    
    # Create visualizations
    create_visualizations(results, cfg, output_dir)
    
    # Save results
    save_results(results, stats, cfg, output_dir)
    
    # Print summary
    logging.info("Simulation completed successfully!")
    logging.info("Summary Statistics:")
    logging.info(f"  - Total patients: {results.n_patients}")
    logging.info(f"  - Total timesteps: {results.n_timesteps}")
    logging.info(f"  - Total interventions: {results.intervention_matrix.nnz}")
    logging.info(f"  - Total incidents: {np.sum(results.incident_matrix)}")
    logging.info(f"  - Intervention coverage: {results.intervention_coverage:.3f}")
    
    if results.counterfactual_incidents is not None:
        logging.info(f"  - Incident reduction: {results.incident_reduction:.3f}")
        logging.info(f"  - Counterfactual incidents: {np.sum(results.counterfactual_incidents)}")
    
    logging.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()