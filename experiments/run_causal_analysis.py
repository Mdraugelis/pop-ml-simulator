#!/usr/bin/env python3
"""
Causal Inference Analysis Experiment Runner
"""

import os
import sys
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import hydra
from omegaconf import DictConfig, OmegaConf
from pop_ml_simulator import VectorizedTemporalRiskSimulator
from pop_ml_simulator.causal_inference import RDDAnalysis, DiDAnalysis, ITSAnalysis


def run_causal_analysis(cfg: DictConfig) -> None:
    """
    Runs a simulation and performs causal inference analysis.
    """
    logging.basicConfig(level=logging.INFO)

    # Initialize simulator
    simulator = VectorizedTemporalRiskSimulator(
        n_patients=cfg.population.n_patients,
        n_timesteps=cfg.temporal.n_timesteps,
        annual_incident_rate=cfg.population.annual_incident_rate,
        intervention_effectiveness=cfg.intervention.effectiveness,
        random_seed=cfg.random_seed
    )

    # Run full simulation
    results = simulator.run_full_simulation(
        prediction_times=cfg.prediction_schedule.prediction_times,
        target_sensitivity=cfg.ml_model.target_sensitivity,
        target_ppv=cfg.ml_model.target_ppv,
        assignment_strategy=cfg.intervention.assignment_strategy,
        threshold=cfg.intervention.threshold
    )

    # Perform causal analyses
    if "rdd" in cfg.causal_inference.methods:
        logging.info("--- RDD Analysis ---")
        rdd_analyzer = RDDAnalysis(
            results,
            risk_threshold=cfg.causal_inference.rdd.threshold,
            bandwidth=cfg.causal_inference.rdd.bandwidth
        )
        rdd_results = rdd_analyzer.analyze_intervention_effect()
        logging.info(f"True effect: {rdd_results['true_effect']:.3f}")
        logging.info(f"Estimated effect: {rdd_results['estimated_effect']:.3f}")

    if "did" in cfg.causal_inference.methods:
        logging.info("--- DiD Analysis ---")
        did_analyzer = DiDAnalysis(
            results,
            intervention_start_time=cfg.causal_inference.did.pre_period_length
        )
        did_results = did_analyzer.analyze_intervention_effect()
        logging.info(f"True effect: {did_results['true_effect']:.3f}")
        logging.info(f"Estimated effect: {did_results['estimated_effect']:.3f}")

    if "its" in cfg.causal_inference.methods:
        logging.info("--- ITS Analysis ---")
        its_analyzer = ITSAnalysis(
            results,
            intervention_time=cfg.causal_inference.its.intervention_time
        )
        its_results = its_analyzer.analyze_intervention_effect()
        logging.info(f"True effect: {its_results['true_effect']:.3f}")
        logging.info(f"Estimated effect: {its_results['relative_effect']:.3f}")


@hydra.main(version_base=None, config_path="../configs", config_name="baseline_simulation")
def main(cfg: DictConfig) -> None:
    """Main experiment runner."""
    run_causal_analysis(cfg)


if __name__ == "__main__":
    main()
