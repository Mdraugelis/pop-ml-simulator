"""
Vectorized temporal risk simulator for healthcare AI intervention modeling.

This module provides a unified orchestration layer that integrates risk
modeling, temporal dynamics, ML predictions, and intervention effects into a
single simulation framework optimized for causal inference research.
"""

import numpy as np
from scipy import sparse  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
from typing import Optional, Dict, List, Tuple, Any
import warnings
from dataclasses import dataclass

from .risk_distribution import assign_patient_risks
from .temporal_dynamics import EnhancedTemporalRiskSimulator
from .hazard_modeling import (
    annual_risk_to_hazard,
    hazard_to_timestep_probability,
    IncidentGenerator
)
from .ml_simulation import MLPredictionSimulator
from .risk_integration import integrate_window_risk, extract_risk_windows
from utils.logging import log_call


@dataclass
class SimulationResults:
    """Container for simulation results and metadata."""

    # Core simulation data
    patient_base_risks: np.ndarray
    temporal_risk_matrix: np.ndarray
    incident_matrix: np.ndarray

    # ML predictions and interventions
    ml_predictions: Dict[int, np.ndarray]
    ml_binary_predictions: Dict[int, np.ndarray]
    intervention_matrix: sparse.csr_matrix
    intervention_times: Dict[int, List[int]]

    # Counterfactual data
    counterfactual_incidents: Optional[np.ndarray] = None

    # Simulation parameters
    n_patients: int = 0
    n_timesteps: int = 0
    intervention_effectiveness: float = 0.0
    ml_prediction_times: Optional[List[int]] = None

    # Performance metrics
    intervention_coverage: float = 0.0
    incident_reduction: float = 0.0

    def __post_init__(self):
        """Validate simulation results structure."""
        if self.ml_prediction_times is None:
            self.ml_prediction_times = []


class VectorizedTemporalRiskSimulator:
    """
    Vectorized temporal risk simulator for healthcare AI intervention modeling.

    This class provides a unified orchestration layer that integrates:
    - Population risk modeling with beta distributions
    - Temporal risk evolution using AR(1) processes
    - ML prediction generation with controlled performance
    - Intervention assignment and effectiveness modeling
    - Incident generation with competing risks
    - Counterfactual outcome generation

    All operations are vectorized for computational efficiency and designed
    to support causal inference research with known ground truth.

    Parameters
    ----------
    n_patients : int
        Number of patients in the simulation
    n_timesteps : int
        Number of time steps to simulate
    annual_incident_rate : float
        Target population annual incident rate
    intervention_effectiveness : float, default=0.25
        Relative risk reduction for treated patients (0.0 to 1.0)
    timestep_duration : float, default=1/52
        Duration of each timestep as fraction of year
    prediction_window : int, default=12
        Length of prediction window in timesteps
    intervention_duration : int, default=1
        Duration of interventions in timesteps. Use -1 for full simulation
        duration
    random_seed : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    results : SimulationResults
        Container for all simulation results and metadata
    temporal_simulator : EnhancedTemporalRiskSimulator
        Temporal risk evolution simulator
    ml_simulator : MLPredictionSimulator
        ML prediction generator
    incident_generator : IncidentGenerator
        Incident generation engine
    """

    def __init__(
        self,
        n_patients: int,
        n_timesteps: int,
        annual_incident_rate: float,
        intervention_effectiveness: float = 0.25,
        timestep_duration: float = 1/52,
        prediction_window: int = 12,
        intervention_duration: int = 1,
        random_seed: Optional[int] = None
    ):
        """Initialize the vectorized temporal risk simulator."""
        self.n_patients = n_patients
        self.n_timesteps = n_timesteps
        self.annual_incident_rate = annual_incident_rate
        self.intervention_effectiveness = intervention_effectiveness
        self.timestep_duration = timestep_duration
        self.prediction_window = prediction_window
        self.intervention_duration = intervention_duration
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize component simulators
        self.temporal_simulator: Optional[EnhancedTemporalRiskSimulator] = None
        self.ml_simulator: Optional[MLPredictionSimulator] = None
        self.incident_generator = IncidentGenerator(timestep_duration)

        # Initialize results container
        self.results = SimulationResults(
            patient_base_risks=np.array([]),
            temporal_risk_matrix=np.array([]),
            incident_matrix=np.array([]),
            ml_predictions={},
            ml_binary_predictions={},
            intervention_matrix=sparse.csr_matrix((0, 0)),
            intervention_times={},
            n_patients=n_patients,
            n_timesteps=n_timesteps,
            intervention_effectiveness=intervention_effectiveness
        )

        # Track simulation state
        self._population_initialized = False
        self._temporal_simulated = False
        self._ml_predictions_generated = False
        self._interventions_assigned = False
        self._incidents_simulated = False
        # Cache for ML training data generation
        self._cached_incident_matrix: Optional[np.ndarray] = None
        self._cached_ml_params: Optional[Dict[str, float]] = None

        # Track active intervention end times for re-enrollment prevention
        self._active_intervention_end_times: Dict[int, int] = {}

    @log_call
    def initialize_population(
        self,
        concentration: float = 0.5,
        rho: float = 0.9,
        sigma: float = 0.1,
        temporal_bounds: Tuple[float, float] = (0.2, 2.5),
        seasonal_amplitude: float = 0.2,
        seasonal_period: int = 52
    ) -> None:
        """
        Initialize patient population with heterogeneous base risks.

        Parameters
        ----------
        concentration : float, default=0.5
            Beta distribution concentration parameter
        rho : float, default=0.9
            AR(1) persistence parameter
        sigma : float, default=0.1
            AR(1) noise standard deviation
        temporal_bounds : tuple, default=(0.2, 2.5)
            (min, max) bounds for temporal modifiers
        seasonal_amplitude : float, default=0.2
            Amplitude of seasonal variation
        seasonal_period : int, default=52
            Period of seasonal cycle
        """
        # Generate base patient risks
        self.results.patient_base_risks = assign_patient_risks(
            self.n_patients,
            self.annual_incident_rate,
            concentration=concentration,
            random_seed=self.random_seed
        )

        # Initialize temporal risk simulator
        self.temporal_simulator = EnhancedTemporalRiskSimulator(
            base_risks=self.results.patient_base_risks,
            rho=rho,
            sigma=sigma,
            temporal_bounds=temporal_bounds,
            seasonal_amplitude=seasonal_amplitude,
            seasonal_period=seasonal_period
        )

        self._population_initialized = True

    @log_call
    def simulate_temporal_evolution(self) -> None:
        """Simulate temporal risk evolution for all patients."""
        if not self._population_initialized:
            raise ValueError("Population must be initialized first")

        # Run temporal simulation
        if self.temporal_simulator is not None:
            self.temporal_simulator.simulate(self.n_timesteps - 1)

            # Extract risk matrix
            self.results.temporal_risk_matrix = (
                self.temporal_simulator.get_risk_matrix()
            )
        else:
            raise ValueError("Temporal simulator not initialized")

        self._temporal_simulated = True

    @log_call
    def generate_ml_predictions(
        self,
        prediction_times: List[int],
        target_sensitivity: float = 0.8,
        target_ppv: float = 0.3
    ) -> None:
        """
        Generate ML predictions at specified time points.
        
        Initial predictions use default noise parameters. Optimization
        happens later during intervention assignment when the assignment
        strategy is known.

        Parameters
        ----------
        prediction_times : List[int]
            Timesteps at which to generate predictions
        target_sensitivity : float, default=0.8
            Target sensitivity for ML model
        target_ppv : float, default=0.3
            Target positive predictive value
        """
        if not self._temporal_simulated:
            raise ValueError("Temporal evolution must be simulated first")

        # Initialize ML simulator
        self.ml_simulator = MLPredictionSimulator(
            target_sensitivity=target_sensitivity,
            target_ppv=target_ppv,
            random_seed=self.random_seed
        )

        # Store prediction times
        self.results.ml_prediction_times = prediction_times

        # Use default parameters for initial predictions
        # Optimization will happen later when assignment strategy is known
        if self._cached_ml_params is None:
            # Set default noise parameters
            self._cached_ml_params = {
                'correlation': 0.7,
                'scale': 0.3
            }
            import logging
            logging.debug(
                "Using default ML parameters for initial predictions. "
                "Optimization will occur during intervention assignment."
            )

        # Set parameters from cache
        self.ml_simulator.noise_correlation = (
            self._cached_ml_params['correlation']
        )
        self.ml_simulator.noise_scale = (
            self._cached_ml_params['scale']
        )

        # Generate predictions at each time point
        for pred_time in prediction_times:
            if pred_time + self.prediction_window > self.n_timesteps:
                warnings.warn(
                    f"Prediction window at time {pred_time} extends beyond "
                    f"simulation horizon. Skipping."
                )
                continue

            # Extract risk windows for this prediction time
            risk_windows = extract_risk_windows(
                self.results.temporal_risk_matrix,
                start_time=pred_time,
                window_length=self.prediction_window
            )

            # Integrate window risks
            integrated_risks = integrate_window_risk(
                risk_windows,
                timestep_duration=self.timestep_duration
            )

            # Generate true labels for this prediction window
            true_labels = self._generate_true_labels(pred_time)

            # Generate ML predictions using cached parameters
            ml_scores, ml_binary = self.ml_simulator.generate_predictions(
                true_labels,
                integrated_risks,
                noise_correlation=self.ml_simulator.noise_correlation or 0.7,
                noise_scale=self.ml_simulator.noise_scale or 0.3
            )

            # Store results
            self.results.ml_predictions[pred_time] = ml_scores
            self.results.ml_binary_predictions[pred_time] = ml_binary

        self._ml_predictions_generated = True

    def _ensure_cached_incident_matrix(self) -> None:
        """Generate and cache incident matrix for ML training if needed."""
        if self._cached_incident_matrix is not None:
            return

        # Generate incident matrix once using vectorized operations
        # Convert all risks to hazards at once
        annual_hazards = annual_risk_to_hazard(
            self.results.temporal_risk_matrix
        )

        # Convert to timestep probabilities (vectorized)
        timestep_probs = hazard_to_timestep_probability(
            annual_hazards, self.timestep_duration
        )

        # Generate all random draws at once
        random_draws = np.random.uniform(
            0, 1, (self.n_patients, self.n_timesteps)
        )

        # Generate incident matrix (vectorized)
        self._cached_incident_matrix = random_draws < timestep_probs

    def _generate_true_labels(self, pred_time: int) -> np.ndarray:
        """Generate true labels for prediction window."""
        if (not hasattr(self, 'results') or
                self.results.incident_matrix.size == 0):
            # Use cached incident matrix for ML training
            self._ensure_cached_incident_matrix()
            incident_matrix = self._cached_incident_matrix
            if incident_matrix is None:
                raise ValueError("Failed to generate cached incident matrix")
        else:
            # Use existing incident matrix
            incident_matrix = self.results.incident_matrix

        # Generate labels based on incidents in prediction window
        window_end = min(
            pred_time + self.prediction_window, self.n_timesteps
        )
        labels = np.any(
            incident_matrix[:, pred_time:window_end],
            axis=1
        )

        # Ensure we have at least one positive label for ML optimization
        # This is needed for proper threshold optimization
        labels_int = labels.astype(int)
        if np.sum(labels_int) == 0:
            # If no natural incidents, assign a small fraction as positive
            n_positive = max(1, int(0.05 * self.n_patients))
            positive_indices = np.random.choice(
                self.n_patients, n_positive, replace=False
            )
            labels_int[positive_indices] = 1

        return labels_int

    def _get_eligible_patients(self, current_time: int) -> np.ndarray:
        """
        Get patients eligible for intervention at current time.

        Parameters
        ----------
        current_time : int
            Current timestep

        Returns
        -------
        eligible_patients : np.ndarray
            Array of patient indices eligible for intervention
        """
        # Start with all patients
        all_patients = np.arange(self.n_patients)

        # Filter out patients currently under intervention
        eligible_mask = np.ones(self.n_patients, dtype=bool)

        for patient_idx, end_time in (
                self._active_intervention_end_times.items()):
            if current_time <= end_time:
                eligible_mask[patient_idx] = False

        return all_patients[eligible_mask]

    def _optimize_for_assignment_strategy(
        self,
        assignment_strategy: str,
        threshold: float,
        treatment_fraction: Optional[float],
        n_iterations: int = 20
    ) -> None:
        """
        Optimize ML parameters for the specific assignment strategy.
        This is the single optimization point in the simulation pipeline.

        Parameters
        ----------
        assignment_strategy : str
            Assignment strategy to optimize for
        threshold : float
            Threshold for ml_threshold strategy
        treatment_fraction : float, optional
            Fraction for top_k or random strategies
        n_iterations : int, default=20
            Number of optimization iterations
        """
        if (self.ml_simulator is None or
                self.results.ml_prediction_times is None):
            return

        # Skip optimization for random strategy
        if assignment_strategy == "random":
            import logging
            logging.debug(
                "Skipping ML optimization for random assignment strategy"
            )
            return

        # Use the first prediction time for optimization
        pred_time = self.results.ml_prediction_times[0]

        if (pred_time + self.prediction_window > self.n_timesteps):
            return  # Skip if prediction window extends beyond simulation

        import logging
        logging.debug(
            f"Optimizing ML parameters for {assignment_strategy} strategy "
            f"(iterations={n_iterations})"
        )

        # Extract risk windows and generate true labels
        risk_windows = extract_risk_windows(
            self.results.temporal_risk_matrix,
            start_time=pred_time,
            window_length=self.prediction_window
        )
        integrated_risks = integrate_window_risk(
            risk_windows,
            timestep_duration=self.timestep_duration
        )
        true_labels = self._generate_true_labels(pred_time)

        # Optimize with assignment strategy parameters
        optimized_params = self.ml_simulator.optimize_noise_parameters(
            true_labels,
            integrated_risks,
            n_iterations=n_iterations,
            assignment_strategy=assignment_strategy,
            assignment_threshold=threshold,
            assignment_fraction=treatment_fraction
        )

        # Update cached parameters
        self._cached_ml_params = optimized_params
        
        # Update ML simulator with optimized parameters
        self.ml_simulator.noise_correlation = optimized_params['correlation']
        self.ml_simulator.noise_scale = optimized_params['scale']
        
        logging.debug(
            f"ML optimization complete: correlation={optimized_params['correlation']:.3f}, "
            f"scale={optimized_params['scale']:.3f}"
        )

    def _regenerate_predictions_after_optimization(self) -> None:
        """
        Regenerate ML predictions using optimized parameters.
        Called after optimization to ensure predictions match the optimized model.
        """
        if (self.ml_simulator is None or
                self.results.ml_prediction_times is None or
                self._cached_ml_params is None):
            return

        import logging
        logging.debug("Regenerating ML predictions with optimized parameters")

        # Clear existing predictions
        self.results.ml_predictions.clear()
        self.results.ml_binary_predictions.clear()

        # Regenerate predictions at each time point with optimized parameters
        for pred_time in self.results.ml_prediction_times:
            if pred_time + self.prediction_window > self.n_timesteps:
                continue

            # Extract risk windows for this prediction time
            risk_windows = extract_risk_windows(
                self.results.temporal_risk_matrix,
                start_time=pred_time,
                window_length=self.prediction_window
            )

            # Integrate window risks
            integrated_risks = integrate_window_risk(
                risk_windows,
                timestep_duration=self.timestep_duration
            )

            # Generate true labels for this prediction window
            true_labels = self._generate_true_labels(pred_time)

            # Generate ML predictions using optimized parameters
            ml_scores, ml_binary = self.ml_simulator.generate_predictions(
                true_labels,
                integrated_risks,
                noise_correlation=self.ml_simulator.noise_correlation,
                noise_scale=self.ml_simulator.noise_scale
            )

            # Store results
            self.results.ml_predictions[pred_time] = ml_scores
            self.results.ml_binary_predictions[pred_time] = ml_binary

    def validate_no_re_enrollment(self) -> bool:
        """
        Validate that no patient was re-enrolled during active intervention.

        Returns
        -------
        is_valid : bool
            True if no re-enrollment violations detected
        """
        if not self._interventions_assigned:
            return True

        # Check each prediction time
        if self.results.ml_prediction_times is None:
            return True

        for i, pred_time in enumerate(self.results.ml_prediction_times):
            if pred_time not in self.results.intervention_times:
                continue

            current_assignments = set(
                self.results.intervention_times[pred_time]
            )

            # Check against all previous intervention assignments
            for j, prev_time in enumerate(
                    self.results.ml_prediction_times[:i]):
                if prev_time not in self.results.intervention_times:
                    continue

                prev_assignments = set(
                    self.results.intervention_times[prev_time]
                )

                # Calculate intervention end time for previous assignments
                if self.intervention_duration == -1:
                    prev_end_time = self.n_timesteps - 1
                else:
                    prev_end_time = (
                        prev_time + self.intervention_duration - 1
                    )

                # Check if current assignment overlaps with intervention
                if pred_time <= prev_end_time:
                    overlap = current_assignments.intersection(
                        prev_assignments
                    )
                    if overlap:
                        return False

        return True

    def validate_assignment_performance(
        self,
        assignment_strategy: str = "ml_threshold",
        threshold: float = 0.5,
        treatment_fraction: Optional[float] = None,
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """
        Validate that ML performance targets are met at assignment level.

        Parameters
        ----------
        assignment_strategy : str, default="ml_threshold"
            Assignment strategy to validate
        threshold : float, default=0.5
            Threshold for ml_threshold strategy
        treatment_fraction : float, optional
            Fraction for top_k or random strategies
        tolerance : float, default=0.1
            Acceptable deviation from targets

        Returns
        -------
        validation_results : dict
            Validation results with performance metrics and pass/fail status
        """
        if (not self._ml_predictions_generated or
                self.results.ml_prediction_times is None):
            return {"error": "ML predictions not generated"}

        results = {}

        for pred_time in self.results.ml_prediction_times:
            if pred_time not in self.results.ml_predictions:
                continue

            predictions = self.results.ml_predictions[pred_time]

            # Generate true labels for this prediction time
            true_labels = self._generate_true_labels(pred_time)

            # Apply assignment strategy to get binary predictions
            if assignment_strategy == "ml_threshold":
                binary_preds = (predictions >= threshold).astype(int)
            elif assignment_strategy == "top_k":
                if treatment_fraction is None:
                    treatment_fraction = 0.2
                k = int(len(predictions) * treatment_fraction)
                if k > 0:
                    top_indices = np.argsort(predictions)[-k:]
                    binary_preds = np.zeros_like(predictions, dtype=int)
                    binary_preds[top_indices] = 1
                else:
                    binary_preds = np.zeros_like(predictions, dtype=int)
            elif assignment_strategy == "random":
                # Random strategy doesn't use ML predictions
                continue
            else:
                continue

            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(
                true_labels, binary_preds
            ).ravel()

            if tp + fn > 0:
                sensitivity = tp / (tp + fn)
            else:
                sensitivity = 0.0

            if tp + fp > 0:
                ppv = tp / (tp + fp)
            else:
                ppv = 0.0

            # Check if within tolerance
            sens_target = (
                self.ml_simulator.target_sensitivity
                if self.ml_simulator else 0.8
            )
            ppv_target = (
                self.ml_simulator.target_ppv
                if self.ml_simulator else 0.3
            )

            sens_meets_target = abs(sensitivity - sens_target) <= tolerance
            ppv_meets_target = abs(ppv - ppv_target) <= tolerance

            results[f"time_{pred_time}"] = {
                "sensitivity": sensitivity,
                "ppv": ppv,
                "sensitivity_target": sens_target,
                "ppv_target": ppv_target,
                "sensitivity_meets_target": sens_meets_target,
                "ppv_meets_target": ppv_meets_target,
                "overall_pass": sens_meets_target and ppv_meets_target,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            }

        # Overall validation summary
        if results:
            all_pass = all(
                result["overall_pass"] for result in results.values()
                if isinstance(result, dict)
            )
            avg_sensitivity = np.mean([
                result["sensitivity"] for result in results.values()
                if isinstance(result, dict)
            ])
            avg_ppv = np.mean([
                result["ppv"] for result in results.values()
                if isinstance(result, dict)
            ])

            results["summary"] = {
                "all_times_pass": all_pass,
                "avg_sensitivity": avg_sensitivity,
                "avg_ppv": avg_ppv,
                "strategy": assignment_strategy
            }

        return results

    @log_call
    def assign_interventions(
        self,
        assignment_strategy: str = "ml_threshold",
        threshold: float = 0.5,
        treatment_fraction: Optional[float] = None,
        n_optimization_iterations: int = 20
    ) -> None:
        """
        Assign interventions based on ML predictions.

        Parameters
        ----------
        assignment_strategy : str, default="ml_threshold"
            Strategy for intervention assignment:
            - "ml_threshold": Use ML prediction threshold
            - "random": Random assignment
            - "top_k": Treat top k% by risk score
        threshold : float, default=0.5
            Threshold for ML-based assignment
        treatment_fraction : float, optional
            Fraction of patients to treat (for random/top_k strategies)
        n_optimization_iterations : int, default=20
            Number of optimization iterations for ML parameters
        """
        if not self._ml_predictions_generated:
            raise ValueError("ML predictions must be generated first")

        # Optimize ML parameters for the specific assignment strategy
        self._optimize_for_assignment_strategy(
            assignment_strategy, threshold, treatment_fraction,
            n_iterations=n_optimization_iterations
        )
        
        # Regenerate predictions with optimized parameters for consistency
        if assignment_strategy != "random":
            self._regenerate_predictions_after_optimization()

        # Initialize intervention tracking
        intervention_data = []
        # Initialize intervention times
        if self.results.ml_prediction_times is not None:
            intervention_times: Dict[int, List[int]] = {
                time: [] for time in self.results.ml_prediction_times
            }
            prediction_times = self.results.ml_prediction_times
        else:
            intervention_times = {}
            prediction_times = []

        # Assign interventions for each prediction time
        for pred_time in prediction_times:
            if pred_time not in self.results.ml_predictions:
                continue

            ml_scores = self.results.ml_predictions[pred_time]

            # Get eligible patients (not currently under intervention)
            eligible_patients = self._get_eligible_patients(pred_time)

            # Filter ML scores to only eligible patients
            eligible_scores = ml_scores[eligible_patients]

            # Determine intervention assignment for eligible patients only
            if assignment_strategy == "ml_threshold":
                eligible_treated = eligible_scores >= threshold
                treated_eligible_indices = np.where(eligible_treated)[0]
                treated_patient_indices = (
                    eligible_patients[treated_eligible_indices]
                )
            elif assignment_strategy == "random":
                if treatment_fraction is None:
                    treatment_fraction = 0.5
                n_eligible = len(eligible_patients)
                n_treated = int(n_eligible * treatment_fraction)
                if n_treated > 0:
                    treated_eligible_indices = np.random.choice(
                        n_eligible, n_treated, replace=False
                    )
                    treated_patient_indices = (
                        eligible_patients[treated_eligible_indices]
                    )
                else:
                    treated_patient_indices = np.array([], dtype=int)
            elif assignment_strategy == "top_k":
                if treatment_fraction is None:
                    treatment_fraction = 0.2
                n_eligible = len(eligible_patients)
                n_treated = int(n_eligible * treatment_fraction)
                if n_treated > 0:
                    top_eligible_indices = (
                        np.argsort(eligible_scores)[-n_treated:]
                    )
                    treated_patient_indices = (
                        eligible_patients[top_eligible_indices]
                    )
                else:
                    treated_patient_indices = np.array([], dtype=int)
            else:
                raise ValueError(
                    f"Unknown assignment strategy: {assignment_strategy}"
                )

            # Calculate intervention end time
            if self.intervention_duration == -1:
                # Full simulation duration
                intervention_end_time = self.n_timesteps - 1
            else:
                intervention_end_time = (
                    pred_time + self.intervention_duration - 1
                )

            # Update active intervention tracking
            for patient_idx in treated_patient_indices:
                self._active_intervention_end_times[patient_idx] = (
                    intervention_end_time
                )

            # Store intervention assignments
            intervention_times[pred_time] = treated_patient_indices.tolist()

            # Add to sparse matrix data for entire intervention duration
            for patient_idx in treated_patient_indices:
                for t in range(
                    pred_time, min(intervention_end_time + 1,
                                   self.n_timesteps)):
                    intervention_data.append((patient_idx, t, 1))

        # Create sparse intervention matrix
        if intervention_data:
            rows, cols, data = zip(*intervention_data)
            self.results.intervention_matrix = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(self.n_patients, self.n_timesteps)
            )
        else:
            self.results.intervention_matrix = sparse.csr_matrix(
                (self.n_patients, self.n_timesteps)
            )

        self.results.intervention_times = intervention_times
        self._interventions_assigned = True

    @log_call
    def simulate_incidents(
        self, generate_counterfactuals: bool = True
    ) -> None:
        """
        Simulate incidents with intervention effects.

        Parameters
        ----------
        generate_counterfactuals : bool, default=True
            Whether to generate counterfactual outcomes
        """
        if not self._interventions_assigned:
            raise ValueError("Interventions must be assigned first")

        # Initialize incident matrices
        self.results.incident_matrix = np.zeros(
            (self.n_patients, self.n_timesteps), dtype=bool
        )

        if generate_counterfactuals:
            self.results.counterfactual_incidents = np.zeros(
                (self.n_patients, self.n_timesteps), dtype=bool
            )

        # Reset incident generator
        self.incident_generator.reset()

        # Simulate incidents for each timestep
        for t in range(self.n_timesteps):
            # Get current risks
            current_risks = self.results.temporal_risk_matrix[:, t]

            # Check for active interventions
            active_interventions = (
                self.results.intervention_matrix[:, t].toarray().flatten()
            )
            intervention_mask = active_interventions > 0

            # Generate incidents with intervention effects
            incidents = self.incident_generator.generate_incidents(
                current_risks,
                intervention_mask=intervention_mask,
                intervention_effectiveness=(
                    self.intervention_effectiveness
                )
            )

            self.results.incident_matrix[:, t] = incidents

            # Generate counterfactual incidents (no intervention)
            if generate_counterfactuals:
                counterfactual_incidents = (
                    self.incident_generator.generate_incidents(
                        current_risks,
                        intervention_mask=None,
                        intervention_effectiveness=0.0
                    )
                )
                if self.results.counterfactual_incidents is not None:
                    self.results.counterfactual_incidents[
                        :, t
                    ] = counterfactual_incidents

        self._incidents_simulated = True
        self._compute_performance_metrics()

    def _compute_performance_metrics(self) -> None:
        """Compute intervention performance metrics."""
        # Intervention coverage
        total_interventions = self.results.intervention_matrix.nnz
        self.results.intervention_coverage = (
            total_interventions / (self.n_patients * self.n_timesteps)
        )

        # Incident reduction (if counterfactuals available)
        if self.results.counterfactual_incidents is not None:
            actual_incidents = np.sum(self.results.incident_matrix)
            counterfactual_incidents = np.sum(
                self.results.counterfactual_incidents
            )

            if counterfactual_incidents > 0:
                self.results.incident_reduction = (
                    (counterfactual_incidents - actual_incidents) /
                    counterfactual_incidents
                )

    @log_call
    def run_full_simulation(
        self,
        prediction_times: List[int],
        target_sensitivity: float = 0.8,
        target_ppv: float = 0.3,
        assignment_strategy: str = "ml_threshold",
        threshold: float = 0.5,
        generate_counterfactuals: bool = True,
        n_optimization_iterations: int = 20,
        treatment_fraction: Optional[float] = None,
        **kwargs
    ) -> SimulationResults:
        """
        Run complete simulation pipeline.

        Parameters
        ----------
        prediction_times : List[int]
            Timesteps at which to generate predictions
        target_sensitivity : float, default=0.8
            Target sensitivity for ML model
        target_ppv : float, default=0.3
            Target positive predictive value
        assignment_strategy : str, default="ml_threshold"
            Strategy for intervention assignment
        threshold : float, default=0.5
            Threshold for ML-based assignment
        generate_counterfactuals : bool, default=True
            Whether to generate counterfactual outcomes
        n_optimization_iterations : int, default=1000
            Number of optimization iterations for ML model
        treatment_fraction : float, optional
            Fraction of patients to treat (for random/top_k strategies)
        **kwargs
            Additional parameters for component methods

        Returns
        -------
        results : SimulationResults
            Complete simulation results
        """
        # Run simulation pipeline
        self.initialize_population(**kwargs)
        self.simulate_temporal_evolution()
        self.generate_ml_predictions(
            prediction_times,
            target_sensitivity=target_sensitivity,
            target_ppv=target_ppv
        )
        self.assign_interventions(
            assignment_strategy=assignment_strategy,
            threshold=threshold,
            treatment_fraction=treatment_fraction,
            n_optimization_iterations=n_optimization_iterations
        )
        self.simulate_incidents(
            generate_counterfactuals=generate_counterfactuals
        )

        return self.results

    def get_patient_trajectory(self, patient_id: int) -> Dict[str, np.ndarray]:
        """
        Get complete trajectory for a specific patient.

        Parameters
        ----------
        patient_id : int
            Patient index (0 to n_patients-1)

        Returns
        -------
        trajectory : dict
            Dictionary containing patient's complete trajectory
        """
        if patient_id < 0 or patient_id >= self.n_patients:
            raise ValueError(f"Patient ID {patient_id} out of range")

        trajectory = {
            'base_risk': self.results.patient_base_risks[patient_id],
            'temporal_risks': self.results.temporal_risk_matrix[patient_id, :],
            'incidents': self.results.incident_matrix[patient_id, :],
            'interventions': (
                self.results.intervention_matrix[
                    patient_id, :
                ].toarray().flatten()
            )
        }

        if self.results.counterfactual_incidents is not None:
            trajectory['counterfactual_incidents'] = (
                self.results.counterfactual_incidents[patient_id, :]
            )

        return trajectory

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the simulation.

        Returns
        -------
        stats : dict
            Dictionary containing key simulation statistics
        """
        stats: Dict[str, Any] = {
            'n_patients': self.n_patients,
            'n_timesteps': self.n_timesteps,
            'intervention_effectiveness': self.intervention_effectiveness,
            'intervention_coverage': self.results.intervention_coverage,
            'incident_reduction': self.results.incident_reduction,
            'total_incidents': int(np.sum(self.results.incident_matrix)),
            'total_interventions': int(self.results.intervention_matrix.nnz),
            'mean_base_risk': float(np.mean(self.results.patient_base_risks)),
            'std_base_risk': float(np.std(self.results.patient_base_risks))
        }

        if self.results.counterfactual_incidents is not None:
            stats['counterfactual_incidents'] = int(
                np.sum(self.results.counterfactual_incidents)
            )

        # Add ML performance metrics
        if self.results.ml_predictions:
            all_predictions = np.concatenate(
                list(self.results.ml_predictions.values())
            )
            stats['ml_prediction_stats'] = {
                'mean_score': float(np.mean(all_predictions)),
                'std_score': float(np.std(all_predictions)),
                'min_score': float(np.min(all_predictions)),
                'max_score': float(np.max(all_predictions))
            }

        return stats
