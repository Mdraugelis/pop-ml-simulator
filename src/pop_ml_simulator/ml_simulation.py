"""
ML simulation module for healthcare AI intervention modeling.

This module provides classes and functions for simulating machine learning
predictions with controlled performance characteristics, enabling realistic
evaluation of AI-guided interventions.
"""

import numpy as np
import pandas as pd  # type: ignore
from scipy import stats  # type: ignore
from sklearn.metrics import roc_auc_score, confusion_matrix  # type: ignore
from typing import Optional, Dict, List, Tuple, Union, Any
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.gridspec import GridSpec  # type: ignore
import seaborn as sns  # type: ignore
from utils.logging import log_call
from .risk_integration import integrate_window_risk, extract_risk_windows
from .hazard_modeling import (
    annual_risk_to_hazard, hazard_to_timestep_probability
)


@log_call
def calculate_theoretical_performance_bounds(
    prevalence: float,
    sensitivity_range: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Calculate theoretical bounds on PPV given sensitivity and prevalence.

    Uses Bayes' theorem to compute PPV for different specificity values.

    Parameters
    ----------
    prevalence : floa
        Population prevalence of the outcome
    sensitivity_range : np.ndarray, optional
        Range of sensitivity values to evaluate

    Returns
    -------
    sensitivity_range : np.ndarray
        Sensitivity values evaluated
    results : dic
        PPV values for different specificity levels

    Examples
    --------
    >>> sens_range, ppv_dict = calculate_theoretical_performance_bounds(0.1)
    >>> print(f"At 90% specificity, max PPV range: {ppv_dict['spec_0.9']}")
    """
    if sensitivity_range is None:
        sensitivity_range = np.linspace(0.1, 1.0, 50)

    specificities = [0.7, 0.8, 0.9, 0.95, 0.99]

    results = {}
    for spec in specificities:
        ppvs = []
        for sens in sensitivity_range:
            # Bayes' theorem:
            # PPV = (sens * prev) / (sens * prev + (1-spec) * (1-prev))
            ppv = (sens * prevalence) / (
                sens * prevalence + (1 - spec) * (1 - prevalence)
            )
            ppvs.append(ppv)
        results[f'spec_{spec}'] = ppvs

    return sensitivity_range, results


@log_call
def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float]:
    """
    Perform Hosmer-Lemeshow goodness-of-fit test for calibration.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted probabilities
    n_bins : int, default=10
        Number of bins for the tes

    Returns
    -------
    hl_statistic : floa
        Hosmer-Lemeshow chi-square statistic
    p_value : floa
        P-value for the test (> 0.05 indicates good calibration)

    Examples
    --------
    >>> hl_stat, p_val = hosmer_lemeshow_test(labels, predictions)
    >>> print(f"Calibration {'PASSED' if p_val > 0.05 else 'FAILED'}")
    """
    # Create bins based on predicted probabilities
    bins = pd.qcut(y_pred, n_bins, duplicates='drop')

    observed = pd.DataFrame({'true': y_true, 'pred': y_pred, 'bin': bins})
    grouped = observed.groupby('bin')

    # Calculate observed and expected for each bin
    obs_events = grouped['true'].sum()
    exp_events = grouped['pred'].sum()
    obs_non_events = grouped['true'].count() - obs_events
    exp_non_events = grouped['true'].count() - exp_events

    # Calculate Hosmer-Lemeshow statistic
    # Avoid division by zero
    valid_mask = (exp_events > 0) & (exp_non_events > 0)

    hl_statistic = (
        np.sum((obs_events[valid_mask] - exp_events[valid_mask])**2 /
               exp_events[valid_mask]) +
        np.sum((obs_non_events[valid_mask] - exp_non_events[valid_mask])**2 /
               exp_non_events[valid_mask])
    )

    # Degrees of freedom = n_bins - 2
    dof = len(grouped) - 2
    p_value = 1 - stats.chi2.cdf(hl_statistic, dof)

    return float(hl_statistic), float(p_value)


class MLPredictionSimulator:
    """
    Simulates ML predictions with controlled performance characteristics.

    Uses calibrated noise to create predictions that achieve targe
    sensitivity and PPV while maintaining realistic patterns.

    Parameters
    ----------
    target_sensitivity : float, default=0.8
        Target recall/sensitivity to achieve
    target_ppv : float, default=0.3
        Target precision/PPV to achieve
    calibration : str, default='sigmoid'
        Calibration function ('sigmoid' or 'linear')
    random_seed : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    noise_correlation : floa
        Optimized correlation between predictions and true risk
    noise_scale : floa
        Optimized noise scale parameter
    threshold : floa
        Optimal decision threshold

    Examples
    --------
    >>> simulator = MLPredictionSimulator(
    ...     target_sensitivity=0.8, target_ppv=0.3)
    >>> params = simulator.optimize_noise_parameters(
    ...     true_labels, risk_scores)
    >>> predictions, binary = simulator.generate_predictions(
    ...     true_labels, risk_scores)
    """

    def __init__(
        self,
        target_sensitivity: float = 0.8,
        target_ppv: float = 0.3,
        calibration: str = 'sigmoid',
        random_seed: Optional[int] = None
    ):
        """Initialize ML prediction simulator."""
        self.target_sensitivity = target_sensitivity
        self.target_ppv = target_ppv
        self.calibration = calibration
        self.random_seed = random_seed

        # Parameters to be optimized
        self.noise_correlation: Optional[float] = None
        self.noise_scale: Optional[float] = None
        self.threshold: Optional[float] = None

    def _apply_calibration(self, scores: np.ndarray) -> np.ndarray:
        """Apply calibration function to raw scores."""
        if self.calibration == 'sigmoid':
            # Sigmoid calibration with parameters
            return 1 / (1 + np.exp(-4 * (scores - 0.5)))
        else:
            # Simple clipping for linear calibration
            return np.clip(scores, 0, 1)

    @log_call
    def generate_predictions(
        self,
        true_labels: np.ndarray,
        risk_scores: np.ndarray,
        noise_correlation: float = 0.7,
        noise_scale: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ML predictions with calibrated noise.

        Parameters
        ----------
        true_labels : np.ndarray
            Binary ground truth labels
        risk_scores : np.ndarray
            True underlying risk scores
        noise_correlation : float, default=0.7
            How much predictions correlate with true risk
        noise_scale : float, default=0.3
            Amount of noise to add

        Returns
        -------
        predictions : np.ndarray
            Predicted probabilities
        binary_preds : np.ndarray
            Binary predictions using optimal threshold
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        n_patients = len(true_labels)

        # Start with true risk scores
        base_scores = risk_scores.copy()

        # Add correlated noise
        noise = np.random.normal(0, noise_scale, n_patients)

        # Combine with correlation
        noisy_scores = (noise_correlation * base_scores +
                        (1 - noise_correlation) * noise)

        # Add reduced label-dependent noise to avoid perfect separation
        # Reduced from 0.2/-0.1 to 0.05/-0.025 to prevent overconfidence
        label_noise = np.where(
            true_labels == 1,
            np.random.normal(0.05, 0.05, n_patients),
            np.random.normal(-0.025, 0.05, n_patients)
        )

        # Add risk-independent noise component for more variability
        independent_noise = np.random.normal(0, 0.1, n_patients)

        noisy_scores += (label_noise + independent_noise) * noise_scale

        # Apply calibration
        predictions = self._apply_calibration(noisy_scores)

        # Find optimal threshold for target performance
        threshold = self._find_optimal_threshold(true_labels, predictions)
        binary_preds = (predictions >= threshold).astype(int)

        return predictions, binary_preds

    def _find_optimal_threshold(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """Find threshold that best achieves target metrics."""
        thresholds = np.linspace(0.01, 0.99, 100)
        best_threshold = 0.5
        best_score = float('inf')

        for thresh in thresholds:
            binary_preds = (predictions >= thresh).astype(int)

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                true_labels, binary_preds
            ).ravel()

            if tp > 0:
                sensitivity = tp / (tp + fn)
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

                # Score based on distance from targets with penalties
                # Weight PPV more heavily and penalize extreme sensitivity
                sens_penalty = abs(sensitivity - self.target_sensitivity)
                ppv_penalty = abs(ppv - self.target_ppv) * 1.5  # Weight PPV

                # Add penalty for extreme sensitivity (>0.95)
                extreme_sens_penalty = 0
                if sensitivity > 0.95:
                    extreme_sens_penalty = (sensitivity - 0.95) * 10

                score = sens_penalty + ppv_penalty + extreme_sens_penalty

                if score < best_score:
                    best_score = score
                    best_threshold = thresh

        self.threshold = best_threshold
        return best_threshold

    @log_call
    def optimize_noise_parameters(
        self,
        true_labels: np.ndarray,
        risk_scores: np.ndarray,
        n_iterations: int = 20
    ) -> Dict[str, float]:
        """
        Optimize noise parameters to achieve target performance.

        Uses grid search to find noise_correlation and noise_scale
        that best achieve target sensitivity and PPV.

        Parameters
        ----------
        true_labels : np.ndarray
            Binary ground truth labels
        risk_scores : np.ndarray
            True underlying risk scores
        n_iterations : int, default=20
            Number of random seeds to average over

        Returns
        -------
        best_params : dic
            Optimized parameters with keys 'correlation' and 'scale'
        """
        correlations = np.linspace(0.5, 0.95, 10)
        scales = np.linspace(0.1, 0.5, 10)

        best_params = {'correlation': 0.7, 'scale': 0.3}
        best_score = float('inf')

        for corr in correlations:
            for scale in scales:
                # Average over multiple random seeds
                scores = []

                for seed in range(n_iterations):
                    self.random_seed = seed
                    _, binary = self.generate_predictions(
                        true_labels, risk_scores, corr, scale
                    )

                    # Calculate metrics
                    tn, fp, fn, tp = confusion_matrix(
                        true_labels, binary
                    ).ravel()

                    if tp > 0:
                        sensitivity = tp / (tp + fn)
                        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

                        score = (abs(sensitivity - self.target_sensitivity) +
                                 abs(ppv - self.target_ppv))
                        scores.append(score)

                avg_score = np.mean(scores) if scores else float('inf')

                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {'correlation': corr, 'scale': scale}

        self.noise_correlation = best_params['correlation']
        self.noise_scale = best_params['scale']

        return best_params

    @log_call
    def generate_temporal_predictions(
        self,
        temporal_risk_matrix: np.ndarray,
        prediction_start_time: int,
        prediction_window_length: int,
        timestep_duration: float = 1/52
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """
        Generate ML predictions using temporal risk matrix.

        This method extends the MLPredictionSimulator to work with temporal
        risk matrices, integrating risks over prediction windows.

        Parameters
        ----------
        temporal_risk_matrix : np.ndarray
            Complete temporal risk matrix, shape (n_patients, n_timesteps)
        prediction_start_time : in
            Starting timestep for the prediction window (0-indexed)
        prediction_window_length : in
            Length of the prediction window in timesteps
        timestep_duration : float, default=1/52
            Duration of each timestep as fraction of year

        Returns
        -------
        predictions : np.ndarray
            Predicted probabilities for each patien
        binary_predictions : np.ndarray
            Binary predictions using optimal threshold
        integration_info : dic
            Information about the integration process and correlations

        Examples
        --------
        >>> simulator = MLPredictionSimulator(target_sensitivity=0.8)
        >>> preds, binary, info = simulator.generate_temporal_predictions(
        ...     temporal_matrix, start_time=10, window_length=12
        ... )
        >>> print(f"Temporal correlation: {info['temporal_correlation']:.3f}")
        """
        # Extract risk windows for all patients
        risk_windows = extract_risk_windows(
            temporal_risk_matrix,
            prediction_start_time,
            prediction_window_length
        )

        # Integrate temporal risks over the window
        # Disable integration noise since we use proper temporal event gen
        integrated_risks = integrate_window_risk(
            risk_windows,
            timestep_duration=timestep_duration,
            add_integration_noise=False
        )

        # Generate realistic labels using proper temporal event generation
        n_patients, n_timesteps = temporal_risk_matrix.shape
        event_matrix, event_times = generate_temporal_events(
            temporal_risk_matrix,
            timestep_duration=timestep_duration,
            random_seed=self.random_seed
        )

        # Extract labels for the prediction window
        window_end = prediction_start_time + prediction_window_length
        true_labels = np.zeros(n_patients, dtype=int)

        for i in range(n_patients):
            if event_times[i] >= 0:  # Patient has an even
                # Check if event occurs within the prediction window
                if prediction_start_time <= event_times[i] < window_end:
                    true_labels[i] = 1

        # Optimize parameters for this specific temporal scenario
        if self.noise_correlation is None or self.noise_scale is None:
            params = self.optimize_noise_parameters(
                true_labels, integrated_risks, n_iterations=15
            )
        else:
            params = {
                'correlation': self.noise_correlation,
                'scale': self.noise_scale
            }

        # Generate predictions using integrated risks
        predictions, binary_predictions = self.generate_predictions(
            true_labels,
            integrated_risks,
            params['correlation'],
            params['scale']
        )

        # Calculate temporal correlation metrics
        n_patients, n_timesteps = temporal_risk_matrix.shape
        integration_info = {
            'window_start': prediction_start_time,
            'window_length': prediction_window_length,
            'mean_integrated_risk': float(np.mean(integrated_risks)),
            'std_integrated_risk': float(np.std(integrated_risks)),
            'optimization_correlation': params['correlation'],
            'optimization_scale': params['scale']
        }

        # Calculate correlation with temporal risk changes
        if prediction_window_length > 1:
            window_end = prediction_start_time + prediction_window_length
            risk_start = temporal_risk_matrix[:, prediction_start_time]
            risk_end = temporal_risk_matrix[:, window_end - 1]
            risk_change = risk_end - risk_start

            if np.std(risk_change) > 0 and np.std(predictions) > 0:
                temporal_correlation = float(
                    np.corrcoef(predictions, risk_change)[0, 1]
                )
            else:
                temporal_correlation = 0.0

            integration_info['temporal_correlation'] = temporal_correlation
            integration_info['mean_risk_change'] = float(np.mean(risk_change))
            integration_info['std_risk_change'] = float(np.std(risk_change))

        # Correlation between predictions and integrated risks
        if np.std(integrated_risks) > 0 and np.std(predictions) > 0:
            integration_correlation = float(
                np.corrcoef(predictions, integrated_risks)[0, 1]
            )
        else:
            integration_correlation = 0.0

        integration_info['integration_correlation'] = integration_correlation

        # Type annotation is correct - dict contains int and float values
        return predictions, binary_predictions, integration_info  # type:ignore


@log_call
def evaluate_threshold_based(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Union[float, int]]:
    """
    Evaluate predictions using a fixed threshold.

    Parameters
    ----------
    true_labels : np.ndarray
        Binary ground truth labels
    predictions : np.ndarray
        Predicted probabilities
    threshold : float, default=0.5
        Decision threshold

    Returns
    -------
    metrics : dic
        Comprehensive performance metrics

    Examples
    --------
    >>> metrics = evaluate_threshold_based(labels, predictions, 0.3)
    >>> print(f"PPV: {metrics['ppv']:.3f}, "
    ...       f"Sensitivity: {metrics['sensitivity']:.3f}")
    """
    binary_preds = (predictions >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()

    # Calculate metrics
    metrics = {
        'threshold': threshold,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        'n_flagged': int(tp + fp),
        'flag_rate': (tp + fp) / len(true_labels)
    }

    return metrics


@log_call
def evaluate_topk(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    k_percent: float = 10.0
) -> Dict[str, Union[float, int]]:
    """
    Evaluate predictions by selecting top K% highest risk.

    Common in resource-constrained settings where only a fixed
    percentage of patients can receive intervention.

    Parameters
    ----------
    true_labels : np.ndarray
        Binary ground truth labels
    predictions : np.ndarray
        Predicted probabilities
    k_percent : float, default=10.0
        Percentage of highest-risk patients to selec

    Returns
    -------
    metrics : dic
        Performance metrics for TopK selection

    Examples
    --------
    >>> metrics = evaluate_topk(labels, predictions, k_percent=5)
    >>> print(f"Top 5% PPV: {metrics['ppv']:.3f}")
    """
    n_patients = len(true_labels)
    k = int(n_patients * k_percent / 100)

    # Get indices of top k predictions
    top_k_indices = np.argsort(predictions)[-k:]

    # Create binary predictions
    binary_preds = np.zeros(n_patients, dtype=int)
    binary_preds[top_k_indices] = 1

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()

    metrics = {
        'k_percent': k_percent,
        'k_patients': k,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'ppv': tp / k if k > 0 else 0.0,  # PPV = tp / (all flagged)
        'lift': ((tp / k) / (np.sum(true_labels) / n_patients)
                 if k > 0 else 0.0),
        'min_score_flagged': (float(np.min(predictions[top_k_indices]))
                              if k > 0 else 0.0)
    }

    return metrics


@log_call
def optimize_alert_threshold(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    capacity_constraint: float = 0.1,
    fatigue_weight: float = 0.1
) -> Dict[str, Union[float, int, Dict]]:
    """
    Optimize alert threshold given capacity constraints and alert fatigue.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    true_labels : np.ndarray
        True outcomes (for simulation)
    capacity_constraint : float, default=0.1
        Maximum fraction of patients that can be flagged
    fatigue_weight : float, default=0.1
        Weight given to minimizing false positives

    Returns
    -------
    results : dic
        Optimization results including threshold and metrics

    Examples
    --------
    >>> result = optimize_alert_threshold(preds, labels,
    ...                                   capacity_constraint=0.05)
    >>> print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
    """
    n_patients = len(predictions)
    max_alerts = int(n_patients * capacity_constraint)

    # Find threshold that gives us exactly capacity_constraint alerts
    sorted_preds = np.sort(predictions)[::-1]
    if max_alerts < n_patients:
        capacity_threshold = sorted_preds[max_alerts - 1]
    else:
        capacity_threshold = 0.0

    # Evaluate performance at this threshold
    metrics = evaluate_threshold_based(
        true_labels, predictions, capacity_threshold)

    # Calculate utility score
    utility = metrics['tp'] - fatigue_weight * metrics['fp']

    return {
        'optimal_threshold': capacity_threshold,
        'n_alerts': max_alerts,
        'metrics': metrics,
        'utility': float(utility),
        'efficiency': metrics['ppv']  # Fraction of alerts useful
    }


@log_call
def analyze_risk_stratified_performance(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    risk_scores: np.ndarray,
    n_bins: int = 5
) -> pd.DataFrame:
    """
    Analyze model performance within risk strata.

    Helps identify if model performs better for high-risk vs
    low-risk patients.

    Parameters
    ----------
    true_labels : np.ndarray
        Binary ground truth labels
    predictions : np.ndarray
        Model predictions
    risk_scores : np.ndarray
        True underlying risk scores
    n_bins : int, default=5
        Number of risk strata to create

    Returns
    -------
    results_df : pd.DataFrame
        Performance metrics by risk stratum

    Examples
    --------
    >>> df = analyze_risk_stratified_performance(labels, preds, risks)
    >>> print(df[['risk_bin', 'prevalence', 'auc']].round(3))
    """
    # Create risk bins
    risk_bins = pd.qcut(risk_scores, n_bins,
                        labels=[f'Q{i+1}' for i in range(n_bins)])

    results = []

    for bin_label in risk_bins.categories:
        mask = risk_bins == bin_label

        if np.sum(mask) > 0:
            # Get subse
            subset_true = true_labels[mask]
            subset_preds = predictions[mask]
            subset_risks = risk_scores[mask]

            # Calculate optimal threshold for this subse
            if np.sum(subset_true) > 0:  # Has positive cases
                best_threshold = 0.5
                best_f1 = 0.0

                for thresh in np.linspace(0.1, 0.9, 20):
                    binary = (subset_preds >= thresh).astype(int)
                    if np.sum(binary) > 0:  # Has predictions
                        cm = confusion_matrix(subset_true, binary)
                        if cm.shape == (1, 1):
                            # Handle case where only one class is presen
                            if np.all(subset_true == 0):
                                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                            else:
                                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                        else:
                            tn, fp, fn, tp = cm.ravel()
                        f1 = (2 * tp / (2 * tp + fp + fn)
                              if (2 * tp + fp + fn) > 0 else 0.0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = thresh

                # Evaluate at best threshold
                metrics = evaluate_threshold_based(
                    subset_true, subset_preds, best_threshold
                )

                # Calculate AUC if possible
                if len(np.unique(subset_true)) > 1:
                    auc = roc_auc_score(subset_true, subset_preds)
                else:
                    auc = np.nan

                results.append({
                    'risk_bin': bin_label,
                    'n_patients': int(np.sum(mask)),
                    'prevalence': float(np.mean(subset_true)),
                    'mean_risk': float(np.mean(subset_risks)),
                    'mean_pred': float(np.mean(subset_preds)),
                    'auc': float(auc) if not np.isnan(auc) else np.nan,
                    'sensitivity': metrics['sensitivity'],
                    'ppv': metrics['ppv'],
                    'f1': metrics['f1'],
                    'optimal_threshold': best_threshold
                })

    return pd.DataFrame(results)


@log_call
def generate_temporal_events(
    temporal_risk_matrix: np.ndarray,
    timestep_duration: float = 1/52,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate temporal events using proper hazard-based simulation.

    This function generates events across the entire temporal period using
    the hazard modeling approach, ensuring epidemiologically valid even
    generation that respects the calibrated incidence rates.

    Parameters
    ----------
    temporal_risk_matrix : np.ndarray
        Complete temporal risk matrix, shape (n_patients, n_timesteps)
    timestep_duration : float, default=1/52
        Duration of each timestep as fraction of year
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    event_matrix : np.ndarray
        Binary matrix indicating if event occurred by each timestep
        Shape: (n_patients, n_timesteps)
    event_times : np.ndarray
        Time of first event for each patient (-1 if no event)
        Shape: (n_patients,)

    Notes
    -----
    This approach:
    1. Converts annual risks to hazard rates
    2. Converts hazards to timestep probabilities
    3. Uses binomial sampling at each timestep
    4. Tracks first event time (competing risk framework)
    5. Ensures proper temporal correlation with risk trajectories
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_patients, n_timesteps = temporal_risk_matrix.shape

    # Initialize event tracking
    event_matrix = np.zeros((n_patients, n_timesteps), dtype=bool)
    event_times = np.full(n_patients, -1, dtype=int)
    had_event = np.zeros(n_patients, dtype=bool)

    # Generate events timestep by timestep
    for t in range(n_timesteps):
        # Get current risks for all patients
        current_risks = temporal_risk_matrix[:, t]

        # Convert to hazards then to timestep probabilities
        current_hazards = annual_risk_to_hazard(current_risks)
        timestep_probs = hazard_to_timestep_probability(
            current_hazards, timestep_duration
        )

        # Only generate events for patients who haven't had one ye
        eligible_mask = ~had_event

        # Generate events using binomial sampling
        if np.sum(eligible_mask) > 0:
            # Ensure timestep_probs is ndarray for indexing
            if isinstance(timestep_probs, np.ndarray):
                eligible_probs = timestep_probs[eligible_mask]
            else:
                eligible_probs = np.array([timestep_probs])[eligible_mask]
            new_events = np.random.binomial(
                1, eligible_probs
            ).astype(bool)
        else:
            new_events = np.array([], dtype=bool)

        # Update event tracking for eligible patients
        eligible_indices = np.where(eligible_mask)[0]
        new_event_indices = eligible_indices[new_events]

        if len(new_event_indices) > 0:
            event_matrix[new_event_indices, t] = True
            event_times[new_event_indices] = t
            had_event[new_event_indices] = True

    return event_matrix, event_times


@log_call
def generate_temporal_ml_predictions(
    temporal_risk_matrix: np.ndarray,
    prediction_start_time: int,
    prediction_window_length: int,
    target_sensitivity: float = 0.8,
    target_ppv: float = 0.3,
    timestep_duration: float = 1/52,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Generate ML predictions using survival-based integration of temporal window
    risks.

    This function represents the core temporal-aware ML prediction approach
    that integrates risk trajectories over prediction windows instead of
    using static base risks.

    Parameters
    ----------
    temporal_risk_matrix : np.ndarray
        Complete temporal risk matrix, shape (n_patients, n_timesteps)
    prediction_start_time : in
        Starting timestep for the prediction window (0-indexed)
    prediction_window_length : in
        Length of the prediction window in timesteps
    target_sensitivity : float, default=0.8
        Target sensitivity to achieve
    target_ppv : float, default=0.3
        Target PPV to achieve
    timestep_duration : float, default=1/52
        Duration of each timestep as fraction of year
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    predictions : np.ndarray
        Predicted probabilities for each patien
    binary_predictions : np.ndarray
        Binary predictions using optimal threshold
    performance_metrics : dic
        Achieved performance metrics and correlation statistics

    Examples
    --------
    >>> # Generate temporal risk matrix
    >>> temporal_matrix = build_temporal_risk_matrix(base_risks, 52)
    >>> # Generate predictions for 12-week window starting at week 10
    >>> preds, binary, metrics = generate_temporal_ml_predictions(
    ...     temporal_matrix, prediction_start_time=10,
    ...     prediction_window_length=12
    ... )
    >>> print(f"Achieved sensitivity: {metrics['sensitivity']:.1%}")

    Notes
    -----
    This function combines temporal risk integration with ML prediction
    simulation to create realistic predictions that account for temporal
    risk evolution. It validates that predictions correlate with temporal
    risk changes and maintain target performance metrics.
    """
    n_patients, n_timesteps = temporal_risk_matrix.shape

    # Validate window bounds
    if prediction_start_time < 0:
        raise ValueError(
            f"prediction_start_time must be non-negative, "
            f"got {prediction_start_time}"
        )

    window_end = prediction_start_time + prediction_window_length
    if window_end > n_timesteps:
        raise ValueError(
            f"Prediction window extends beyond available timesteps. "
            f"Requested window [{prediction_start_time}, {window_end}), "
            f"but only have {n_timesteps} timesteps"
        )

    # Extract risk windows for all patients
    risk_windows = extract_risk_windows(
        temporal_risk_matrix,
        prediction_start_time,
        prediction_window_length
    )

    # Integrate temporal risks over the window
    # Disable integration noise since we use proper temporal event generation
    integrated_risks = integrate_window_risk(
        risk_windows,
        timestep_duration=timestep_duration,
        add_integration_noise=False
    )

    # Generate ground truth labels using proper temporal event generation
    # This ensures epidemiologically valid event rates and respects the
    # calibrated hazard model (Issues #56 and #57)
    event_matrix, event_times = generate_temporal_events(
        temporal_risk_matrix,
        timestep_duration=timestep_duration,
        random_seed=random_seed
    )

    # Extract labels for the prediction window
    # A patient has a positive label if they will have an event during
    # the prediction window
    window_end = prediction_start_time + prediction_window_length
    true_labels = np.zeros(n_patients, dtype=int)

    for i in range(n_patients):
        if event_times[i] >= 0:  # Patient has an even
            # Check if event occurs within the prediction window
            if prediction_start_time <= event_times[i] < window_end:
                true_labels[i] = 1

    # Create ML prediction simulator
    ml_simulator = MLPredictionSimulator(
        target_sensitivity=target_sensitivity,
        target_ppv=target_ppv,
        random_seed=random_seed
    )

    # Optimize noise parameters using integrated risks and generated labels
    optimization_params = ml_simulator.optimize_noise_parameters(
        true_labels, integrated_risks, n_iterations=15
    )

    # Generate ML predictions using survival-based integration of temporal
    # risks
    predictions, binary_predictions = ml_simulator.generate_predictions(
        true_labels,
        integrated_risks,
        optimization_params['correlation'],
        optimization_params['scale']
    )

    # Calculate performance metrics
    performance_metrics = evaluate_threshold_based(
        true_labels, predictions, ml_simulator.threshold or 0.5
    )

    # Calculate temporal correlation metrics
    # Correlation between predictions and temporal risk changes
    if prediction_window_length > 1:
        # Calculate risk change over the window
        risk_start = temporal_risk_matrix[:, prediction_start_time]
        risk_end = temporal_risk_matrix[:, window_end - 1]
        risk_change = risk_end - risk_start

        # Correlation between predictions and risk changes
        if np.std(risk_change) > 0 and np.std(predictions) > 0:
            temporal_correlation = float(
                np.corrcoef(predictions, risk_change)[0, 1]
            )
        else:
            temporal_correlation = 0.0
    else:
        temporal_correlation = np.nan

    # Correlation between predictions and integrated risks
    integrated_correlation = float(
        np.corrcoef(predictions, integrated_risks)[0, 1]
    )

    # Combine all metrics into a single dic
    # Copy numeric metrics
    all_metrics: Dict[str, object] = dict(performance_metrics)
    all_metrics.update({
        'temporal_correlation': temporal_correlation,
        'integrated_risk_correlation': integrated_correlation,
        'mean_integrated_risk': float(np.mean(integrated_risks)),
        'window_length': prediction_window_length,
        'optimization_correlation': optimization_params['correlation'],
        'optimization_scale': optimization_params['scale']
    })

    return predictions, binary_predictions, all_metrics


@log_call
def validate_temporal_sensitivity(
    temporal_risks: np.ndarray,
    predictions: np.ndarray,
    min_correlation: float = 0.5
) -> Dict[str, float]:
    """
    Validate that predictions are sensitive to temporal risk changes.

    This function ensures that ML predictions appropriately reflec
    temporal variations in patient risk, which is a key requiremen
    for realistic temporal ML simulation.

    Parameters
    ----------
    temporal_risks : np.ndarray
        Temporal risk values, shape (n_patients, n_timesteps)
    predictions : np.ndarray
        ML predictions for each patien
    min_correlation : float, default=0.5
        Minimum required correlation for validation to pass

    Returns
    -------
    validation_results : dic
        Correlation statistics and validation status

    Examples
    --------
    >>> validation = validate_temporal_sensitivity(
    ...     temporal_matrix, predictions
    ... )
    >>> if validation['passes_threshold']:
    ...     print("Temporal sensitivity validation PASSED")
    """
    n_patients, n_timesteps = temporal_risks.shape

    # Calculate various temporal sensitivity metrics
    correlations = []

    # Correlation with mean temporal risk
    mean_temporal_risk = np.mean(temporal_risks, axis=1)
    if np.std(mean_temporal_risk) > 0 and np.std(predictions) > 0:
        mean_correlation = float(
            np.corrcoef(predictions, mean_temporal_risk)[0, 1]
        )
    else:
        mean_correlation = 0.0

    correlations.append(mean_correlation)

    # Correlation with temporal variance (risk volatility)
    temporal_variance = np.var(temporal_risks, axis=1)
    if np.std(temporal_variance) > 0 and np.std(predictions) > 0:
        variance_correlation = float(
            np.corrcoef(predictions, temporal_variance)[0, 1]
        )
    else:
        variance_correlation = 0.0

    # Correlation with final risk value
    final_risk = temporal_risks[:, -1]
    if np.std(final_risk) > 0 and np.std(predictions) > 0:
        final_correlation = float(
            np.corrcoef(predictions, final_risk)[0, 1]
        )
    else:
        final_correlation = 0.0

    correlations.append(final_correlation)

    # Overall temporal sensitivity score
    max_correlation = max(correlations)
    mean_abs_correlation = np.mean(np.abs(correlations))

    validation_results = {
        'mean_risk_correlation': mean_correlation,
        'variance_correlation': variance_correlation,
        'final_risk_correlation': final_correlation,
        'max_correlation': max_correlation,
        'mean_abs_correlation': mean_abs_correlation,
        'passes_threshold': max_correlation >= min_correlation,
        'min_required_correlation': min_correlation
    }

    return validation_results


@log_call
def benchmark_temporal_ml_performance(
    temporal_matrix: np.ndarray,
    base_risks: np.ndarray,
    window_configs: List[Dict],
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Benchmark temporal ML performance across different window configurations.

    Compares temporal-aware predictions against static baseline approaches
    across various prediction windows using survival-based integration.

    Parameters
    ----------
    temporal_matrix : np.ndarray
        Complete temporal risk matrix, shape (n_patients, n_timesteps)
    base_risks : np.ndarray
        Static base risks for comparison baseline
    window_configs : list of dic
        List of window configurations to test. Each dict should contain:
        - 'start_time': in
        - 'window_length': in
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    benchmark_results : pd.DataFrame
        Performance comparison results

    Examples
    --------
    >>> configs = [
    ...     {'start_time': 10, 'window_length': 4},
    ...     {'start_time': 10, 'window_length': 12},
    ...     {'start_time': 20, 'window_length': 8}
    ... ]
    >>> results = benchmark_temporal_ml_performance(
    ...     temporal_matrix, base_risks, configs
    ... )
    >>> print(results[['config', 'temporal_auc', 'static_auc']].round(3))
    """
    results = []

    for i, config in enumerate(window_configs):
        config_id = f"config_{i+1}"

        try:
            # Generate temporal predictions
            temporal_preds, temporal_binary, temporal_metrics = \
                generate_temporal_ml_predictions(
                    temporal_matrix,
                    prediction_start_time=config['start_time'],
                    prediction_window_length=config['window_length'],
                    random_seed=random_seed
                )

            # Generate static baseline predictions for comparison
            # Create labels based on base risks for fair comparison
            if random_seed is not None:
                np.random.seed(random_seed + 1000)  # Different seed

            static_label_probs = base_risks * 1.2
            static_label_probs = np.clip(static_label_probs, 0.01, 0.95)
            static_labels = np.random.binomial(1, static_label_probs)

            static_simulator = MLPredictionSimulator(
                target_sensitivity=0.8,
                target_ppv=0.3,
                random_seed=random_seed
            )

            static_params = static_simulator.optimize_noise_parameters(
                static_labels, base_risks, n_iterations=10
            )

            static_preds, static_binary = (
                static_simulator.generate_predictions(
                    static_labels,
                    base_risks,
                    static_params['correlation'],
                    static_params['scale']
                )
            )

            static_metrics = evaluate_threshold_based(
                static_labels, static_preds, static_simulator.threshold or 0.5
            )

            # Calculate AUCs if possible
            temporal_auc = np.nan
            static_auc = np.nan

            # For temporal predictions
            if len(np.unique(temporal_binary)) > 1:
                # Create labels from temporal predictions for AUC calculation
                # Use binary predictions as proxy
                temp_labels = temporal_binary
                if len(np.unique(temp_labels)) > 1:
                    temporal_auc = roc_auc_score(temp_labels, temporal_preds)

            # For static predictions
            if len(np.unique(static_labels)) > 1:
                static_auc = roc_auc_score(static_labels, static_preds)

            # Compile results
            result = {
                'config_id': config_id,
                'start_time': config['start_time'],
                'window_length': config['window_length'],
                'temporal_sensitivity': temporal_metrics['sensitivity'],
                'temporal_ppv': temporal_metrics['ppv'],
                'temporal_f1': temporal_metrics['f1'],
                'temporal_auc': temporal_auc,
                'static_sensitivity': static_metrics['sensitivity'],
                'static_ppv': static_metrics['ppv'],
                'static_f1': static_metrics['f1'],
                'static_auc': static_auc,
                'temporal_correlation': (
                    temporal_metrics['temporal_correlation']
                ),
                'integrated_correlation': (
                    temporal_metrics['integrated_risk_correlation']
                ),
                'mean_integrated_risk': (
                    temporal_metrics['mean_integrated_risk']
                )
            }

            results.append(result)

        except Exception as e:
            # Log failed configuration but continue
            result = {
                'config_id': config_id,
                'start_time': config['start_time'],
                'window_length': config['window_length'],
                'error': str(e)
            }
            results.append(result)

    return pd.DataFrame(results)


@log_call
def analyze_patient_journey_enhanced(
    patient_id: int,
    temporal_risk_matrix: np.ndarray,
    prediction_start: int,
    window_length: int,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    event_times: np.ndarray,
    integrated_risks: Optional[np.ndarray] = None,
    base_risks: Optional[np.ndarray] = None,
    intervention_mask: Optional[np.ndarray] = None,
    timestep_duration: float = 1/52,
    figsize: Tuple[int, int] = (20, 16),
    show_plot: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive 8-panel visualization for a single patient's
    journey.

    This function creates a detailed visualization showing how a patient's risk
    evolves over time, how the ML model makes predictions, and where the
    patient falls in the confusion matrix.

    Parameters
    ----------
    patient_id : in
        Index of the patient to analyze
    temporal_risk_matrix : np.ndarray
        Complete temporal risk matrix, shape (n_patients, n_timesteps)
    prediction_start : in
        Starting timestep for the prediction window
    window_length : in
        Length of the prediction window in timesteps
    predictions : np.ndarray
        ML predictions for all patients
    true_labels : np.ndarray
        True binary outcomes for all patients
    event_times : np.ndarray
        Time of first event for each patient (-1 if no event)
    integrated_risks : np.ndarray, optional
        Integrated risks over prediction window
    base_risks : np.ndarray, optional
        Static base risks for all patients
    intervention_mask : np.ndarray, optional
        Boolean mask indicating which patients received intervention
    timestep_duration : float, default=1/52
        Duration of each timestep as fraction of year
    figsize : tuple, default=(20, 16)
        Figure size for the visualization
    show_plot : bool, default=True
        Whether to display the plo

    Returns
    -------
    analysis_results : dic
        Dictionary containing:
        - patient_id: in
        - classification: str (TP/FP/TN/FN)
        - base_risk: floa
        - integrated_risk: floa
        - ml_prediction: floa
        - true_label: in
        - event_time: in
        - risk_trajectory: np.ndarray
        - window_risks: np.ndarray
        - cumulative_hazard: floa
        - figure: matplotlib.figure.Figure (if show_plot=False)

    Examples
    --------
    >>> analysis = analyze_patient_journey_enhanced(
    ...     patient_id=42,
    ...     temporal_risk_matrix=temporal_matrix,
    ...     prediction_start=20,
    ...     window_length=12,
    ...     predictions=ml_predictions,
    ...     true_labels=labels,
    ...     event_times=event_times
    ... )
    >>> print(f"Patient classified as: {analysis['classification']}")
    """
    n_patients, n_timesteps = temporal_risk_matrix.shape

    # Extract patient data
    patient_risks = temporal_risk_matrix[patient_id, :]
    patient_pred = predictions[patient_id]
    patient_label = true_labels[patient_id]
    patient_event_time = event_times[patient_id]

    # Determine classification
    pred_binary = 1 if patient_pred >= 0.5 else 0  # Default threshold
    if patient_label == 1 and pred_binary == 1:
        classification = "True Positive"
        class_color = '#2E7D32'  # Dark green
    elif patient_label == 0 and pred_binary == 1:
        classification = "False Positive"
        class_color = '#F57C00'  # Orange
    elif patient_label == 0 and pred_binary == 0:
        classification = "True Negative"
        class_color = '#1976D2'  # Blue
    else:
        classification = "False Negative"
        class_color = '#D32F2F'  # Red

    # Calculate integrated risk if not provided
    if integrated_risks is None:
        window_risks = patient_risks[
            prediction_start:prediction_start + window_length
        ]
        timestep_duration = 1/52
        clipped_risks = np.clip(window_risks, 1e-10, 1 - 1e-10)
        timestep_hazards = -np.log(1 - clipped_risks) / timestep_duration
        cumulative_hazard = np.sum(timestep_hazards) * timestep_duration
        integrated_risk = 1 - np.exp(-cumulative_hazard)
    else:
        integrated_risk = integrated_risks[patient_id]
        window_risks = patient_risks[
            prediction_start:prediction_start + window_length
        ]
        # Calculate cumulative hazard for display
        clipped_risks = np.clip(window_risks, 1e-10, 1 - 1e-10)
        timestep_hazards = -np.log(1 - clipped_risks) / timestep_duration
        cumulative_hazard = np.sum(timestep_hazards) * timestep_duration

    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(
        f'Patient {patient_id} - AI Quadrant Analysis - {classification}',
        fontsize=20, fontweight='bold', color=class_color
    )

    # Panel 1: Full year risk trajectory
    ax1 = fig.add_subplot(gs[0, :2])
    weeks = np.arange(n_timesteps)
    ax1.plot(weeks, patient_risks, 'b-', linewidth=2, label='Risk Level')

    # Mark prediction window
    ax1.axvspan(prediction_start, prediction_start + window_length,
                alpha=0.3, color='yellow', label='Prediction Window')

    # Mark event if it occurred
    if patient_event_time >= 0:
        ax1.plot(patient_event_time, patient_risks[patient_event_time],
                 'r*', markersize=20, label='Event Occurred')

    # Add base risk line if available
    if base_risks is not None:
        ax1.axhline(base_risks[patient_id], color='green', linestyle=':',
                    alpha=0.7,
                    label=f'Base Risk: {base_risks[patient_id]:.3f}')

    ax1.set_xlabel('Week')
    ax1.set_ylabel('Risk Level')
    ax1.set_title('Full Year Risk Trajectory')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(patient_risks.max() * 1.1, 0.1))

    # Panel 2: Cumulative hazard and event generation
    ax2 = fig.add_subplot(gs[0, 2:])

    # Calculate cumulative hazard over time
    cumulative_hazards = []
    for t in range(1, n_timesteps + 1):
        risks_up_to_t = patient_risks[:t]
        clipped = np.clip(risks_up_to_t, 1e-10, 1 - 1e-10)
        hazards = -np.log(1 - clipped) / timestep_duration
        cum_hazard = np.sum(hazards) * timestep_duration
        cumulative_hazards.append(cum_hazard)

    ax2.plot(weeks, cumulative_hazards, 'purple', linewidth=2)
    ax2.fill_between(weeks, 0, cumulative_hazards, alpha=0.3, color='purple')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Cumulative Hazard')
    ax2.set_title('Cumulative Hazard Over Time')
    ax2.grid(True, alpha=0.3)

    # Add event probability on secondary y-axis
    ax2_twin = ax2.twinx()
    event_probs = 1 - np.exp(-np.array(cumulative_hazards))
    ax2_twin.plot(weeks, event_probs, 'g--', linewidth=2, alpha=0.7)
    ax2_twin.set_ylabel('Event Probability', color='g')
    ax2_twin.tick_params(axis='y', labelcolor='g')

    # Panel 3: Prediction window detail
    ax3 = fig.add_subplot(gs[1, :2])
    window_weeks = np.arange(window_length)
    ax3.bar(window_weeks, window_risks, alpha=0.7, color='skyblue')
    ax3.plot(window_weeks, window_risks, 'ro-', linewidth=2, markersize=8)

    # Add integrated risk line
    ax3.axhline(integrated_risk, color='red', linestyle='--',
                linewidth=2, label=f'Integrated Risk: {integrated_risk:.3f}')

    ax3.set_xlabel('Week in Window')
    ax3.set_ylabel('Risk Level')
    ax3.set_title('Prediction Window Risk Details')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add statistics
    stats_text = (
        f'Mean: {np.mean(window_risks):.3f}\n'
        f'Std: {np.std(window_risks):.3f}\n'
        f'Min: {np.min(window_risks):.3f}\n'
        f'Max: {np.max(window_risks):.3f}'
    )
    ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
             ha='right', va='top', bbox=dict(boxstyle='round',
                                             facecolor='wheat', alpha=0.8))

    # Panel 4: Risk components breakdown
    ax4 = fig.add_subplot(gs[1, 2:])
    components = ['Base Risk', 'Mean Window\nRisk',
                  'Integrated\nRisk', 'ML\nPrediction']
    values = [
        base_risks[patient_id] if base_risks is not None else np.mean(
            patient_risks),
        np.mean(window_risks),
        integrated_risk,
        patient_pred
    ]
    colors = ['green', 'orange', 'red', 'purple']

    bars = ax4.bar(components, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Risk/Prediction Value')
    ax4.set_title('Risk Components Comparison')
    ax4.set_ylim(0, max(values) * 1.2)

    # Add value labels
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Panel 5: Event timeline
    ax5 = fig.add_subplot(gs[2, :])

    # Create timeline
    ax5.axhline(0, color='black', linewidth=2)
    ax5.set_xlim(-5, n_timesteps + 5)
    ax5.set_ylim(-1, 1)

    # Mark key time points
    ax5.scatter(0, 0, s=100, c='green', marker='o', zorder=5)
    ax5.text(0, 0.1, 'Start', ha='center', va='bottom')

    ax5.scatter(prediction_start, 0, s=150, c='blue', marker='s', zorder=5)
    ax5.text(prediction_start, -0.1, 'Prediction\nStart',
             ha='center', va='top')

    ax5.scatter(prediction_start + window_length, 0, s=150, c='purple',
                marker='s', zorder=5)
    ax5.text(prediction_start + window_length, 0.1, 'Window\nEnd',
             ha='center', va='bottom')

    # Mark event if occurred
    if patient_event_time >= 0:
        ax5.scatter(patient_event_time, 0, s=300, c='red',
                    marker='*', zorder=6)
        ax5.text(patient_event_time, -0.2, 'EVENT',
                 ha='center', va='top', fontweight='bold', color='red')

    # Mark intervention if applicable
    if intervention_mask is not None and intervention_mask[patient_id]:
        ax5.scatter(prediction_start + window_length + 1, 0, s=200,
                    c='darkgreen', marker='^', zorder=5)
        ax5.text(prediction_start + window_length + 1, 0.1,
                 'Intervention', ha='center', va='bottom', color='darkgreen')

    ax5.set_xlabel('Week')
    ax5.set_title('Event Timeline')
    ax5.set_yticks([])
    ax5.grid(True, axis='x', alpha=0.3)

    # Panel 6: Confusion matrix position
    ax6 = fig.add_subplot(gs[2, 2:])

    # Create confusion matrix visualization
    cm_data = np.array([[0.5, 0.2], [0.1, 0.2]])  # Example proportions

    # Highlight patient's position
    if classification == "True Positive":
        cm_data[1, 1] = 1.0
    elif classification == "False Positive":
        cm_data[1, 0] = 1.0
    elif classification == "True Negative":
        cm_data[0, 0] = 1.0
    else:  # False Negative
        cm_data[0, 1] = 1.0

    sns.heatmap(cm_data, annot=False, cmap='Blues', ax=ax6,
                cbar=False, vmin=0, vmax=1)

    # Add labels
    ax6.set_xticklabels(['Predicted\nNegative', 'Predicted\nPositive'])
    ax6.set_yticklabels(['True\nNegative', 'True\nPositive'])
    ax6.set_title(f'Confusion Matrix Position: {classification}')

    # Add quadrant labels
    quadrant_labels = [
        ('TN', 0.5, 0.5),
        ('FP', 1.5, 0.5),
        ('FN', 0.5, 1.5),
        ('TP', 1.5, 1.5)
    ]

    for label, x, y in quadrant_labels:
        color = 'white' if (label == classification[:2]) else 'gray'
        weight = 'bold' if (label == classification[:2]) else 'normal'
        ax6.text(x, y, label, ha='center', va='center',
                 fontsize=16, color=color, fontweight=weight)

    # Panel 7: Risk evolution patterns
    ax7 = fig.add_subplot(gs[3, :2])

    # Calculate risk changes
    risk_changes = np.diff(patient_risks)
    ax7.plot(weeks[1:], risk_changes, 'b-', linewidth=2, alpha=0.7)
    ax7.fill_between(weeks[1:], 0, risk_changes,
                     where=(risk_changes > 0).tolist(),
                     color='red', alpha=0.3, label='Risk Increase')
    ax7.fill_between(weeks[1:], 0, risk_changes,
                     where=(risk_changes < 0).tolist(),
                     color='green', alpha=0.3, label='Risk Decrease')

    ax7.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax7.set_xlabel('Week')
    ax7.set_ylabel('Risk Change')
    ax7.set_title('Week-to-Week Risk Evolution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Panel 8: Patient summary statistics
    ax8 = fig.add_subplot(gs[3, 2:])
    ax8.axis('off')

    # Calculate percentiles
    risk_percentile = stats.percentileofscore(
        temporal_risk_matrix.flatten(), integrated_risk
    )
    pred_percentile = stats.percentileofscore(predictions, patient_pred)

    # Format base risk value
    base_risk_str = (f"{base_risks[patient_id]:.3f}"
                     if base_risks is not None else "N/A")
    summary_text = f"""Patient {patient_id} Summary:

Classification: {classification}
Base Risk: {base_risk_str}
Risk Percentile: {risk_percentile:.1f}%

Temporal Statistics:
- Mean Risk (full year): {np.mean(patient_risks):.3f}
- Risk Volatility (std): {np.std(patient_risks):.3f}
- Risk Trend: {"Increasing" if patient_risks[-1] > patient_risks[0] else "Decreasing"}

Prediction Window:
- Start Week: {prediction_start}
- Window Length: {window_length} weeks
- Mean Window Risk: {np.mean(window_risks):.3f}
- Integrated Risk: {integrated_risk:.3f}
- Cumulative Hazard: {cumulative_hazard:.3f}

ML Prediction:
- Prediction Score: {patient_pred:.3f}
- Prediction Percentile: {pred_percentile:.1f}%
- True Label: {patient_label}
- Event Time: {f"Week {patient_event_time}"
               if patient_event_time >= 0 else "No Event"}

Clinical Impact:
- Intervention: {"Yes" if intervention_mask is not None and
                 intervention_mask[patient_id] else "No"}
- Outcome: {classification}
"""

    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
             fontsize=12, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Prepare return dictionary
    analysis_results = {
        'patient_id': patient_id,
        'classification': classification,
        'base_risk': (base_risks[patient_id]
                      if base_risks is not None else None),
        'integrated_risk': integrated_risk,
        'ml_prediction': patient_pred,
        'true_label': patient_label,
        'event_time': patient_event_time,
        'risk_trajectory': patient_risks,
        'window_risks': window_risks,
        'cumulative_hazard': cumulative_hazard,
        'risk_percentile': risk_percentile,
        'prediction_percentile': pred_percentile
    }

    if show_plot:
        plt.show()
    else:
        analysis_results['figure'] = fig
        plt.close(fig)

    return analysis_results


@log_call
def plot_ai_quadrant(
    true_risks: np.ndarray,
    predictions: np.ndarray,
    true_events: np.ndarray,
    intervention_mask: Optional[np.ndarray] = None,
    risk_threshold: float = 0.1,
    prediction_threshold: Optional[float] = None,
    title: str = "AI Intervention Quadrant Analysis",
    figsize: Tuple[int, int] = (12, 10),
    show_stats: bool = True,
    highlight_interventions: bool = True,
    show_plot: bool = True
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create AI Quadrant visualization for intervention analysis.

    This function creates a 2x2 scatter plot showing the relationship between
    true risk and AI predictions, with quadrants representing differen
    intervention scenarios:
    - High Risk + AI Flagged → High Priority (should intervene)
    - High Risk + AI Missed → Missed Opportunities
    - Low Risk + AI Flagged → False Alarms
    - Low Risk + AI Correct → Correct Rejections

    Parameters
    ----------
    true_risks : np.ndarray
        True underlying risks for each patien
    predictions : np.ndarray
        AI prediction scores for each patien
    true_events : np.ndarray
        Binary array indicating which patients had events
    intervention_mask : np.ndarray, optional
        Boolean mask indicating which patients received intervention
    risk_threshold : float, default=0.1
        Threshold for defining "high risk" patients
    prediction_threshold : float, optional
        Threshold for AI predictions. If None, uses median of predictions
    title : str, default="AI Intervention Quadrant Analysis"
        Title for the plo
    figsize : tuple, default=(12, 10)
        Figure size for the visualization
    show_stats : bool, default=True
        Whether to display quadrant statistics
    highlight_interventions : bool, default=True
        Whether to highlight patients who received interventions
    show_plot : bool, default=True
        Whether to display the plo

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure objec
    stats : dic
        Dictionary containing quadrant statistics and metrics

    Examples
    --------
    >>> fig, stats = plot_ai_quadrant(
    ...     true_risks=integrated_risks,
    ...     predictions=ml_predictions,
    ...     true_events=event_occurred,
    ...     intervention_mask=intervention_patients
    ... )
    >>> print(f"High priority patients: {stats['high_priority']['count']}")
    """
    # Determine prediction threshold if not provided
    if prediction_threshold is None:
        prediction_threshold = np.median(predictions)

    # Create quadrant classifications
    high_risk = true_risks >= risk_threshold
    ai_flagged = predictions >= prediction_threshold

    # Define quadrants
    high_priority = high_risk & ai_flagged      # High risk + AI flagged
    missed_opportunities = high_risk & ~ai_flagged  # High risk + AI missed
    false_alarms = ~high_risk & ai_flagged     # Low risk + AI flagged
    correct_rejections = ~high_risk & ~ai_flagged  # Low risk + AI correc

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for outcomes (if needed later)
    # event_colors = {
    #     'No Event': '#1976D2',      # Blue
    #     'Event': '#D32F2F'          # Red
    # }

    # Plot points by quadrant and event status
    quadrants = [
        ('High Priority', high_priority, '#2E7D32'),      # Dark green
        ('Missed Opportunities', missed_opportunities, '#FF5722'),
        # Deep orange
        ('False Alarms', false_alarms, '#FF9800'),        # Orange
        ('Correct Rejections', correct_rejections, '#4CAF50')
        # Green
    ]

    # Plot each quadran
    for quad_name, mask, quad_color in quadrants:
        if np.sum(mask) > 0:
            # Separate by event status
            quad_no_event = mask & (true_events == 0)
            quad_event = mask & (true_events == 1)

            # Plot patients without events
            if np.sum(quad_no_event) > 0:
                ax.scatter(
                    true_risks[quad_no_event],
                    predictions[quad_no_event],
                    c=quad_color,
                    alpha=0.6,
                    s=60,
                    marker='o',
                    edgecolors='white',
                    linewidth=0.5,
                    label=f'{quad_name} (No Event)'
                )

            # Plot patients with events (use different marker)
            if np.sum(quad_event) > 0:
                ax.scatter(
                    true_risks[quad_event],
                    predictions[quad_event],
                    c=quad_color,
                    alpha=0.8,
                    s=100,
                    marker='*',
                    edgecolors='black',
                    linewidth=1,
                    label=f'{quad_name} (Event)'
                )

    # Highlight interventions if requested
    if highlight_interventions and intervention_mask is not None:
        intervention_patients = intervention_mask == 1
        if np.sum(intervention_patients) > 0:
            ax.scatter(
                true_risks[intervention_patients],
                predictions[intervention_patients],
                facecolors='none',
                edgecolors='purple',
                linewidth=3,
                s=150,
                marker='s',
                label='Received Intervention'
            )

    # Add quadrant boundary lines
    ax.axhline(y=prediction_threshold, color='black', linestyle='--',
               linewidth=2, alpha=0.7, label='Prediction Threshold')
    ax.axvline(x=risk_threshold, color='black', linestyle='--',
               linewidth=2, alpha=0.7, label='Risk Threshold')

    # Calculate quadrant centers for labels
    x_low = ax.get_xlim()[0] + (risk_threshold - ax.get_xlim()[0]) / 2
    x_high = risk_threshold + (ax.get_xlim()[1] - risk_threshold) / 2
    y_low = ax.get_ylim()[0] + (prediction_threshold - ax.get_ylim()[0]) / 2
    y_high = prediction_threshold + \
        (ax.get_ylim()[1] - prediction_threshold) / 2

    # Add quadrant background colors
    ax.axhspan(prediction_threshold, ax.get_ylim()[1],
               xmin=0, xmax=(risk_threshold - ax.get_xlim()[0]) /
               (ax.get_xlim()[1] - ax.get_xlim()[0]),
               alpha=0.1, color='orange')  # False Alarms
    ax.axhspan(prediction_threshold, ax.get_ylim()[1],
               xmin=(risk_threshold - ax.get_xlim()[0]) /
               (ax.get_xlim()[1] - ax.get_xlim()[0]), xmax=1,
               alpha=0.1, color='green')   # High Priority
    ax.axhspan(ax.get_ylim()[0], prediction_threshold,
               xmin=0, xmax=(risk_threshold - ax.get_xlim()[0]) /
               (ax.get_xlim()[1] - ax.get_xlim()[0]),
               alpha=0.1, color='lightgreen')  # Correct Rejections
    ax.axhspan(ax.get_ylim()[0], prediction_threshold,
               xmin=(risk_threshold - ax.get_xlim()[0]) /
               (ax.get_xlim()[1] - ax.get_xlim()[0]), xmax=1,
               alpha=0.1, color='red')     # Missed Opportunities

    # Add quadrant text labels
    label_props: Dict[str, Any] = dict(
        fontsize=12, fontweight='bold', ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='white', alpha=0.8))

    # Only add labels if we have reasonable space
    if (ax.get_xlim()[1] - ax.get_xlim()[0] > 0.05 and
            ax.get_ylim()[1] - ax.get_ylim()[0] > 0.05):
        ax.text(x_low, y_high, 'False\nAlarms', **label_props)
        ax.text(x_high, y_high, 'High\nPriority', **label_props)
        ax.text(x_low, y_low, 'Correct\nRejections', **label_props)
        ax.text(x_high, y_low, 'Missed\nOpportunities', **label_props)

    # Customize plo
    ax.set_xlabel('True Risk', fontsize=12)
    ax.set_ylabel('AI Prediction Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Calculate statistics
    stats = {
        'high_priority': {
            'count': np.sum(high_priority),
            'event_rate': (np.mean(true_events[high_priority])
                           if np.sum(high_priority) > 0 else 0),
            'mean_risk': (np.mean(true_risks[high_priority])
                          if np.sum(high_priority) > 0 else 0),
            'mean_prediction': (np.mean(predictions[high_priority])
                                if np.sum(high_priority) > 0 else 0)
        },
        'missed_opportunities': {
            'count': np.sum(missed_opportunities),
            'event_rate': (np.mean(true_events[missed_opportunities])
                           if np.sum(missed_opportunities) > 0 else 0),
            'mean_risk': (np.mean(true_risks[missed_opportunities])
                          if np.sum(missed_opportunities) > 0 else 0),
            'mean_prediction': (np.mean(predictions[missed_opportunities])
                                if np.sum(missed_opportunities) > 0 else 0)
        },
        'false_alarms': {
            'count': np.sum(false_alarms),
            'event_rate': (np.mean(true_events[false_alarms])
                           if np.sum(false_alarms) > 0 else 0),
            'mean_risk': (np.mean(true_risks[false_alarms])
                          if np.sum(false_alarms) > 0 else 0),
            'mean_prediction': (np.mean(predictions[false_alarms])
                                if np.sum(false_alarms) > 0 else 0)
        },
        'correct_rejections': {
            'count': np.sum(correct_rejections),
            'event_rate': (np.mean(true_events[correct_rejections])
                           if np.sum(correct_rejections) > 0 else 0),
            'mean_risk': (np.mean(true_risks[correct_rejections])
                          if np.sum(correct_rejections) > 0 else 0),
            'mean_prediction': (np.mean(predictions[correct_rejections])
                                if np.sum(correct_rejections) > 0 else 0)
        },
        'thresholds': {
            'risk_threshold': risk_threshold,
            'prediction_threshold': prediction_threshold
        },
        'overall': {
            'total_patients': len(true_risks),
            'total_events': np.sum(true_events),
            'event_rate': np.mean(true_events),
            'interventions': (np.sum(intervention_mask)
                              if intervention_mask is not None else 0)
        }
    }

    # Add statistics text if requested
    if show_stats:
        stats_text = f"""Quadrant Statistics:
High Priority: {stats['high_priority']['count']} patients
  Event Rate: {stats['high_priority']['event_rate']:.1%}

Missed Opportunities: {stats['missed_opportunities']['count']} patients
  Event Rate: {stats['missed_opportunities']['event_rate']:.1%}

False Alarms: {stats['false_alarms']['count']} patients
  Event Rate: {stats['false_alarms']['event_rate']:.1%}

Correct Rejections: {stats['correct_rejections']['count']} patients
  Event Rate: {stats['correct_rejections']['event_rate']:.1%}"""

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, va='top', ha='left', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, stats


@log_call
def analyze_patients_by_outcome(
    temporal_risk_matrix: np.ndarray,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    event_times: np.ndarray,
    prediction_start: int,
    window_length: int,
    integrated_risks: Optional[np.ndarray] = None,
    base_risks: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    n_samples_per_quadrant: int = 5,
    figsize: Tuple[int, int] = (16, 12),
    show_plot: bool = True
) -> Dict[str, Any]:
    """
    Analyze multiple patients grouped by their prediction outcome quadrant.

    This function provides insights into how different types of patients
    (TP, FP, TN, FN) differ in their risk patterns and characteristics.

    Parameters
    ----------
    temporal_risk_matrix : np.ndarray
        Complete temporal risk matrix, shape (n_patients, n_timesteps)
    predictions : np.ndarray
        ML predictions for all patients
    true_labels : np.ndarray
        True binary outcomes for all patients
    event_times : np.ndarray
        Time of first event for each patient (-1 if no event)
    prediction_start : in
        Starting timestep for the prediction window
    window_length : in
        Length of the prediction window in timesteps
    integrated_risks : np.ndarray, optional
        Integrated risks over prediction window
    base_risks : np.ndarray, optional
        Static base risks for all patients
    threshold : float, default=0.5
        Classification threshold for binary predictions
    n_samples_per_quadrant : int, default=5
        Number of sample patients to analyze per quadran
    figsize : tuple, default=(16, 12)
        Figure size for the visualization
    show_plot : bool, default=True
        Whether to display the plo

    Returns
    -------
    analysis_results : dic
        Dictionary containing:
        - quadrant_stats: DataFrame with statistics by quadran
        - sample_patients: dict with patient IDs by quadran
        - risk_patterns: dict with risk pattern analysis
        - figure: matplotlib.figure.Figure (if show_plot=False)

    Examples
    --------
    >>> analysis = analyze_patients_by_outcome(
    ...     temporal_matrix, predictions, labels, event_times,
    ...     prediction_start=20, window_length=12
    ... )
    >>> print(analysis['quadrant_stats'])
    """
    n_patients, n_timesteps = temporal_risk_matrix.shape

    # Create binary predictions
    binary_predictions = (predictions >= threshold).astype(int)

    # Identify quadrants
    tp_mask = (true_labels == 1) & (binary_predictions == 1)
    fp_mask = (true_labels == 0) & (binary_predictions == 1)
    tn_mask = (true_labels == 0) & (binary_predictions == 0)
    fn_mask = (true_labels == 1) & (binary_predictions == 0)

    quadrant_masks = {
        'True Positive': tp_mask,
        'False Positive': fp_mask,
        'True Negative': tn_mask,
        'False Negative': fn_mask
    }

    quadrant_colors = {
        'True Positive': '#2E7D32',
        'False Positive': '#F57C00',
        'True Negative': '#1976D2',
        'False Negative': '#D32F2F'
    }

    # Calculate integrated risks if not provided
    if integrated_risks is None:
        window_risks = temporal_risk_matrix[:, prediction_start:
                                            prediction_start + window_length]
        timestep_duration = 1/52
        clipped_risks = np.clip(window_risks, 1e-10, 1 - 1e-10)
        timestep_hazards = -np.log(1 - clipped_risks) / timestep_duration
        cumulative_hazards = np.sum(
            timestep_hazards, axis=1) * timestep_duration
        integrated_risks = 1 - np.exp(-cumulative_hazards)

    # Calculate statistics for each quadran
    quadrant_stats = []
    sample_patients = {}

    for quadrant, mask in quadrant_masks.items():
        n_patients_in_quadrant = np.sum(mask)

        if n_patients_in_quadrant > 0:
            # Basic statistics
            quadrant_integrated_risks = integrated_risks[mask]
            quadrant_predictions = predictions[mask]
            quadrant_base_risks = (base_risks[mask] if base_risks is not None
                                   else np.nan)

            # Temporal statistics
            quadrant_risk_volatility = np.std(
                temporal_risk_matrix[mask], axis=1)
            quadrant_risk_trends = []

            for i in np.where(mask)[0]:
                trend = ('Increasing' if temporal_risk_matrix[i, -1] >
                         temporal_risk_matrix[i, 0] else 'Decreasing')
                quadrant_risk_trends.append(trend)

            increasing_pct = np.mean(
                [t == 'Increasing' for t in quadrant_risk_trends])

            # Event timing analysis
            quadrant_event_times = event_times[mask]
            has_events = quadrant_event_times >= 0

            stats = {
                'quadrant': quadrant,
                'n_patients': n_patients_in_quadrant,
                'percentage': n_patients_in_quadrant / n_patients * 100,
                'mean_integrated_risk': np.mean(quadrant_integrated_risks),
                'std_integrated_risk': np.std(quadrant_integrated_risks),
                'mean_prediction': np.mean(quadrant_predictions),
                'std_prediction': np.std(quadrant_predictions),
                'mean_base_risk': (np.mean(quadrant_base_risks)
                                   if base_risks is not None else np.nan),
                'mean_risk_volatility': np.mean(quadrant_risk_volatility),
                'increasing_trend_pct': increasing_pct * 100,
                'event_rate': np.mean(has_events) * 100,
                'mean_event_time': (np.mean(quadrant_event_times[has_events])
                                    if np.any(has_events) else np.nan)
            }

            quadrant_stats.append(stats)

            # Sample patients for detailed analysis
            if n_patients_in_quadrant >= n_samples_per_quadrant:
                sample_indices = np.random.choice(
                    np.where(mask)[0],
                    n_samples_per_quadrant,
                    replace=False
                )
            else:
                sample_indices = np.where(mask)[0]

            sample_patients[quadrant] = sample_indices

    quadrant_stats_df = pd.DataFrame(quadrant_stats)

    # Create visualization
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('AI Quadrant Analysis - Patient Outcomes by Prediction Type',
                 fontsize=16, fontweight='bold')

    # Panel 1: Quadrant overview
    ax1 = fig.add_subplot(gs[0, 0])

    if not quadrant_stats_df.empty:
        quadrant_names = quadrant_stats_df['quadrant'].tolist()
        quadrant_counts = quadrant_stats_df['n_patients'].tolist()
        colors = [quadrant_colors[q] for q in quadrant_names]

        bars = ax1.bar(range(len(quadrant_names)),
                       quadrant_counts, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(quadrant_names)))
        ax1.set_xticklabels([q.replace(' ', '\n')
                            for q in quadrant_names], rotation=0)
        ax1.set_ylabel('Number of Patients')
        ax1.set_title('Patients by Quadrant')

        # Add percentage labels
        for bar, count, pct in zip(bars, quadrant_counts,
                                   quadrant_stats_df['percentage']):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, f'{count}\n({pct:.1f}%)',
                     ha='center', va='bottom', fontweight='bold')

    # Panel 2: Risk distributions by quadran
    ax2 = fig.add_subplot(gs[0, 1])

    for quadrant, mask in quadrant_masks.items():
        if np.sum(mask) > 0:
            quadrant_risks = integrated_risks[mask]
            ax2.hist(quadrant_risks, bins=20, alpha=0.6, density=True,
                     label=quadrant, color=quadrant_colors[quadrant])

    ax2.set_xlabel('Integrated Risk')
    ax2.set_ylabel('Density')
    ax2.set_title('Risk Distribution by Quadrant')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Prediction distributions by quadran
    ax3 = fig.add_subplot(gs[0, 2])

    for quadrant, mask in quadrant_masks.items():
        if np.sum(mask) > 0:
            quadrant_preds = predictions[mask]
            ax3.hist(quadrant_preds, bins=20, alpha=0.6, density=True,
                     label=quadrant, color=quadrant_colors[quadrant])

    ax3.axvline(threshold, color='black', linestyle='--',
                linewidth=2, label='Threshold')
    ax3.set_xlabel('ML Prediction')
    ax3.set_ylabel('Density')
    ax3.set_title('Prediction Distribution by Quadrant')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Risk vs Prediction scatter
    ax4 = fig.add_subplot(gs[1, :])

    for quadrant, mask in quadrant_masks.items():
        if np.sum(mask) > 0:
            ax4.scatter(integrated_risks[mask], predictions[mask],
                        alpha=0.6, s=50, label=quadrant,
                        color=quadrant_colors[quadrant])

    ax4.axhline(threshold, color='black', linestyle='--',
                alpha=0.7, label='Threshold')
    ax4.set_xlabel('Integrated Risk')
    ax4.set_ylabel('ML Prediction')
    ax4.set_title('Risk vs Prediction by Quadrant')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel 5: Risk volatility comparison
    ax5 = fig.add_subplot(gs[2, 0])

    if not quadrant_stats_df.empty:
        volatility_means = quadrant_stats_df['mean_risk_volatility'].tolist()
        quadrant_names = quadrant_stats_df['quadrant'].tolist()
        colors = [quadrant_colors[q] for q in quadrant_names]

        bars = ax5.bar(range(len(quadrant_names)),
                       volatility_means, color=colors, alpha=0.7)
        ax5.set_xticks(range(len(quadrant_names)))
        ax5.set_xticklabels([q.replace(' ', '\n')
                            for q in quadrant_names], rotation=0)
        ax5.set_ylabel('Mean Risk Volatility')
        ax5.set_title('Risk Volatility by Quadrant')

        # Add value labels
        for bar, vol in zip(bars, volatility_means):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{vol:.3f}', ha='center', va='bottom', fontweight='bold')

    # Panel 6: Event timing analysis
    ax6 = fig.add_subplot(gs[2, 1])

    if not quadrant_stats_df.empty:
        event_rates = quadrant_stats_df['event_rate'].tolist()
        quadrant_names = quadrant_stats_df['quadrant'].tolist()
        colors = [quadrant_colors[q] for q in quadrant_names]

        bars = ax6.bar(range(len(quadrant_names)),
                       event_rates, color=colors, alpha=0.7)
        ax6.set_xticks(range(len(quadrant_names)))
        ax6.set_xticklabels([q.replace(' ', '\n')
                            for q in quadrant_names], rotation=0)
        ax6.set_ylabel('Event Rate (%)')
        ax6.set_title('Event Rate by Quadrant')

        # Add value labels
        for bar, rate in zip(bars, event_rates):
            ax6.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, f'{rate:.1f}%',
                     ha='center', va='bottom', fontweight='bold')

    # Panel 7: Summary statistics table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    if not quadrant_stats_df.empty:
        # Create summary table
        summary_data = []
        for _, row in quadrant_stats_df.iterrows():
            summary_data.append([
                row['quadrant'][:2],  # Abbreviation
                f"{row['n_patients']}",
                f"{row['mean_integrated_risk']:.3f}",
                f"{row['mean_prediction']:.3f}",
                f"{row['event_rate']:.1f}%"
            ])

        # Create table
        table = ax7.table(cellText=summary_data,
                          colLabels=['Type', 'N', 'Risk', 'Pred', 'Events'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.15, 0.15, 0.2, 0.2, 0.2])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color code rows
        for i, row in enumerate(summary_data):
            quadrant_full = [k for k, v in {
                'TP': 'True Positive', 'FP': 'False Positive',
                'TN': 'True Negative', 'FN': 'False Negative'
            }.items() if v.startswith(row[0])][0]
            quadrant_name = {
                'TP': 'True Positive', 'FP': 'False Positive',
                'TN': 'True Negative', 'FN': 'False Negative'
            }[quadrant_full]

            for j in range(5):
                table[(i+1, j)].set_facecolor(quadrant_colors[quadrant_name])
                table[(i+1, j)].set_alpha(0.3)

        ax7.set_title('Summary Statistics')

    plt.tight_layout()

    # Risk pattern analysis
    risk_patterns = {}
    for quadrant, mask in quadrant_masks.items():
        if np.sum(mask) > 0:
            quadrant_trajectories = temporal_risk_matrix[mask]

            # Calculate mean trajectory
            mean_trajectory = np.mean(quadrant_trajectories, axis=0)

            # Calculate correlation with time
            time_points = np.arange(n_timesteps)
            correlations = []
            for i in range(np.sum(mask)):
                if np.std(quadrant_trajectories[i]) > 0:
                    corr = np.corrcoef(
                        time_points, quadrant_trajectories[i])[0, 1]
                    correlations.append(corr)

            risk_patterns[quadrant] = {
                'mean_trajectory': mean_trajectory,
                'time_correlation': (np.mean(correlations)
                                     if correlations else 0),
                'trajectory_std': np.std(quadrant_trajectories, axis=0)
            }

    # Prepare return dictionary
    analysis_results = {
        'quadrant_stats': quadrant_stats_df,
        'sample_patients': sample_patients,
        'risk_patterns': risk_patterns,
        'quadrant_masks': quadrant_masks,
        'threshold': threshold,
        'confusion_matrix': {
            'tp': np.sum(tp_mask),
            'fp': np.sum(fp_mask),
            'tn': np.sum(tn_mask),
            'fn': np.sum(fn_mask)
        }
    }

    if show_plot:
        plt.show()
    else:
        analysis_results['figure'] = fig
        plt.close(fig)

    return analysis_results
