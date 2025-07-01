"""
ML simulation module for healthcare AI intervention modeling.

This module provides classes and functions for simulating machine learning
predictions with controlled performance characteristics, enabling realistic
evaluation of AI-guided interventions.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
from typing import Optional, Dict, List, Tuple, Union
from utils.logging import log_call


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
    prevalence : float
        Population prevalence of the outcome
    sensitivity_range : np.ndarray, optional
        Range of sensitivity values to evaluate
        
    Returns
    -------
    sensitivity_range : np.ndarray
        Sensitivity values evaluated
    results : dict
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
            # Bayes' theorem: PPV = (sens * prev) / (sens * prev + (1-spec) * (1-prev))
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
        Number of bins for the test
        
    Returns
    -------
    hl_statistic : float
        Hosmer-Lemeshow chi-square statistic
    p_value : float
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
    
    Uses calibrated noise to create predictions that achieve target
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
    noise_correlation : float
        Optimized correlation between predictions and true risk
    noise_scale : float
        Optimized noise scale parameter
    threshold : float
        Optimal decision threshold
        
    Examples
    --------
    >>> simulator = MLPredictionSimulator(target_sensitivity=0.8, target_ppv=0.3)
    >>> params = simulator.optimize_noise_parameters(true_labels, risk_scores)
    >>> predictions, binary = simulator.generate_predictions(true_labels, risk_scores)
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
        
        # Add label-dependent noise (true positives get boost)
        label_noise = np.where(
            true_labels == 1,
            np.random.normal(0.2, 0.1, n_patients),
            np.random.normal(-0.1, 0.1, n_patients)
        )
        
        noisy_scores += label_noise * noise_scale
        
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
                
                # Score based on distance from targets
                score = (abs(sensitivity - self.target_sensitivity) + 
                        abs(ppv - self.target_ppv))
                
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
        best_params : dict
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
    metrics : dict
        Comprehensive performance metrics
        
    Examples
    --------
    >>> metrics = evaluate_threshold_based(labels, predictions, 0.3)
    >>> print(f"PPV: {metrics['ppv']:.3f}, Sensitivity: {metrics['sensitivity']:.3f}")
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
        Percentage of highest-risk patients to select
        
    Returns
    -------
    metrics : dict
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
        'lift': ((tp / k) / (np.sum(true_labels) / n_patients)) if k > 0 else 0.0,
        'min_score_flagged': float(np.min(predictions[top_k_indices])) if k > 0 else 0.0
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
    results : dict
        Optimization results including threshold and metrics
        
    Examples
    --------
    >>> result = optimize_alert_threshold(preds, labels, capacity_constraint=0.05)
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
    metrics = evaluate_threshold_based(true_labels, predictions, capacity_threshold)
    
    # Calculate utility score
    utility = metrics['tp'] - fatigue_weight * metrics['fp']
    
    return {
        'optimal_threshold': capacity_threshold,
        'n_alerts': max_alerts,
        'metrics': metrics,
        'utility': float(utility),
        'efficiency': metrics['ppv']  # What fraction of alerts are useful
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
    
    Helps identify if model performs better for high-risk vs low-risk patients.
    
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
            # Get subset
            subset_true = true_labels[mask]
            subset_preds = predictions[mask]
            subset_risks = risk_scores[mask]
            
            # Calculate optimal threshold for this subset
            if np.sum(subset_true) > 0:  # Has positive cases
                best_threshold = 0.5
                best_f1 = 0
                
                for thresh in np.linspace(0.1, 0.9, 20):
                    binary = (subset_preds >= thresh).astype(int)
                    if np.sum(binary) > 0:  # Has predictions
                        cm = confusion_matrix(subset_true, binary)
                        if cm.shape == (1, 1):
                            # Handle case where only one class is present
                            if np.all(subset_true == 0):
                                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                            else:
                                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                        else:
                            tn, fp, fn, tp = cm.ravel()
                        f1 = (2 * tp / (2 * tp + fp + fn) 
                             if (2 * tp + fp + fn) > 0 else 0)
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