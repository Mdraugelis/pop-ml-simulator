"""
Causal Inference Analysis Methods

This module provides implementations of common causal inference methods
for analyzing the results of healthcare AI intervention simulations.
"""

import numpy as np
import pandas as pd
from statsmodels.api import OLS
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from typing import Dict, Any

from .vectorized_simulator import SimulationResults


class RDDAnalysis:
    """
    Regression Discontinuity Design (RDD) Analysis
    """

    def __init__(self, results: SimulationResults, risk_threshold: float = 0.5, bandwidth: float = 0.1):
        self.results = results
        self.risk_threshold = risk_threshold
        self.bandwidth = bandwidth

    def analyze_intervention_effect(self) -> Dict[str, Any]:
        """
        Analyzes the intervention effect using RDD.
        """
        # Get risk scores and outcomes from the first prediction time
        pred_time = self.results.ml_prediction_times[0]
        risk_scores = self.results.ml_predictions[pred_time]
        outcomes = self.results.incident_matrix.any(axis=1)
        treated = risk_scores >= self.risk_threshold

        # Focus on patients near threshold
        near_threshold = np.abs(risk_scores - self.risk_threshold) <= self.bandwidth

        # Local linear regression on each side
        left_side = (risk_scores < self.risk_threshold) & near_threshold
        right_side = (risk_scores >= self.risk_threshold) & near_threshold

        X_left = (risk_scores[left_side] - self.risk_threshold).reshape(-1, 1)
        y_left = outcomes[left_side]
        model_left = LinearRegression().fit(X_left, y_left)

        X_right = (risk_scores[right_side] - self.risk_threshold).reshape(-1, 1)
        y_right = outcomes[right_side]
        model_right = LinearRegression().fit(X_right, y_right)

        left_limit = model_left.predict([[0]])[0]
        right_limit = model_right.predict([[0]])[0]

        rdd_effect = right_limit - left_limit

        return {
            'estimated_effect': rdd_effect,
            'true_effect': self.results.intervention_effectiveness,
            'n_near_threshold': near_threshold.sum(),
            'bandwidth_used': self.bandwidth
        }


class DiDAnalysis:
    """
    Difference-in-Differences (DiD) Analysis
    """

    def __init__(self, results: SimulationResults, intervention_start_time: int = 24):
        self.results = results
        self.intervention_start = intervention_start_time

    def analyze_intervention_effect(self) -> Dict[str, Any]:
        """
        Analyzes the intervention effect using DiD.
        """
        pre_period = np.arange(0, self.intervention_start)
        post_period = np.arange(self.intervention_start, self.results.n_timesteps)

        # This is a simplified assumption for DiD. In a real scenario,
        # treatment groups would be defined more explicitly.
        treated = self.results.intervention_matrix.getnnz(axis=1) > 0
        control = ~treated

        incidents = self.results.incident_matrix

        pre_treated = incidents[treated][:, pre_period].mean()
        pre_control = incidents[control][:, pre_period].mean()

        post_treated = incidents[treated][:, post_period].mean()
        post_control = incidents[control][:, post_period].mean()

        did_effect = (post_treated - pre_treated) - (post_control - pre_control)

        baseline_risk = pre_control
        relative_effect = -did_effect / baseline_risk

        return {
            'estimated_effect': relative_effect,
            'true_effect': self.results.intervention_effectiveness,
            'pre_treated': pre_treated,
            'pre_control': pre_control,
            'post_treated': post_treated,
            'post_control': post_control,
            'did_estimate': did_effect
        }


class ITSAnalysis:
    """
    Interrupted Time Series (ITS) Analysis
    """

    def __init__(self, results: SimulationResults, intervention_time: int = 24):
        self.results = results
        self.intervention_time = intervention_time

    def analyze_intervention_effect(self) -> Dict[str, Any]:
        """
        Analyzes the intervention effect using ITS.
        """
        monthly_rates = self.results.incident_matrix.mean(axis=0)

        time = np.arange(len(monthly_rates))
        post_intervention = (time >= self.intervention_time).astype(int)
        time_since_intervention = np.maximum(0, time - self.intervention_time)

        X = np.column_stack([
            np.ones_like(time),
            time,
            post_intervention,
            time_since_intervention
        ])

        model = OLS(monthly_rates, X).fit()

        baseline_level = model.params[0]
        baseline_slope = model.params[1]
        level_change = model.params[2]
        slope_change = model.params[3]

        pre_trend_projection = baseline_level + baseline_slope * len(monthly_rates)
        actual_final = monthly_rates[-1]
        absolute_effect = actual_final - pre_trend_projection
        relative_effect = -absolute_effect / pre_trend_projection

        return {
            'model': model,
            'baseline_level': baseline_level,
            'baseline_slope': baseline_slope,
            'level_change': level_change,
            'slope_change': slope_change,
            'relative_effect': relative_effect,
            'true_effect': self.results.intervention_effectiveness,
            'durbin_watson': sm.stats.durbin_watson(model.resid)
        }
