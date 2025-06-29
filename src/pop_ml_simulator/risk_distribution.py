"""
Risk distribution module for healthcare AI temporal simulation.

This module provides functions for assigning patient-level risk scores using
beta distributions to model realistic population heterogeneity.
"""

import numpy as np
from typing import Optional, Union


def assign_patient_risks(
    n_patients: int,
    annual_incident_rate: float,
    concentration: float = 0.5,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Assign individual risk scores to patients using beta distribution.
    
    This function creates a right-skewed risk distribution that matches
    real-world healthcare patterns where most patients have low risk
    while a small fraction drives the majority of events.
    
    Parameters
    ----------
    n_patients : int
        Number of patients to simulate
    annual_incident_rate : float
        Target population annual incident rate (e.g., 0.1 for 10%)
    concentration : float, default=0.5
        Beta distribution concentration parameter. Lower values create
        more heterogeneous populations.
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    base_annual_risks : np.ndarray
        Individual annual risk scores for each patient, bounded [0, 0.99]
        
    Examples
    --------
    >>> risks = assign_patient_risks(10000, 0.1, concentration=0.5)
    >>> print(f"Mean risk: {np.mean(risks):.3f}")
    Mean risk: 0.100
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Distribution parameters for right-skewed shape
    alpha = concentration
    beta_param = alpha * (1/annual_incident_rate - 1)
    
    # Sample all patient risks at once
    raw_risks = np.random.beta(alpha, beta_param, n_patients)
    
    # Scale to ensure population mean equals target
    scaling_factor = annual_incident_rate / np.mean(raw_risks)
    base_annual_risks = np.clip(raw_risks * scaling_factor, 0, 0.99)
    
    return base_annual_risks


def simulate_annual_incidents(
    risks: np.ndarray,
    n_simulations: int = 100,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate annual incidents based on individual risk scores.
    
    For each patient, their risk score represents their probability
    of having an incident in a given year. This function runs multiple
    simulations to generate incident counts.
    
    Parameters
    ----------
    risks : np.ndarray
        Individual annual risk scores for each patient
    n_simulations : int, default=100
        Number of simulation runs
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    incident_counts : np.ndarray
        Array of shape (n_simulations,) containing the total number
        of incidents in each simulation
        
    Examples
    --------
    >>> risks = assign_patient_risks(10000, 0.1)
    >>> counts = simulate_annual_incidents(risks, n_simulations=1000)
    >>> print(f"Mean incidents: {np.mean(counts):.1f}")
    Mean incidents: 1000.0
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n_patients = len(risks)
    incident_counts = []
    
    for sim in range(n_simulations):
        # For each patient, draw from Bernoulli distribution with p = risk
        incidents = np.random.binomial(1, risks)
        incident_counts.append(np.sum(incidents))
    
    return np.array(incident_counts)
