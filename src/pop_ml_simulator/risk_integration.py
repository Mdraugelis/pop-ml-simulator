"""
Risk integration module for temporal risk windows.

This module provides methods for converting risk trajectories over prediction
windows into single integrated risk values using survival-based integration.
"""

import numpy as np
import warnings


def integrate_window_risk(
    window_risks: np.ndarray,
    timestep_duration: float = 1/52,
    add_integration_noise: bool = True,
    noise_scale: float = 0.1
) -> np.ndarray:
    """
    Convert temporal risk trajectory over window to single risk prediction.

    This function integrates patient risk values over a prediction window
    using survival-based integration to produce a single risk estimate.

    Parameters
    ----------
    window_risks : np.ndarray
        Risk values for each patient over the prediction window.
        Shape: (n_patients, window_length) or (window_length,) for single
    timestep_duration : float, default=1/52
        Duration of each timestep as fraction of year (e.g., 1/52 for weekly)
    add_integration_noise : bool, default=True
        Whether to add patient-specific noise to reduce deterministic output
    noise_scale : float, default=0.1
        Scale of noise to add (if add_integration_noise=True)

    Returns
    -------
    integrated_risks : np.ndarray
        Single integrated risk value for each patient.
        Shape: (n_patients,) or scalar for single patient input

    Examples
    --------
    >>> # Single patient with increasing risk over 4 weeks
    >>> risks = np.array([0.1, 0.15, 0.2, 0.25])
    >>> integrated = integrate_window_risk(risks)
    >>> print(f"Integrated risk: {integrated:.3f}")

    >>> # Multiple patients
    >>> risks = np.random.uniform(0.05, 0.15, size=(100, 12))
    >>> integrated = integrate_window_risk(risks)
    >>> print(f"Shape: {integrated.shape}")

    Notes
    -----
    This method properly accounts for the cumulative nature of risk over
    time, following the mathematical relationship: S(t) = exp(-∫h(s)ds)
    """
    # Handle input dimensions
    single_patient = False
    if window_risks.ndim == 1:
        single_patient = True
        window_risks = window_risks.reshape(1, -1)

    if window_risks.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape "
                         f"{window_risks.shape}")

    n_patients, window_length = window_risks.shape

    # Validate risks are in [0, 1]
    if np.any((window_risks < 0) | (window_risks > 1)):
        warnings.warn("Risk values outside [0, 1] will be clipped")
        window_risks = np.clip(window_risks, 0.0, 1.0)

    # Convert risks to hazards for each timestep
    # Clip to avoid log(0) issues
    clipped_risks = np.clip(window_risks, 1e-10, 1 - 1e-10)

    # Convert each timestep risk to hazard rate
    # These are conditional probabilities, so we need to adjust
    timestep_hazards = -np.log(1 - clipped_risks) / timestep_duration

    # Sum hazards over window (cumulative hazard)
    cumulative_hazard = np.sum(timestep_hazards, axis=1) * \
        timestep_duration

    # Convert back to probability: P = 1 - exp(-H)
    integrated_risks = 1 - np.exp(-cumulative_hazard)

    # Add patient-specific noise to reduce deterministic output
    if add_integration_noise:
        # Add noise proportional to the risk level to maintain realism
        noise = np.random.normal(0, noise_scale * integrated_risks)
        integrated_risks = integrated_risks + noise

    # Final validation
    integrated_risks = np.clip(integrated_risks, 0.0, 1.0)

    # Return scalar for single patient input
    if single_patient:
        return integrated_risks[0]

    return integrated_risks


def extract_risk_windows(
    temporal_risks: np.ndarray,
    start_time: int,
    window_length: int
) -> np.ndarray:
    """
    Extract risk windows for all patients from temporal risk matrix.

    Parameters
    ----------
    temporal_risks : np.ndarray
        Complete temporal risk matrix. Shape: (n_patients, total_timesteps)
    start_time : int
        Starting timestep for the window (0-indexed)
    window_length : int
        Length of the prediction window in timesteps

    Returns
    -------
    window_risks : np.ndarray
        Risk values for all patients over the specified window.
        Shape: (n_patients, window_length)

    Raises
    ------
    ValueError
        If window extends beyond available timesteps

    Examples
    --------
    >>> # Extract 12-week windows starting at week 10
    >>> temporal_risks = np.random.uniform(0.05, 0.15, size=(1000, 52))
    >>> windows = extract_risk_windows(temporal_risks, start_time=10,
    ...                               window_length=12)
    >>> print(f"Window shape: {windows.shape}")
    """
    n_patients, total_timesteps = temporal_risks.shape
    end_time = start_time + window_length

    if start_time < 0:
        raise ValueError(f"start_time must be non-negative, got {start_time}")

    if end_time > total_timesteps:
        raise ValueError(
            f"Window extends beyond available timesteps. "
            f"Requested [{start_time}, {end_time}), but only have "
            f"{total_timesteps} timesteps"
        )

    return temporal_risks[:, start_time:end_time]


def validate_integration_bounds(
    integrated_risks: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    """
    Validate that integrated risks are valid probabilities.

    Parameters
    ----------
    integrated_risks : np.ndarray
        Integrated risk values to validate
    tolerance : float, default=1e-10
        Numerical tolerance for bounds checking

    Returns
    -------
    valid : bool
        True if all values are in [0, 1] within tolerance

    Warns
    -----
    UserWarning
        If any values are outside valid probability range
    """
    min_val = np.min(integrated_risks)
    max_val = np.max(integrated_risks)

    valid = True

    if min_val < -tolerance:
        warnings.warn(f"Found negative integrated risks: min = {min_val}")
        valid = False

    if max_val > 1 + tolerance:
        warnings.warn(f"Found integrated risks > 1: max = {max_val}")
        valid = False

    return valid
