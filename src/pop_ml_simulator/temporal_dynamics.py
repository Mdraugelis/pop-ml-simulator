"""
Temporal dynamics module for healthcare AI simulation.

This module provides classes and functions for modeling time-varying
patient risks using autoregressive processes, seasonal effects, and shocks.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from utils.logging import log_call


@log_call
def simulate_ar1_process(
    n_timesteps: int,
    rho: float,
    sigma: float,
    mu: float = 1.0,
    initial_value: Optional[float] = None,
    bounds: Tuple[float, float] = (0.5, 2.0),
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate an AR(1) autoregressive process with bounds.

    The AR(1) process is defined as:
    X_t = rho * X_{t-1} + (1-rho) * mu + epsilon_t

    where epsilon_t ~ N(0, sigma^2)

    Parameters
    ----------
    n_timesteps : int
        Number of time steps to simulate
    rho : float
        Persistence parameter (0 < rho < 1). Higher values mean
        more autocorrelation.
    sigma : float
        Standard deviation of noise term
    mu : float, default=1.0
        Long-term mean of the process
    initial_value : float, optional
        Starting value. Defaults to mu if not provided.
    bounds : tuple of float, default=(0.5, 2.0)
        (min, max) bounds for the process values
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    values : np.ndarray
        Time series of AR(1) process values

    Examples
    --------
    >>> trajectory = simulate_ar1_process(52, rho=0.9, sigma=0.1)
    >>> print(f"Mean: {np.mean(trajectory):.3f}")
    Mean: 1.002
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if initial_value is None:
        initial_value = mu

    values = np.zeros(n_timesteps)
    values[0] = initial_value

    for t in range(1, n_timesteps):
        # AR(1) update
        noise = np.random.normal(0, sigma)
        values[t] = rho * values[t-1] + (1 - rho) * mu + noise

        # Apply bounds
        values[t] = np.clip(values[t], bounds[0], bounds[1])

    return values


class TemporalRiskSimulator:
    """
    Simulates time-varying patient risks using AR(1) for temporal modifiers.

    Each patient's time-varying risk is modeled as:
    risk_i(t) = base_risk_i * temporal_modifier_i(t)

    where temporal_modifier follows an AR(1) process.

    Parameters
    ----------
    base_risks : np.ndarray
        Base annual risk for each patient
    rho : float, default=0.9
        AR(1) persistence parameter
    sigma : float, default=0.1
        Noise standard deviation
    bounds : tuple, default=(0.5, 2.0)
        (min, max) bounds for temporal modifier

    Attributes
    ----------
    n_patients : int
        Number of patients in simulation
    current_modifiers : np.ndarray
        Current temporal risk modifiers for all patients
    modifier_history : list of np.ndarray
        History of temporal modifiers over time
    risk_history : list of np.ndarray
        History of time-varying risks over time
    """

    def __init__(
        self,
        base_risks: np.ndarray,
        rho: float = 0.9,
        sigma: float = 0.1,
        bounds: Tuple[float, float] = (0.5, 2.0)
    ):
        """Initialize the temporal risk simulator."""
        self.base_risks = base_risks
        self.n_patients = len(base_risks)
        self.rho = rho
        self.sigma = sigma
        self.bounds = bounds

        # Initialize temporal modifiers at 1.0
        self.current_modifiers = np.ones(self.n_patients)

        # Store history
        self.modifier_history = [self.current_modifiers.copy()]
        self.risk_history = [self.base_risks * self.current_modifiers]

    def step(self) -> None:
        """
        Advance one time step, updating all patient risk modifiers.

        Uses vectorized AR(1) update for all patients simultaneously.
        """
        # Generate noise for all patients
        noise = np.random.normal(0, self.sigma, self.n_patients)

        # AR(1) update for all patients
        self.current_modifiers = (self.rho * self.current_modifiers +
                                  (1 - self.rho) * 1.0 + noise)

        # Apply bounds
        self.current_modifiers = np.clip(self.current_modifiers,
                                         self.bounds[0], self.bounds[1])

        # Store history
        self.modifier_history.append(self.current_modifiers.copy())
        self.risk_history.append(self.base_risks * self.current_modifiers)

    def simulate(self, n_timesteps: int) -> None:
        """
        Simulate multiple time steps.

        Parameters
        ----------
        n_timesteps : int
            Number of time steps to simulate
        """
        for _ in range(n_timesteps):
            self.step()

    def get_current_risks(self) -> np.ndarray:
        """
        Get current time-varying risks for all patients.

        Returns
        -------
        current_risks : np.ndarray
            Current annual risk for each patient
        """
        return self.base_risks * self.current_modifiers

    def get_histories(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get full history as numpy arrays.

        Returns
        -------
        modifier_history : np.ndarray
            Shape (n_patients, n_timesteps) array of temporal modifiers
        risk_history : np.ndarray
            Shape (n_patients, n_timesteps) array of time-varying risks
        """
        return (np.array(self.modifier_history).T,
                np.array(self.risk_history).T)


class EnhancedTemporalRiskSimulator(TemporalRiskSimulator):
    """
    Extended temporal risk simulator with seasonal effects and external shocks.

    Adds seasonal patterns and the ability to model external shocks
    (e.g., pandemics, flu seasons) on top of the base AR(1) process.

    Parameters
    ----------
    base_risks : np.ndarray
        Base annual risk for each patient
    rho : float, default=0.9
        AR(1) persistence parameter
    sigma : float, default=0.1
        Noise standard deviation
    bounds : tuple, default=(0.5, 2.0)
        (min, max) bounds for temporal modifier
    seasonal_amplitude : float, default=0.2
        Amplitude of seasonal variation (0 = no seasonality)
    seasonal_period : int, default=52
        Period of seasonal cycle (e.g., 52 for weekly data with annual cycle)
    """

    def __init__(
        self,
        base_risks: np.ndarray,
        rho: float = 0.9,
        sigma: float = 0.1,
        bounds: Tuple[float, float] = (0.5, 2.0),
        seasonal_amplitude: float = 0.2,
        seasonal_period: int = 52
    ):
        """Initialize enhanced temporal risk simulator."""
        super().__init__(base_risks, rho, sigma, bounds)
        self.seasonal_amplitude = seasonal_amplitude
        self.seasonal_period = seasonal_period
        self.time_step = 0
        self.external_shocks: List[Dict] = []

    def add_shock(
        self,
        time_step: int,
        magnitude: float,
        duration: int,
        affected_fraction: float = 1.0,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Add an external shock (e.g., pandemic, flu outbreak).

        Parameters
        ----------
        time_step : int
            When the shock occurs
        magnitude : float
            Multiplicative effect on risk (e.g., 1.5 = 50% increase)
        duration : int
            How long the shock lasts (in time steps)
        affected_fraction : float, default=1.0
            Fraction of population affected (0 to 1)
        random_seed : int, optional
            Random seed for selecting affected patients
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.external_shocks.append({
            'time': time_step,
            'magnitude': magnitude,
            'duration': duration,
            'affected_fraction': affected_fraction,
            'affected_patients': np.random.choice(
                self.n_patients,
                int(self.n_patients * affected_fraction),
                replace=False
            )
        })

    def get_seasonal_modifier(self) -> float:
        """
        Calculate seasonal risk modifier.

        Uses sinusoidal pattern with peak in winter (phase shifted by pi/2).

        Returns
        -------
        modifier : float
            Seasonal risk modifier
        """
        phase = 2 * np.pi * self.time_step / self.seasonal_period
        return 1.0 + self.seasonal_amplitude * np.sin(phase + np.pi/2)

    def get_shock_modifier(self) -> np.ndarray:
        """
        Calculate shock modifier for current time step.

        Returns
        -------
        shock_modifier : np.ndarray
            Multiplicative shock effect for each patient
        """
        shock_modifier = np.ones(self.n_patients)

        for shock in self.external_shocks:
            if (shock['time'] <= self.time_step <
                    shock['time'] + shock['duration']):
                affected_patients = shock['affected_patients']
                shock_modifier[affected_patients] *= shock['magnitude']

        return shock_modifier

    def step(self) -> None:
        """
        Advance one time step with seasonal and shock effects.

        Applies standard AR(1) update, then multiplies by seasonal
        and shock modifiers.
        """
        # Standard AR(1) update
        super().step()

        # Apply seasonal modifier
        seasonal_mod = self.get_seasonal_modifier()
        self.current_modifiers *= seasonal_mod

        # Apply shock modifier
        shock_mod = self.get_shock_modifier()
        self.current_modifiers *= shock_mod

        # Re-apply bounds
        self.current_modifiers = np.clip(self.current_modifiers,
                                         self.bounds[0], self.bounds[1])

        # Update risk history with new modifiers
        self.risk_history[-1] = self.base_risks * self.current_modifiers

        self.time_step += 1
