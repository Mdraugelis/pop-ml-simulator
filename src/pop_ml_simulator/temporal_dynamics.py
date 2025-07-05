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
    """Extended temporal risk simulator with seasonal effects and shocks.

    This version enforces strict bounds on temporal modifiers and overall
    probability to maintain population calibration and clinical
    interpretability.

    Parameters
    ----------
    base_risks : np.ndarray
        Base annual risk for each patient
    rho : float, default=0.9
        AR(1) persistence parameter
    sigma : float, default=0.1
        Noise standard deviation
    temporal_bounds : tuple, default=(0.2, 2.5)
        (min, max) bounds for temporal modifier
    max_risk_threshold : float, default=0.95
        Absolute maximum risk allowed
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
        temporal_bounds: Tuple[float, float] = (0.2, 2.5),
        max_risk_threshold: float = 0.95,
        seasonal_amplitude: float = 0.2,
        seasonal_period: int = 52
    ):
        """Initialize enhanced temporal risk simulator with bounds."""
        super().__init__(base_risks, rho, sigma, temporal_bounds)
        self.temporal_bounds = temporal_bounds
        self.max_risk_threshold = max_risk_threshold
        self.seasonal_amplitude = seasonal_amplitude
        self.seasonal_period = seasonal_period
        self.time_step = 0
        self.external_shocks: List[Dict] = []
        self.target_population_rate = float(np.mean(base_risks))

    @property
    def temporal_modifiers(self) -> np.ndarray:
        """Expose current modifiers for backward compatibility."""
        return self.current_modifiers

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

        if magnitude > self.temporal_bounds[1]:
            raise ValueError(
                f"Shock magnitude {magnitude} exceeds "
                f"temporal bounds {self.temporal_bounds[1]}"
            )
        if magnitude < 0.1:
            raise ValueError(f"Shock magnitude {magnitude} too small")

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

    def step(self) -> np.ndarray:  # type: ignore[override]
        """Advance one time step with bounded temporal evolution."""
        # Get seasonal and shock effects
        seasonal_modifier = self.get_seasonal_modifier()
        shock_modifiers = self.get_shock_modifier()

        # Generate AR(1) evolution around the seasonal baseline
        # The AR(1) process evolves around seasonal_modifier instead of 1.0
        noise = np.random.normal(0, self.sigma, self.n_patients)
        ar1_modifiers = (self.rho * self.current_modifiers +
                         (1 - self.rho) * seasonal_modifier + noise)

        # Apply shock effects multiplicatively
        self.current_modifiers = ar1_modifiers * shock_modifiers

        # Apply temporal bounds to modifiers
        self.current_modifiers = np.clip(
            self.current_modifiers,
            self.temporal_bounds[0],
            self.temporal_bounds[1]
        )

        # Calculate risks and apply risk threshold
        temporal_risks = self.base_risks * self.current_modifiers
        temporal_risks = np.clip(temporal_risks, 0.0, self.max_risk_threshold)

        # Preserve population rate while respecting bounds
        temporal_risks = self._preserve_population_rate(temporal_risks)

        # Validate final risks
        self._validate_temporal_risks(temporal_risks)

        # Store history
        self.modifier_history.append(self.current_modifiers.copy())
        self.risk_history.append(temporal_risks.copy())
        self.time_step += 1

        return temporal_risks

    def _preserve_population_rate(
        self, temporal_risks: np.ndarray
    ) -> np.ndarray:
        """Rescale risks to preserve the original population incident rate.

        Only applies rescaling if drift exceeds seasonal variation bounds.
        """
        current_mean = float(np.mean(temporal_risks))
        if current_mean > 0:
            # Allow drift up to seasonal amplitude plus some tolerance
            allowed_drift = self.seasonal_amplitude + 0.05
            rate_ratio = current_mean / self.target_population_rate

            # Only rescale if drift is beyond seasonal variation
            if (rate_ratio > (1 + allowed_drift) or
                    rate_ratio < (1 - allowed_drift)):
                scaling_factor = self.target_population_rate / current_mean
                temporal_risks = temporal_risks * scaling_factor
                temporal_risks = np.clip(
                    temporal_risks,
                    0.0,
                    self.max_risk_threshold,
                )
        return temporal_risks

    def _validate_temporal_risks(self, temporal_risks: np.ndarray) -> None:
        """Validate temporal risk constraints."""
        max_risk = float(np.max(temporal_risks))
        min_risk = float(np.min(temporal_risks))
        if max_risk > 1.0:
            raise ValueError(f"Temporal risk exceeded 1.0: {max_risk:.4f}")
        if min_risk < 0.0:
            raise ValueError(f"Negative temporal risk: {min_risk:.4f}")

        # Allow drift up to seasonal amplitude plus tolerance
        current_pop_rate = float(np.mean(temporal_risks))
        rate_ratio = current_pop_rate / self.target_population_rate
        allowed_drift = self.seasonal_amplitude + 0.05

        if (rate_ratio > (1 + allowed_drift) or
                rate_ratio < (1 - allowed_drift)):
            raise ValueError(
                f"Population rate drift exceeds seasonal bounds: "
                f"current={current_pop_rate:.4f}, "
                f"target={self.target_population_rate:.4f}, "
                f"ratio={rate_ratio:.4f}, "
                f"allowed_drift={allowed_drift:.4f}"
            )

    def get_current_risks(self) -> np.ndarray:
        """Return current temporal risks with constraints applied."""
        # Return the most recent risks from history if available
        if self.risk_history:
            return self.risk_history[-1].copy()

        # Otherwise calculate from current modifiers
        temporal_risks = self.base_risks * self.current_modifiers
        temporal_risks = np.clip(temporal_risks, 0.0, self.max_risk_threshold)
        return self._preserve_population_rate(temporal_risks)

    def get_risk_matrix(self) -> np.ndarray:
        """
        Get the complete temporal risk matrix.

        Returns
        -------
        risk_matrix : np.ndarray
            Shape (n_patients, n_timesteps) array of temporal risks
        """
        if not self.risk_history:
            raise ValueError("No simulation history available. "
                             "Run simulation first.")

        return np.array(self.risk_history).T

    def get_patient_trajectory(self, patient_id: int) -> np.ndarray:
        """
        Get risk trajectory for a specific patient.

        Parameters
        ----------
        patient_id : int
            Index of the patient (0 to n_patients-1)

        Returns
        -------
        trajectory : np.ndarray
            Risk values over time for the specified patient
        """
        if patient_id < 0 or patient_id >= self.n_patients:
            raise ValueError(
                f"Patient ID {patient_id} out of range "
                f"[0, {self.n_patients-1}]")

        if not self.risk_history:
            raise ValueError("No simulation history available. "
                             "Run simulation first.")
        return np.array([risks[patient_id] for risks in self.risk_history])

    def get_timestep_risks(self, timestep: int) -> np.ndarray:
        """
        Get risk values for all patients at a specific timestep.

        Parameters
        ----------
        timestep : int
            Timestep index (0 to n_timesteps-1)

        Returns
        -------
        risks : np.ndarray
            Risk values for all patients at the specified timestep
        """
        if not self.risk_history:
            raise ValueError("No simulation history available. "
                             "Run simulation first.")

        if timestep < 0 or timestep >= len(self.risk_history):
            raise ValueError(
                f"Timestep {timestep} out of range "
                f"[0, {len(self.risk_history)-1}]")
        return self.risk_history[timestep].copy()


def build_temporal_risk_matrix(
    base_risks: np.ndarray,
    n_timesteps: int,
    rho: float = 0.9,
    sigma: float = 0.1,
    temporal_bounds: Tuple[float, float] = (0.2, 2.5),
    max_risk_threshold: float = 0.95,
    seasonal_amplitude: float = 0.2,
    seasonal_period: int = 52,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Build a temporal risk matrix for all patients over time.

    This function creates a complete (n_patients, n_timesteps) matrix of
    time-varying risks using the Enhanced Temporal Risk Simulator with
    AR(1) dynamics, seasonal effects, and safety bounds.

    Parameters
    ----------
    base_risks : np.ndarray
        Base annual risk for each patient, shape (n_patients,)
    n_timesteps : int
        Number of timesteps to simulate
    rho : float, default=0.9
        AR(1) persistence parameter (0 < rho < 1)
    sigma : float, default=0.1
        Noise standard deviation for AR(1) process
    temporal_bounds : tuple, default=(0.2, 2.5)
        (min, max) bounds for temporal modifiers
    max_risk_threshold : float, default=0.95
        Absolute maximum risk allowed (clinical safety)
    seasonal_amplitude : float, default=0.2
        Amplitude of seasonal variation (0 = no seasonality)
    seasonal_period : int, default=52
        Period of seasonal cycle (e.g., 52 for weekly data with annual cycle)
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    risk_matrix : np.ndarray
        Shape (n_patients, n_timesteps) matrix of temporal risks.
        - risk_matrix[i, 0] == base_risks[i] (initial condition)
        - All values in [0, 1] range
        - Temporal autocorrelation > 0.8 for patient trajectories

    Examples
    --------
    >>> import numpy as np
    >>> from pop_ml_simulator import assign_patient_risks
    >>>
    >>> # Create patient population
    >>> n_patients = 1000
    >>> base_risks = assign_patient_risks(n_patients, 0.1, random_seed=42)
    >>>
    >>> # Build temporal risk matrix
    >>> risk_matrix = build_temporal_risk_matrix(
    ...     base_risks, n_timesteps=52, random_seed=42
    ... )
    >>>
    >>> print(f"Matrix shape: {risk_matrix.shape}")
    >>> print(f"Initial risks match base: "
    ...       f"{np.allclose(risk_matrix[:, 0], base_risks)}")
    >>> print(f"All risks in [0,1]: "
    ...       f"{np.all((risk_matrix >= 0) & (risk_matrix <= 1))}")
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create enhanced temporal risk simulator
    simulator = EnhancedTemporalRiskSimulator(
        base_risks=base_risks,
        rho=rho,
        sigma=sigma,
        temporal_bounds=temporal_bounds,
        max_risk_threshold=max_risk_threshold,
        seasonal_amplitude=seasonal_amplitude,
        seasonal_period=seasonal_period
    )

    # Simulate for n_timesteps (subtract 1 because we start with initial state)
    if n_timesteps > 1:
        simulator.simulate(n_timesteps - 1)

    # Return the risk matrix
    return simulator.get_risk_matrix()
