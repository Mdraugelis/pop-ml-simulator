"""
Hazard modeling module for healthcare AI simulation.

This module provides functions and classes for converting between risk probabilities
and hazard rates, and generating healthcare incidents using hazard-based approaches.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union


def annual_risk_to_hazard(annual_risk: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert annual risk (probability) to constant hazard rate.
    
    Uses the relationship between survival function and hazard:
    S(1 year) = 1 - annual_risk = exp(-h * 1)
    Therefore: h = -ln(1 - annual_risk)
    
    Parameters
    ----------
    annual_risk : float or np.ndarray
        Probability of event within one year (0 to 1)
        
    Returns
    -------
    annual_hazard : float or np.ndarray
        Constant hazard rate (events per year)
        
    Examples
    --------
    >>> hazard = annual_risk_to_hazard(0.1)
    >>> print(f"10% annual risk = {hazard:.3f} hazard rate")
    10% annual risk = 0.105 hazard rate
    """
    # Handle edge cases
    annual_risk = np.asarray(annual_risk)
    
    # Clip to avoid numerical issues: risk = 1.0 would give infinite hazard
    # In practice, risks should be < 1.0 anyway
    clipped_risk = np.clip(annual_risk, 0.0, 0.999999)
    
    # Warn if clipping occurred
    if np.any(annual_risk >= 1.0):
        import warnings
        warnings.warn(
            "Annual risk values >= 1.0 detected and clipped to 0.999999. "
            "Consider using risks < 1.0 for realistic modeling.",
            RuntimeWarning
        )
    
    hazard = -np.log(1 - clipped_risk)
    
    # Return scalar if input was scalar
    if np.isscalar(annual_risk):
        return float(hazard)
    return hazard


def hazard_to_timestep_probability(
    hazard: Union[float, np.ndarray],
    timestep_duration: float
) -> Union[float, np.ndarray]:
    """
    Convert hazard rate to probability for a specific timestep.
    
    For constant hazard h and duration Δt:
    P(event in timestep) = 1 - exp(-h * Δt)
    
    Parameters
    ----------
    hazard : float or np.ndarray
        Hazard rate (events per unit time)
    timestep_duration : float
        Duration of timestep in same units as hazard
        
    Returns
    -------
    timestep_prob : float or np.ndarray
        Probability of event during timestep
        
    Examples
    --------
    >>> annual_hazard = 0.105  # From 10% annual risk
    >>> weekly_prob = hazard_to_timestep_probability(annual_hazard, 1/52)
    >>> print(f"Weekly probability: {weekly_prob:.4f}")
    Weekly probability: 0.0020
    """
    return 1 - np.exp(-hazard * timestep_duration)


class IncidentGenerator:
    """
    Generates healthcare incidents based on individual patient risks.
    
    Converts annual risks to timestep probabilities and generates
    stochastic incident events while tracking history and statistics.
    
    Parameters
    ----------
    timestep_duration : float, default=1/52
        Duration of each timestep as fraction of year (1/52 for weekly)
        
    Attributes
    ----------
    incident_history : list of np.ndarray
        History of incident occurrences for each timestep
    cumulative_incidents : np.ndarray
        Count of incidents per patient (only first incident counted)
    """
    
    def __init__(self, timestep_duration: float = 1/52):
        """Initialize incident generator."""
        self.timestep_duration = timestep_duration
        self.incident_history: List[np.ndarray] = []
        self.cumulative_incidents: Optional[np.ndarray] = None
    
    def generate_incidents(
        self,
        annual_risks: np.ndarray,
        intervention_mask: Optional[np.ndarray] = None,
        intervention_effectiveness: float = 0.0
    ) -> np.ndarray:
        """
        Generate incidents for one timestep.
        
        Parameters
        ----------
        annual_risks : np.ndarray
            Current annual risk for each patient
        intervention_mask : np.ndarray, optional
            Boolean mask indicating which patients receive intervention
        intervention_effectiveness : float, default=0.0
            Relative risk reduction for treated patients (0.0 to 1.0)
            
        Returns
        -------
        incidents : np.ndarray
            Boolean array indicating which patients had incidents
            
        Examples
        --------
        >>> gen = IncidentGenerator(timestep_duration=1/52)
        >>> risks = np.array([0.1, 0.2, 0.05])
        >>> incidents = gen.generate_incidents(risks)
        >>> print(f"Incidents this week: {np.sum(incidents)}")
        Incidents this week: 0
        """
        n_patients = len(annual_risks)
        
        # Apply intervention effect
        modified_risks = annual_risks.copy()
        if intervention_mask is not None:
            modified_risks[intervention_mask] *= (1 - intervention_effectiveness)
        
        # Convert to hazards
        annual_hazards = annual_risk_to_hazard(modified_risks)
        
        # Convert to timestep probabilities
        timestep_probs = hazard_to_timestep_probability(
            annual_hazards, self.timestep_duration
        )
        
        # Generate incidents
        random_draws = np.random.uniform(0, 1, n_patients)
        incidents = random_draws < timestep_probs
        
        # Update history
        self.incident_history.append(incidents)
        
        # Update cumulative incidents
        if self.cumulative_incidents is None:
            self.cumulative_incidents = incidents.astype(int)
        else:
            # Only count first incident per patient
            new_incidents = incidents & (self.cumulative_incidents == 0)
            self.cumulative_incidents += new_incidents.astype(int)
        
        return incidents
    
    def get_cumulative_incidence(self) -> float:
        """
        Get cumulative incidence rate.
        
        Returns
        -------
        cumulative_incidence : float
            Fraction of patients who have had at least one incident
        """
        if self.cumulative_incidents is None:
            return 0.0
        return np.mean(self.cumulative_incidents > 0)
    
    def reset(self) -> None:
        """Reset incident tracking to initial state."""
        self.incident_history = []
        self.cumulative_incidents = None


class CompetingRiskIncidentGenerator(IncidentGenerator):
    """
    Extended incident generator supporting competing risks and censoring.
    
    Handles multiple types of events that can occur (e.g., readmission vs death)
    and censoring due to loss to follow-up.
    
    Parameters
    ----------
    timestep_duration : float, default=1/52
        Duration of each timestep as fraction of year
    event_types : list of str
        Names of different event types to track
        
    Attributes
    ----------
    event_history : dict
        History of each event type
    cumulative_events : dict
        Cumulative count for each event type
    censored : np.ndarray
        Boolean mask of censored patients
    """
    
    def __init__(
        self,
        timestep_duration: float = 1/52,
        event_types: List[str] = ['primary', 'death']
    ):
        """Initialize competing risk incident generator."""
        super().__init__(timestep_duration)
        self.event_types = event_types
        self.event_history: Dict[str, List[np.ndarray]] = {
            event: [] for event in event_types
        }
        self.cumulative_events: Dict[str, Optional[np.ndarray]] = {
            event: None for event in event_types
        }
        self.censored: Optional[np.ndarray] = None
    
    def generate_competing_incidents(
        self,
        annual_risks_dict: Dict[str, np.ndarray],
        censoring_prob: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Generate incidents for multiple competing event types.
        
        Parameters
        ----------
        annual_risks_dict : dict
            Dictionary mapping event type to annual risk array
        censoring_prob : float, default=0.0
            Probability of censoring per timestep
            
        Returns
        -------
        events_dict : dict
            Dictionary mapping event type to boolean incident array
            
        Examples
        --------
        >>> gen = CompetingRiskIncidentGenerator(event_types=['readmit', 'death'])
        >>> risks = {'readmit': np.array([0.3, 0.2]), 'death': np.array([0.1, 0.05])}
        >>> events = gen.generate_competing_incidents(risks)
        >>> print(f"Readmissions: {np.sum(events['readmit'])}")
        Readmissions: 0
        """
        n_patients = len(list(annual_risks_dict.values())[0])
        
        if self.censored is None:
            self.censored = np.zeros(n_patients, dtype=bool)
        
        # Track which patients are still at risk
        at_risk = ~self.censored
        for event_type in self.event_types:
            if self.cumulative_events[event_type] is not None:
                at_risk &= (self.cumulative_events[event_type] == 0)
        
        events_dict = {}
        
        # Generate events for at-risk patients
        for event_type in self.event_types:
            events = np.zeros(n_patients, dtype=bool)
            
            if event_type in annual_risks_dict and np.any(at_risk):
                # Get risks for this event type
                annual_risks = annual_risks_dict[event_type]
                
                # Convert to timestep probabilities
                annual_hazards = annual_risk_to_hazard(annual_risks[at_risk])
                timestep_probs = hazard_to_timestep_probability(
                    annual_hazards, self.timestep_duration
                )
                
                # Generate events
                random_draws = np.random.uniform(0, 1, np.sum(at_risk))
                events[at_risk] = random_draws < timestep_probs
            
            events_dict[event_type] = events
            
            # Update history
            self.event_history[event_type].append(events)
            
            # Update cumulative
            if self.cumulative_events[event_type] is None:
                self.cumulative_events[event_type] = events.astype(int)
            else:
                new_events = events & (self.cumulative_events[event_type] == 0)
                self.cumulative_events[event_type] += new_events.astype(int)
        
        # Handle censoring
        if censoring_prob > 0:
            new_censoring = np.random.uniform(0, 1, n_patients) < censoring_prob
            new_censoring &= at_risk  # Can only censor at-risk patients
            self.censored |= new_censoring
        
        return events_dict
    
    def get_cumulative_incidence_competing(self) -> Dict[str, float]:
        """
        Get cumulative incidence for each event type accounting for competing risks.
        
        Returns
        -------
        results : dict
            Dictionary mapping event type (and 'censored') to cumulative incidence
        """
        results = {}
        n_patients = len(self.censored) if self.censored is not None else 0
        
        for event_type in self.event_types:
            if self.cumulative_events[event_type] is not None:
                # Only count events that occurred (not censored)
                events_occurred = self.cumulative_events[event_type] > 0
                if self.censored is not None:
                    events_occurred &= ~self.censored
                
                # Cumulative incidence function
                results[event_type] = np.sum(events_occurred) / n_patients if n_patients > 0 else 0.0
            else:
                results[event_type] = 0.0
        
        if self.censored is not None:
            results['censored'] = np.mean(self.censored)
        else:
            results['censored'] = 0.0
            
        return results