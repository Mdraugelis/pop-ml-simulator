"""Population-based Machine Learning Simulator package."""

from typing import List

from .sample import add
from .risk_distribution import assign_patient_risks, simulate_annual_incidents
from .temporal_dynamics import (
    TemporalRiskSimulator,
    EnhancedTemporalRiskSimulator,
    simulate_ar1_process,
    build_temporal_risk_matrix
)
from .hazard_modeling import (
    annual_risk_to_hazard,
    hazard_to_timestep_probability,
    IncidentGenerator,
    CompetingRiskIncidentGenerator
)
from .ml_simulation import (
    MLPredictionSimulator,
    calculate_theoretical_performance_bounds,
    hosmer_lemeshow_test,
    evaluate_threshold_based,
    evaluate_topk,
    optimize_alert_threshold,
    analyze_risk_stratified_performance,
    generate_temporal_ml_predictions,
    validate_temporal_sensitivity,
    benchmark_temporal_ml_performance,
    analyze_patient_journey_enhanced,
    analyze_patients_by_outcome,
    plot_ai_quadrant
)
from .risk_integration import (
    integrate_window_risk,
    extract_risk_windows,
    validate_integration_bounds
)
from .vectorized_simulator import (
    VectorizedTemporalRiskSimulator,
    SimulationResults
)

__all__: List[str] = [
    "add",
    # Risk distribution functions
    "assign_patient_risks",
    "simulate_annual_incidents",
    # Temporal dynamics classes and functions
    "TemporalRiskSimulator",
    "EnhancedTemporalRiskSimulator",
    "simulate_ar1_process",
    "build_temporal_risk_matrix",
    # Hazard modeling functions and classes
    "annual_risk_to_hazard",
    "hazard_to_timestep_probability",
    "IncidentGenerator",
    "CompetingRiskIncidentGenerator",
    # ML simulation functions and classes
    "MLPredictionSimulator",
    "calculate_theoretical_performance_bounds",
    "hosmer_lemeshow_test",
    "evaluate_threshold_based",
    "evaluate_topk",
    "optimize_alert_threshold",
    "analyze_risk_stratified_performance",
    # Risk integration functions
    "integrate_window_risk",
    "extract_risk_windows",
    "validate_integration_bounds",
    # Temporal ML functions
    "generate_temporal_ml_predictions",
    "validate_temporal_sensitivity",
    "benchmark_temporal_ml_performance",
    # AI Quadrant visualization functions
    "analyze_patient_journey_enhanced",
    "analyze_patients_by_outcome",
    "plot_ai_quadrant",
    # Vectorized simulator
    "VectorizedTemporalRiskSimulator",
    "SimulationResults",
]
__version__ = "0.1.0"
