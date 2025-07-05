
# Healthcare AI Temporal Simulation Framework

## The Challenge of Healthcare AI Evaluation

Healthcare AI systems promise to revolutionize patient care by identifying high-risk individuals for early intervention. However, evaluating these systems presents a fundamental challenge: in real-world deployments, we never observe the counterfactual—what would have happened without the AI intervention. This causes a particular challenge when the AI system is set to predict the risk of an unwanted outcome, so that an intervention is deployed to reduce the risk of occurrence. This is often the case, creating an unobservable outcome.

**Example 1: Stroke Prevention Program**
Consider an AI system that predicts which patients are at high risk of stroke within the next 12 months. When the model identifies a high-risk patient, the clinical team intervenes with preventive measures—perhaps prescribing anticoagulants, intensifying blood pressure management, or enrolling the patient in a specialized stroke prevention clinic. If the patient doesn't have a stroke, we face an attribution problem: Did the AI-guided intervention prevent the stroke, or was the patient never going to have one? The very success of the intervention obscures our ability to measure the AI's accuracy and impact.

Or in a case where an AI program outcomes are completely observable but there are confounding factors that make attribution difficult.

**Example 2: Breast Cancer Screening Program**
An AI system identifies patients at elevated risk for breast cancer, prompting earlier or more frequent mammogram scheduling. When these patients receive timely screening, we can observe whether cancer is detected. However, attribution remains challenging: Did the AI system improve early detection rates, or were these improvements due to concurrent initiatives? Perhaps the health system simultaneously launched a general awareness campaign, hired additional scheduling staff, or implemented reminder systems that improved screening rates across all patients. The presence of these confounding factors makes it difficult to isolate the AI's specific contribution to improved outcomes.

**Example 3: Heart Failure Readmission Prevention Program**
A health system deploys an AI model to predict which patients are at high risk of readmission within 30 days after heart failure hospitalization. The model triggers automatic enrollment in a post-discharge monitoring program with daily weight checks, medication reminders, and nurse phone calls. This creates a complex evaluation challenge with multiple layers:

 - Partial Observability: Some high-risk patients will be readmitted despite the intervention (observable), while others avoid readmission (counterfactual unknown)
- Time-Varying Effects: The intervention's effectiveness may decay over time - very effective in week 1, less so by week 4
- Spillover Effects: The enhanced monitoring program might improve care protocols for all heart failure patients, not just those flagged by the AI
- Competing Events: Patients might die before the 30-day window ends, creating censoring issues in the analysis


Traditional RCTs can address this but are expensive, time-consuming, and ethically complex. This creates a critical need for sophisticated simulation frameworks that can model realistic healthcare scenarios with known ground truth, as well as sophisticated operational monitoring methods to ensure these programs maintain their impact over time.

## Project Purpose Statement

This project is a temporal simulation framework that enables comprehensive causal inference analysis of AI-guided healthcare interventions. The system will support Regression Discontinuity Design (RDD), Difference-in-Differences (DiD), Interrupted Time Series (ITS), and Synthetic Control methods through a modular, incrementally buildable architecture.

**Primary Goal:** Build a temporal simulation framework for testing causal inference methods with known ground truth

**Secondary Goals:**
- Explore and Validate AI intervention deployment design and strategies
- Train operations and research on engineering design and causal inference methods
- Compare method performance across scenarios
- Enable policy testing before real-world implementation

## Framework Components

The simulator comprises:
- **Population Simulation:** Age-stratified cohorts with realistic disease incidence using hazard-based modeling
- **Machine Learning Simulation:** Generate models that hit exact PPV and Sensitivity targets
- **Clinical Intervention + Causal Analysis:** Model intervention effects with known ground truth

## What This Framework Enables

### Real-World Applications:
1. **AI Pilot Studies:** Test different AI deployment strategies before real implementation
2. **Resource Planning:** Understand staffing needs for different intervention scenarios
3. **Policy Evaluation:** Simulate the effect of changing treatment guidelines
4. **Threshold Optimization:** Find required model performance and optimal risk cutoffs for interventions
5. **Method Validation:** Test whether your causal inference approach can detect true effects

### Causal Inference Capabilities:
- **Known Ground Truth:** Unlike real studies, you KNOW the true causal effect
- **Perfect Counterfactuals:** You can see what would have happened without intervention
- **Method Comparison:** Test RDD vs DiD vs ITS on the same data
- **Power Analysis:** Determine sample sizes needed to detect effects
- **Robustness Testing:** See how sensitive your estimates are to violations of assumptions

## Technical Implementation

### 
### Introduction: Building Realistic Healthcare AI Simulations

The core strategy of this simulation framework is to balance multiple levels of fidelity to achieve our goal of enabling robust design decisions and requirements setting for healthcare AI interventions. Rather than attempting to model every clinical detail, we focus on the essential elements that drive causal inference validity while maintaining computational efficiency.

**Simulation Strategy: Targeted Fidelity**

Our approach prioritizes mathematical rigor in the areas that matter most for causal inference:

-   **High fidelity** for population risk distributions, temporal dynamics, and intervention mechanics
-   **Simplified abstractions** for clinical workflows and patient characteristics that don't affect causal estimates. Simulating targeted machine learning model characteristics.
-   **Known ground truth** at every level, enabling validation of inference methods

This targeted approach allows us to simulate populations of 100,000+ patients over multi-year horizons while maintaining the statistical properties necessary for valid causal inference.

**Core Simulation Components**

**1. Heterogeneous Patient Risk** Real patient populations exhibit extreme risk heterogeneity - most patients have minimal risk while a small fraction drive the majority of events. It's likely the true risk distribution is unknown but it's reasonbable to start with a  beta-distributed risk assignment to capture this reality, ensuring that:

-   Intervention effects vary realistically across the population
-   Resource requirements concentrate among truly high-risk patients
-   ML models face realistic discrimination challenges
-   Population-level incident rates remain precisely controlled

As the program progresses, the development team will create a trusted risk model that may show non-beta distribution.  The team can change the base distribution model that could more closely represent mixed distributions.  The code design should enable modular risk distributions 

**2. Time-Varying Risk** Patient risk fluctuates due to seasonal patterns, life events, disease progression, and healthcare interactions. Our AR(1) temporal risk process introduces these dynamics while maintaining mathematical tractability, enabling:

-   Realistic prediction window challenges
-   Time-dependent intervention effects
-   Natural variation that tests model robustness
-   Temporal patterns that support time-series methods (ITS, DiD)

**3. ML Model Simulation** Rather than training actual ML models, we simulate their behavior based on the true underlying risk function plus calibrated noise. This approach:

-   Generates risk scores with specified PPV and sensitivity
-   Models prediction windows matching clinical reality
-   Allows precise control of model performance parameters
-   Enables rapid testing of different performance scenarios

**4. Quasi-Experimental Method Support** The simulation mechanics are specifically designed to enable evaluation across different causal inference approaches:

-   **RDD**: Continuous risk scores with configurable thresholds for intervention
-   **DiD**: Pre/post periods with treatment/control group assignments
-   **ITS**: Long time series with clear intervention points
-   **Synthetic Control**: Multiple units with staggered intervention timing

**5. Comprehensive Sensitivity Analysis** The framework enables systematic exploration of key parameters:

-   **Policy variations**: Different risk thresholds, intervention criteria, resource constraints
-   **Intervention efficacy**: From minimal (5%) to substantial (50%) risk reduction
-   **Model performance**: PPV from 20-60%, sensitivity from 30-80%
-   **Population characteristics**: Varying base rates, risk distributions, temporal patterns

This design philosophy - focused fidelity where it matters, elegant simplification elsewhere - creates a platform for healthcare AI evaluation that bridges the gap between theoretical causal inference and practical healthcare operations.

### Patient Risk Assignment Process

#### Phase 1: Individual Risk Initialization (One-Time Setup)

**1. Beta-Distributed Individual Risk Assignment**

Each patient receives a unique individual annual incident risk sampled from a beta distribution:

```python
# Distribution parameters for right-skewed shape
concentration = 0.5
alpha = concentration
beta_param = alpha * (1/annual_incident_rate - 1)

# Sample all patient risks at once
raw_risks = np.random.beta(alpha, beta_param, n_patients)

# Scale to ensure population mean equals target
scaling_factor = annual_incident_rate / np.mean(raw_risks)
base_annual_risks = np.clip(raw_risks * scaling_factor, 0, 0.99)
```

**2. Risk Score Storage**

Store each patient's individual annual risk score for efficient access during temporal simulation.

#### Phase 2: Temporal Incident Simulation (Each Timestep)

**1. Risk Modification by Intervention Status**
- Treatment Group: `Modified_risk = Individual_risk × (1 - intervention_effectiveness)`
- Control Group: `Modified_risk = Individual_risk`

**2. Convert Annual Risk to Monthly Hazard**
```python
monthly_hazard = -ln(1 - modified_annual_risk) / 12
```

**3. Convert Monthly Hazard to Timestep Probability**
```python
timestep_probability = 1 - exp(-monthly_hazard × δt)
```

**4. Incident Outcome Determination**
- Generate random number for each patient
- If random_number ≤ timestep_probability, patient has incident

**5. Population-Level Validation**
- Track cumulative incidents
- Compare against expected total with tolerance (±5%)

### Time-Varying Individual Risk Implementation

For each patient i at time t:
```
risk_i(t) = base_risk_i × temporal_modifier_i(t)
```

Temporal modifier follows AR(1) process:
```
temporal_modifier_i(t) = ρ × temporal_modifier_i(t-1) + (1-ρ) × 1.0 + ε_i(t)
```
where:
- ρ = 0.8-0.95 (persistence parameter)
- ε_i(t) ~ N(0, σ²) with σ = 0.1-0.2
- Bounded to [0.5, 2.0].       

###  Competing Risks and Censoring
The simulation should create competing risk of death and censorship.

-   **Competing Risks**: Multiple types of events (e.g., death vs readmission)
-   **Censoring**: Incomplete observation (e.g., patient leaves study)                    

### ML Model Simulation with Prediction Windows

Healthcare ML models forecast risk over clinically meaningful windows:
- "30% chance of readmission within 30 days"
- "High risk of stroke within 12 months"

**True Label Generation:**
```python
def get_true_label(patient_i, current_time, prediction_window):
    """True label = 1 if incident occurs within prediction_window"""
    future_times = range(current_time + 1, current_time + prediction_window + 1)
    for t in future_times:
        if incident_occurs(patient_i, t):
            return 1
    return 0
```

### Vectorized Implementation

```python
class VectorizedTemporalRiskSimulator:
    def __init__(self, n_patients, n_timesteps, annual_incident_rate,
                 prediction_window=12, intervention_effectiveness=0.2):
        self.n_patients = n_patients
        self.n_timesteps = n_timesteps
        self.annual_incident_rate = annual_incident_rate
        self.prediction_window = prediction_window
        self.intervention_effectiveness = intervention_effectiveness
        
        # Initialize everything upfront
        self._initialize_base_risks()
        self._initialize_temporal_modifiers()
        self._assign_intervention_groups()
    
    def _initialize_base_risks(self):
        """Vectorized initialization of heterogeneous base risks."""
        # Beta distribution parameters
        concentration = 0.5
        alpha = concentration
        beta_param = alpha * (1/self.annual_incident_rate - 1)
        
        # Sample all patient risks at once
        raw_risks = np.random.beta(alpha, beta_param, self.n_patients)
        
        # Scale to ensure population mean equals target
        scaling_factor = self.annual_incident_rate / np.mean(raw_risks)
        self.base_annual_risks = np.clip(raw_risks * scaling_factor, 0, 0.99)
        
        # Convert to monthly hazards
        self.base_monthly_hazards = -np.log(1 - self.base_annual_risks) / 12
    
    def simulate_all_incidents(self):
        """Vectorized simulation of all incidents across all timesteps."""
        # Pre-compute time-varying hazards
        time_varying_hazards = (
            self.base_monthly_hazards[:, np.newaxis] * 
            self.temporal_modifiers
        )
        
        # Apply intervention effect
        intervention_multiplier = np.where(
            self.is_treatment[:, np.newaxis],
            1 - self.intervention_effectiveness,
            1.0
        )
        modified_hazards = time_varying_hazards * intervention_multiplier
        
        # Convert to probabilities
        monthly_probs = 1 - np.exp(-modified_hazards)
        
        # Generate all random draws at once
        random_draws = np.random.random((self.n_patients, self.n_timesteps))
        
        # Determine incidents
        self.incident_matrix = random_draws < monthly_probs
```

## Key Design Principles

1. **Vectorization is Essential**
   - Eliminates patient-level loops (100,000x speedup)
   - Pre-computes everything possible
   - Uses broadcasting for element-wise operations
   - Leverages numpy's C-level optimizations

2. **Maintain Population Constraints**
   - Beta distribution parameters chosen to yield exact population mean risk
   - AR(1) temporal modifiers centered at 1.0
   - Proper hazard-to-probability conversions

3. **Realistic ML Performance**
   - Moderate PPV (20-40%)
   - Variable sensitivity (30-70%)
   - Isotonic regression calibration

## Simulation Performance

The entire simulation for 100k patients × 48 months should run in seconds rather than minutes, enabling:
- Rapid experimentation with different parameters
- Monte Carlo validation across multiple random seeds
- Real-time interactive exploration of intervention strategies
- Scaling to million-patient simulations when needed

## Next Steps

1. Implement causal inference methods (RDD, DiD, ITS)
2. Create visualization tools for results
3. Develop sensitivity analysis capabilities
4. Build user-friendly configuration interface
5. Validate against real-world healthcare data patterns

# Causal Inference Methods for Healthcare AI Evaluation

Based on your simulation framework design, here's an instructive guide to implementing and evaluating the three quasi-experimental methods within your healthcare AI temporal simulation system.

## Regression Discontinuity Design (RDD)

### Method Introduction

Regression Discontinuity Design exploits situations where treatment assignment depends on whether a continuous "running variable" (in healthcare AI, typically a risk score) crosses a predetermined threshold. The key insight: patients just above and just below the threshold are essentially similar except for their treatment status, creating a natural experiment at the boundary.

In healthcare AI contexts, RDD is particularly powerful because:

-   Risk scores are continuous but interventions are binary (treat/don't treat)
-   Thresholds are often set by clinical guidelines or resource constraints
-   The discontinuity provides causal identification without randomization

### Simulation Requirements for RDD

Your simulation must provide:

1.  **Continuous Risk Scores**: The ML model must output granular risk predictions (not just binary classifications)
2.  **Sharp Threshold Assignment**: Treatment strictly determined by score crossing threshold
3.  **Sufficient Sample Near Threshold**: Adequate patients with scores close to the cutoff
4.  **No Manipulation**: Patients/providers cannot game their scores to cross the threshold
5.  **Smooth Baseline Risk**: The true underlying risk function should be continuous across the threshold

### Implementation Example

```python
class RDDAnalysis:
    def __init__(self, simulator, risk_threshold=0.3, bandwidth=0.1):
        self.simulator = simulator
        self.risk_threshold = risk_threshold
        self.bandwidth = bandwidth
        
    def analyze_intervention_effect(self):
        # Get risk scores and outcomes
        risk_scores = self.simulator.ml_risk_scores
        outcomes = self.simulator.incident_matrix.any(axis=1)  # Any incident
        treated = risk_scores >= self.risk_threshold
        
        # Focus on patients near threshold
        near_threshold = np.abs(risk_scores - self.risk_threshold) <= self.bandwidth
        
        # Local linear regression on each side
        left_side = (risk_scores < self.risk_threshold) & near_threshold
        right_side = (risk_scores >= self.risk_threshold) & near_threshold
        
        # Estimate discontinuity at threshold
        from sklearn.linear_model import LinearRegression
        
        # Fit separate regressions
        X_left = (risk_scores[left_side] - self.risk_threshold).reshape(-1, 1)
        y_left = outcomes[left_side]
        model_left = LinearRegression().fit(X_left, y_left)
        
        X_right = (risk_scores[right_side] - self.risk_threshold).reshape(-1, 1)
        y_right = outcomes[right_side]
        model_right = LinearRegression().fit(X_right, y_right)
        
        # Estimate at threshold
        left_limit = model_left.predict([[0]])[0]
        right_limit = model_right.predict([[0]])[0]
        
        # RDD estimate of causal effect
        rdd_effect = right_limit - left_limit
        
        return {
            'estimated_effect': rdd_effect,
            'true_effect': self.simulator.intervention_effectiveness,
            'n_near_threshold': near_threshold.sum(),
            'bandwidth_used': self.bandwidth
        }

```

### Validation Example

```python
# Run simulation with known intervention effect
simulator = VectorizedTemporalRiskSimulator(
    n_patients=100000,
    n_timesteps=12,
    annual_incident_rate=0.05,
    intervention_effectiveness=0.25  # True 25% risk reduction
)

# Generate ML scores and assign treatment
simulator.generate_ml_scores(ppv=0.35, sensitivity=0.60)
simulator.assign_treatment_by_threshold(threshold=0.3)
simulator.simulate_all_incidents()

# Apply RDD analysis
rdd = RDDAnalysis(simulator, risk_threshold=0.3)
results = rdd.analyze_intervention_effect()

print(f"True intervention effect: {results['true_effect']}")
print(f"RDD estimated effect: {results['estimated_effect']:.3f}")
print(f"Estimation error: {abs(results['estimated_effect'] - results['true_effect']):.3f}")

```

## Difference-in-Differences (DiD)

### Method Introduction

Difference-in-Differences compares changes over time between a treatment group and a control group. The method assumes that without the intervention, both groups would have followed parallel trends. By comparing the difference in outcomes before and after intervention between groups, DiD isolates the causal effect.

In healthcare AI deployment:

-   Some clinics/units adopt the AI system while others don't
-   Phased rollouts create natural treatment and control groups
-   Historical data provides the "before" period

### Simulation Requirements for DiD

Your simulation must provide:

1.  **Pre/Post Periods**: Clear time periods before and after intervention
2.  **Treatment/Control Groups**: Distinct groups with and without intervention
3.  **Parallel Trends**: Similar outcome trends in absence of treatment
4.  **No Spillovers**: Treatment in one group doesn't affect control group
5.  **Temporal Stability**: Consistent data collection across periods

### Implementation Example

```python
class DiDAnalysis:
    def __init__(self, simulator, intervention_start_time=24):
        self.simulator = simulator
        self.intervention_start = intervention_start_time
        
    def analyze_intervention_effect(self):
        # Split time into pre/post periods
        pre_period = np.arange(0, self.intervention_start)
        post_period = np.arange(self.intervention_start, self.simulator.n_timesteps)
        
        # Get treatment/control groups
        treated = self.simulator.is_treatment
        control = ~treated
        
        # Calculate incident rates for each group/period
        incidents = self.simulator.incident_matrix
        
        # Pre-period rates
        pre_treated = incidents[treated][:, pre_period].mean()
        pre_control = incidents[control][:, pre_period].mean()
        
        # Post-period rates
        post_treated = incidents[treated][:, post_period].mean()
        post_control = incidents[control][:, post_period].mean()
        
        # DiD estimate
        did_effect = (post_treated - pre_treated) - (post_control - pre_control)
        
        # Convert to relative effect
        baseline_risk = pre_control
        relative_effect = -did_effect / baseline_risk  # Negative because reduction
        
        return {
            'estimated_effect': relative_effect,
            'true_effect': self.simulator.intervention_effectiveness,
            'pre_treated': pre_treated,
            'pre_control': pre_control,
            'post_treated': post_treated,
            'post_control': post_control,
            'did_estimate': did_effect
        }
        
    def test_parallel_trends(self):
        """Validate parallel trends assumption in pre-period"""
        pre_period = np.arange(0, self.intervention_start)
        treated = self.simulator.is_treatment
        
        # Monthly incident rates by group
        treated_trends = self.simulator.incident_matrix[treated][:, pre_period].mean(axis=0)
        control_trends = self.simulator.incident_matrix[~treated][:, pre_period].mean(axis=0)
        
        # Test if trends are parallel (similar slopes)
        from scipy import stats
        time_points = np.arange(len(pre_period))
        
        treated_slope = stats.linregress(time_points, treated_trends).slope
        control_slope = stats.linregress(time_points, control_trends).slope
        
        return {
            'treated_slope': treated_slope,
            'control_slope': control_slope,
            'slope_difference': abs(treated_slope - control_slope),
            'parallel': abs(treated_slope - control_slope) < 0.001
        }

```

### Validation Example

```python
# Configure simulation for DiD
simulator = VectorizedTemporalRiskSimulator(
    n_patients=50000,
    n_timesteps=48,  # 4 years monthly
    annual_incident_rate=0.08,
    intervention_effectiveness=0.30
)

# Randomly assign treatment/control groups
simulator.assign_treatment_groups(treatment_fraction=0.5)

# No intervention in first 24 months (pre-period)
simulator.intervention_start_time = 24
simulator.simulate_all_incidents()

# Apply DiD analysis
did = DiDAnalysis(simulator, intervention_start_time=24)

# Check parallel trends assumption
trends = did.test_parallel_trends()
print(f"Parallel trends test - Slope difference: {trends['slope_difference']:.5f}")

# Estimate intervention effect
results = did.analyze_intervention_effect()
print(f"\nDiD Results:")
print(f"True effect: {results['true_effect']}")
print(f"Estimated effect: {results['estimated_effect']:.3f}")
print(f"Pre-period: Treated={results['pre_treated']:.4f}, Control={results['pre_control']:.4f}")
print(f"Post-period: Treated={results['post_treated']:.4f}, Control={results['post_control']:.4f}")

```

## Interrupted Time Series (ITS)

### Method Introduction

Interrupted Time Series analysis examines outcome trends before and after an intervention point, testing whether the intervention caused a change in level (immediate effect) or slope (gradual effect) of the outcome trajectory. Unlike DiD, ITS doesn't require a control group but relies on the counterfactual assumption that pre-intervention trends would have continued.

In healthcare AI:

-   System-wide AI deployments create clear interruption points
-   Long baseline periods establish robust trend estimates
-   Can detect both immediate and gradual intervention effects

### Simulation Requirements for ITS

Your simulation must provide:

1.  **Long Time Series**: Sufficient pre/post observations (typically 12+ each)
2.  **Clear Intervention Point**: Precise timing of AI system activation
3.  **Stable Pre-Trends**: Consistent patterns before intervention
4.  **No Concurrent Changes**: Other system changes don't coincide with AI launch
5.  **Autocorrelation Handling**: Account for temporal dependencies

### Implementation Example

```python
class ITSAnalysis:
    def __init__(self, simulator, intervention_time=24):
        self.simulator = simulator
        self.intervention_time = intervention_time
        
    def analyze_intervention_effect(self):
        # Aggregate incident rates across all patients
        monthly_rates = self.simulator.incident_matrix.mean(axis=0)
        
        # Create time variable
        time = np.arange(len(monthly_rates))
        post_intervention = (time >= self.intervention_time).astype(int)
        time_since_intervention = np.maximum(0, time - self.intervention_time)
        
        # Segmented regression model
        from statsmodels.api import OLS
        import statsmodels.api as sm
        
        # Design matrix: intercept, time, level_change, slope_change
        X = np.column_stack([
            np.ones_like(time),  # Intercept
            time,                # Pre-intervention trend
            post_intervention,   # Level change at intervention
            time_since_intervention  # Slope change after intervention
        ])
        
        # Fit model
        model = OLS(monthly_rates, X).fit()
        
        # Extract coefficients
        baseline_level = model.params[0]
        baseline_slope = model.params[1]
        level_change = model.params[2]
        slope_change = model.params[3]
        
        # Calculate relative effect at end of study
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
            'true_effect': self.simulator.intervention_effectiveness,
            'durbin_watson': sm.stats.durbin_watson(model.resid)
        }
    
    def plot_its_results(self, results):
        import matplotlib.pyplot as plt
        
        time = np.arange(self.simulator.n_timesteps)
        monthly_rates = self.simulator.incident_matrix.mean(axis=0)
        
        # Predicted values from model
        predicted = results['model'].predict()
        
        # Counterfactual (what would have happened without intervention)
        counterfactual = (results['baseline_level'] + 
                         results['baseline_slope'] * time)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, monthly_rates, 'o', label='Observed', alpha=0.5)
        plt.plot(time, predicted, '-', label='ITS Model', linewidth=2)
        plt.plot(time[self.intervention_time:], 
                counterfactual[self.intervention_time:], 
                '--', label='Counterfactual', linewidth=2)
        plt.axvline(self.intervention_time, color='red', 
                   linestyle=':', label='Intervention')
        
        plt.xlabel('Time (months)')
        plt.ylabel('Monthly Incident Rate')
        plt.title('Interrupted Time Series Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()

```

### Validation Example

```python
# Configure simulation for ITS
simulator = VectorizedTemporalRiskSimulator(
    n_patients=100000,
    n_timesteps=48,  # 4 years
    annual_incident_rate=0.06,
    intervention_effectiveness=0.20
)

# All patients receive intervention after time point
simulator.intervention_time = 24
simulator.apply_population_wide_intervention(start_time=24)
simulator.simulate_all_incidents()

# Apply ITS analysis
its = ITSAnalysis(simulator, intervention_time=24)
results = its.analyze_intervention_effect()

print("ITS Results:")
print(f"Baseline monthly rate: {results['baseline_level']:.4f}")
print(f"Baseline trend: {results['baseline_slope']:.6f} per month")
print(f"Level change at intervention: {results['level_change']:.4f}")
print(f"Slope change after intervention: {results['slope_change']:.6f}")
print(f"\nTrue effect: {results['true_effect']}")
print(f"Estimated relative effect: {results['relative_effect']:.3f}")
print(f"Durbin-Watson statistic: {results['durbin_watson']:.2f}")

# Visualize results
fig = its.plot_its_results(results)

```

## Summary: Method Selection Guide

Method

Best For

Key Requirements

Simulation Fidelity Needs

**RDD**

Risk score-based interventions

Continuous scores, clear threshold

Accurate ML risk scores near threshold

**DiD**

Phased rollouts, geographic variation

Treatment/control groups, pre/post data

Parallel baseline trends, no spillovers

**ITS**

System-wide deployments

Long time series, clear start date

Stable pre-trends, isolated intervention

Each method has distinct advantages:

-   **RDD** provides strong causal identification at the threshold but limited generalizability
-   **DiD** handles time-varying confounders but requires untreated comparison groups
-   **ITS** works without control groups but is vulnerable to concurrent changes

Your simulation framework enables testing all three methods on the same underlying data, revealing which approach best suits different deployment scenarios and validating that your causal estimates recover the true known intervention effects.
