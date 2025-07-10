#!/usr/bin/env python3
"""
Quick Results Viewer - Matplotlib Only

Simple script to view Hydra simulation results using only matplotlib.
No additional dependencies required beyond basic requirements.

Usage:
    python quick_results_view.py outputs/baseline_simulation/
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(results_dir):
    """Load all simulation results."""
    results_path = Path(results_dir)
    
    print(f"📂 Loading results from: {results_path}")
    
    # Load summary statistics
    stats_file = results_path / "summary_statistics.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print("✅ Summary statistics loaded")
    else:
        stats = {}
        print("⚠️ Summary statistics not found")
    
    # Load data files
    data = {}
    data_files = {
        'patient_risks': 'patient_risks.csv',
        'temporal_matrix': 'temporal_risk_matrix.csv',
        'incidents': 'incident_matrix.csv',
        'interventions': 'intervention_matrix.csv',
        'counterfactuals': 'counterfactual_incidents.csv',
        'ml_predictions': 'ml_predictions.csv'
    }
    
    for key, filename in data_files.items():
        file_path = results_path / filename
        if file_path.exists():
            data[key] = pd.read_csv(file_path)
            print(f"✅ {key}: {data[key].shape}")
        else:
            print(f"⚠️ {key}: not found")
    
    return stats, data

def print_summary(stats):
    """Print comprehensive summary."""
    if not stats:
        print("❌ No summary statistics available")
        return
    
    print("\n" + "="*60)
    print("🏥 HEALTHCARE AI SIMULATION RESULTS")
    print("="*60)
    
    # Simulation parameters
    sim_params = stats.get('simulation_parameters', {})
    print(f"\n📊 SIMULATION SETUP")
    print(f"   Patients: {sim_params.get('n_patients', 'N/A'):,}")
    print(f"   Timesteps: {sim_params.get('n_timesteps', 'N/A')}")
    print(f"   Intervention effectiveness: {sim_params.get('intervention_effectiveness', 0):.1%}")
    print(f"   Prediction times: {sim_params.get('prediction_times', [])}")
    
    # Population statistics
    pop_stats = stats.get('population_statistics', {})
    print(f"\n👥 POPULATION")
    print(f"   Mean base risk: {pop_stats.get('mean_base_risk', 0):.3f}")
    print(f"   Risk std dev: {pop_stats.get('std_base_risk', 0):.3f}")
    print(f"   Risk range: {pop_stats.get('min_base_risk', 0):.6f} - {pop_stats.get('max_base_risk', 0):.3f}")
    
    # Intervention statistics
    int_stats = stats.get('intervention_statistics', {})
    print(f"\n🎯 INTERVENTIONS")
    print(f"   Total interventions: {int_stats.get('total_interventions', 0):,}")
    print(f"   Unique patients treated: {int_stats.get('unique_patients_treated', 0):,}")
    print(f"   Coverage: {int_stats.get('intervention_coverage', 0):.1%}")
    
    # Outcome statistics
    outcome_stats = stats.get('outcome_statistics', {})
    cf_stats = stats.get('counterfactual_statistics', {})
    print(f"\n📈 OUTCOMES & EFFECTIVENESS")
    print(f"   Actual incidents: {outcome_stats.get('total_incidents', 0):,}")
    print(f"   Counterfactual incidents: {cf_stats.get('counterfactual_incidents', 0):,}")
    
    incident_reduction = cf_stats.get('incident_reduction', 0)
    if incident_reduction > 0:
        print(f"   ✅ Incident reduction: {incident_reduction:.1%}")
    else:
        print(f"   ❌ Incident increase: {abs(incident_reduction):.1%}")
    
    # ML Performance
    ml_stats = stats.get('ml_prediction_statistics', {})
    if ml_stats:
        print(f"\n🤖 ML PERFORMANCE")
        print(f"   Mean prediction score: {ml_stats.get('mean_ml_score', 0):.3f}")
        print(f"   Positive prediction rate: {ml_stats.get('positive_prediction_rate', 0):.1%}")
        print(f"   Total predictions: {ml_stats.get('total_predictions', 0):,}")
    
    # Validation results
    validation = stats.get('validation_results', {})
    if validation:
        print(f"\n✅ VALIDATION STATUS")
        for check, passed in validation.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {check.replace('_', ' ').title()}: {status}")

def create_dashboard(stats, data, output_path):
    """Create comprehensive matplotlib dashboard."""
    print("\n📊 Creating results dashboard...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Healthcare AI Intervention Simulation - Results Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Risk Distribution
    if 'patient_risks' in data:
        ax = axes[0, 0]
        risks = data['patient_risks']['base_risk']
        ax.hist(risks, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Base Annual Risk')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Patient Risk Distribution')
        ax.grid(True, alpha=0.3)
    
    # 2. Temporal Risk Evolution (sample)
    if 'temporal_matrix' in data:
        ax = axes[0, 1]
        temporal_data = data['temporal_matrix']
        
        # Sample 50 patients for visualization
        n_sample = min(50, len(temporal_data))
        sample_indices = np.random.choice(len(temporal_data), n_sample, replace=False)
        
        for i in sample_indices:
            patient_risks = temporal_data.iloc[i].values
            ax.plot(patient_risks, alpha=0.2, color='blue')
        
        # Plot mean risk over time
        mean_risks = temporal_data.mean(axis=0)
        ax.plot(mean_risks, color='red', linewidth=3, label='Population Mean')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Risk Level')
        ax.set_title('Temporal Risk Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Intervention Coverage Over Time
    if 'interventions' in data:
        ax = axes[0, 2]
        intervention_data = data['interventions']
        coverage_by_time = intervention_data.sum(axis=0)
        ax.plot(coverage_by_time, color='green')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Number of Interventions')
        ax.set_title('Interventions Over Time')
        ax.grid(True, alpha=0.3)
    
    # 4. ML Prediction Distribution
    if 'ml_predictions' in data:
        ax = axes[1, 0]
        ml_data = data['ml_predictions']
        
        # Get all prediction columns
        prediction_cols = [col for col in ml_data.columns if col != ml_data.columns[0]]  # Skip index column
        if prediction_cols:
            all_predictions = []
            for col in prediction_cols:
                valid_preds = ml_data[col].dropna()
                all_predictions.extend(valid_preds)
            
            if all_predictions:
                ax.hist(all_predictions, bins=50, alpha=0.7, edgecolor='black', color='orange')
                ax.set_xlabel('ML Prediction Score')
                ax.set_ylabel('Frequency')
                ax.set_title('ML Prediction Distribution')
                ax.grid(True, alpha=0.3)
    
    # 5. Incident Comparison
    if 'incidents' in data and 'counterfactuals' in data:
        ax = axes[1, 1]
        incidents = data['incidents'].sum(axis=0)
        counterfactuals = data['counterfactuals'].sum(axis=0)
        
        time_steps = range(len(incidents))
        ax.plot(time_steps, incidents, label='Actual Incidents', linewidth=2, color='red')
        ax.plot(time_steps, counterfactuals, label='Counterfactual', linewidth=2, linestyle='--', color='orange')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Number of Incidents')
        ax.set_title('Incident Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Risk vs Prediction Scatter (if available)
    if 'patient_risks' in data and 'ml_predictions' in data:
        ax = axes[1, 2]
        risks = data['patient_risks']['base_risk']
        ml_data = data['ml_predictions']
        
        # Get first prediction column
        prediction_cols = [col for col in ml_data.columns if col != ml_data.columns[0]]
        if prediction_cols and len(ml_data) > 0:
            predictions = ml_data[prediction_cols[0]].dropna()
            
            # Ensure we have matching lengths
            min_len = min(len(risks), len(predictions))
            if min_len > 0:
                matching_risks = risks.iloc[:min_len]
                matching_predictions = predictions.iloc[:min_len]
                
                ax.scatter(matching_risks, matching_predictions, alpha=0.6, s=20)
                ax.set_xlabel('Base Risk')
                ax.set_ylabel('ML Prediction Score')
                ax.set_title('Risk vs ML Prediction')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No matching data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Risk vs ML Prediction')
        else:
            ax.text(0.5, 0.5, 'No prediction data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk vs ML Prediction')
    
    # 7. Effectiveness by Time Period
    if 'incidents' in data and 'counterfactuals' in data:
        ax = axes[2, 0]
        incidents = data['incidents'].sum(axis=0)
        counterfactuals = data['counterfactuals'].sum(axis=0)
        
        effectiveness = np.where(counterfactuals > 0, 
                                (counterfactuals - incidents) / counterfactuals, 0)
        
        ax.plot(effectiveness, color='purple')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Effectiveness Rate')
        ax.set_title('Intervention Effectiveness Over Time')
        ax.grid(True, alpha=0.3)
    
    # 8. Patient Risk Quintiles
    if 'patient_risks' in data:
        ax = axes[2, 1]
        risks = data['patient_risks']['base_risk']
        
        # Create risk quintiles
        quintiles = pd.qcut(risks, 5, labels=['Low', 'Low-Med', 'Med', 'Med-High', 'High'])
        quintile_counts = quintiles.value_counts().sort_index()
        
        ax.bar(range(len(quintile_counts)), quintile_counts.values, color='skyblue')
        ax.set_xticks(range(len(quintile_counts)))
        ax.set_xticklabels(quintile_counts.index, rotation=45)
        ax.set_ylabel('Number of Patients')
        ax.set_title('Patients by Risk Quintile')
        ax.grid(True, alpha=0.3)
    
    # 9. Summary Statistics Text
    ax = axes[2, 2]
    ax.axis('off')
    
    if stats:
        sim_params = stats.get('simulation_parameters', {})
        int_stats = stats.get('intervention_statistics', {})
        outcome_stats = stats.get('outcome_statistics', {})
        cf_stats = stats.get('counterfactual_statistics', {})
        
        summary_text = f"""SIMULATION SUMMARY

Patients: {sim_params.get('n_patients', 'N/A'):,}
Timesteps: {sim_params.get('n_timesteps', 'N/A')}

INTERVENTIONS
Total: {int_stats.get('total_interventions', 0):,}
Coverage: {int_stats.get('intervention_coverage', 0):.1%}
Unique patients: {int_stats.get('unique_patients_treated', 0):,}

OUTCOMES
Actual incidents: {outcome_stats.get('total_incidents', 0):,}
Counterfactual: {cf_stats.get('counterfactual_incidents', 0):,}
Reduction: {cf_stats.get('incident_reduction', 0):.1%}
"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the dashboard
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Dashboard saved to {output_path}")
    
    plt.show()

def generate_recommendations(stats):
    """Generate actionable recommendations."""
    print("\n💡 RECOMMENDATIONS")
    print("="*40)
    
    if not stats:
        print("❌ No statistics available for recommendations")
        return
    
    recommendations = []
    
    # Check intervention coverage
    int_stats = stats.get('intervention_statistics', {})
    coverage = int_stats.get('intervention_coverage', 0)
    
    if coverage < 0.05:
        recommendations.append(
            f"📈 LOW COVERAGE: Coverage is {coverage:.1%}. Lower the ML threshold (try 0.3 instead of 0.5)"
        )
    elif coverage > 0.5:
        recommendations.append(
            f"📉 HIGH COVERAGE: Coverage is {coverage:.1%}. Raise the ML threshold for better targeting"
        )
    
    # Check incident reduction
    cf_stats = stats.get('counterfactual_statistics', {})
    incident_reduction = cf_stats.get('incident_reduction', 0)
    
    if incident_reduction <= 0:
        recommendations.append(
            "⚠️ NEGATIVE EFFECTIVENESS: Try increasing intervention effectiveness parameter to 0.4"
        )
    elif incident_reduction < 0.1:
        recommendations.append(
            f"📊 LOW EFFECTIVENESS: {incident_reduction:.1%} reduction. Consider parameter tuning"
        )
    
    # Check validation failures
    validation = stats.get('validation_results', {})
    failed_checks = [check for check, passed in validation.items() if not passed]
    
    if failed_checks:
        recommendations.append(
            f"✅ VALIDATION ISSUES: {', '.join(failed_checks)}. Review parameters"
        )
    
    # Print recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✅ Results look good! No major issues identified.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick Hydra Results Viewer")
    parser.add_argument('results_dir', help='Path to simulation results directory')
    
    args = parser.parse_args()
    
    try:
        # Load results
        stats, data = load_results(args.results_dir)
        
        # Print summary
        print_summary(stats)
        
        # Create dashboard
        output_path = Path(args.results_dir) / 'quick_dashboard.png'
        create_dashboard(stats, data, output_path)
        
        # Generate recommendations
        generate_recommendations(stats)
        
        print(f"\n📁 Results saved to: {args.results_dir}")
        print("🎉 Quick analysis complete!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())