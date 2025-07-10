#!/usr/bin/env python3
"""
Interactive Results Analysis for Hydra Simulation Outputs

This script provides comprehensive analysis tools for exploring simulation results
from the VectorizedTemporalRiskSimulator experiments.

Usage:
    python analyze_results.py outputs/baseline_simulation/
    python analyze_results.py --help
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

class SimulationResultsAnalyzer:
    """Comprehensive analysis tool for simulation results."""
    
    def __init__(self, results_dir: str):
        """Initialize analyzer with results directory."""
        self.results_dir = Path(results_dir)
        self.data = {}
        self.stats = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all available data files."""
        print(f"📂 Loading data from {self.results_dir}")
        
        # Load summary statistics
        stats_file = self.results_dir / "summary_statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
            print("✓ Summary statistics loaded")
        
        # Load CSV data files
        data_files = {
            'patient_risks': 'patient_risks.csv',
            'temporal_matrix': 'temporal_risk_matrix.csv',
            'incidents': 'incident_matrix.csv',
            'interventions': 'intervention_matrix.csv',
            'counterfactuals': 'counterfactual_incidents.csv',
            'ml_predictions': 'ml_predictions.csv'
        }
        
        for key, filename in data_files.items():
            file_path = self.results_dir / filename
            if file_path.exists():
                self.data[key] = pd.read_csv(file_path)
                print(f"✓ {key} loaded ({self.data[key].shape})")
            else:
                print(f"⚠️  {key} not found")
    
    def print_summary(self):
        """Print comprehensive summary of results."""
        print("\n" + "="*60)
        print("🏥 HEALTHCARE AI SIMULATION RESULTS SUMMARY")
        print("="*60)
        
        if not self.stats:
            print("❌ No summary statistics available")
            return
        
        # Simulation parameters
        sim_params = self.stats.get('simulation_parameters', {})
        print(f"\n📊 SIMULATION PARAMETERS")
        print(f"   Patients: {sim_params.get('n_patients', 'N/A'):,}")
        print(f"   Timesteps: {sim_params.get('n_timesteps', 'N/A')}")
        print(f"   Prediction times: {sim_params.get('prediction_times', 'N/A')}")
        print(f"   Intervention effectiveness: {sim_params.get('intervention_effectiveness', 'N/A'):.1%}")
        
        # Population statistics
        pop_stats = self.stats.get('population_statistics', {})
        print(f"\n👥 POPULATION CHARACTERISTICS")
        print(f"   Mean base risk: {pop_stats.get('mean_base_risk', 0):.3f}")
        print(f"   Risk std dev: {pop_stats.get('std_base_risk', 0):.3f}")
        print(f"   Risk range: {pop_stats.get('min_base_risk', 0):.6f} - {pop_stats.get('max_base_risk', 0):.3f}")
        
        # Intervention statistics
        int_stats = self.stats.get('intervention_statistics', {})
        print(f"\n🎯 INTERVENTION ANALYSIS")
        print(f"   Total interventions: {int_stats.get('total_interventions', 0):,}")
        print(f"   Unique patients treated: {int_stats.get('unique_patients_treated', 0):,}")
        print(f"   Coverage: {int_stats.get('intervention_coverage', 0):.1%}")
        
        # Outcome statistics
        outcome_stats = self.stats.get('outcome_statistics', {})
        cf_stats = self.stats.get('counterfactual_statistics', {})
        print(f"\n📈 OUTCOMES & EFFECTIVENESS")
        print(f"   Actual incidents: {outcome_stats.get('total_incidents', 0):,}")
        print(f"   Counterfactual incidents: {cf_stats.get('counterfactual_incidents', 0):,}")
        
        incident_reduction = cf_stats.get('incident_reduction', 0)
        if incident_reduction > 0:
            print(f"   ✅ Incident reduction: {incident_reduction:.1%}")
        else:
            print(f"   ❌ Incident increase: {abs(incident_reduction):.1%}")
        
        # ML Performance
        ml_stats = self.stats.get('ml_prediction_statistics', {})
        if ml_stats:
            print(f"\n🤖 ML MODEL PERFORMANCE")
            print(f"   Mean prediction score: {ml_stats.get('mean_ml_score', 0):.3f}")
            print(f"   Positive prediction rate: {ml_stats.get('positive_prediction_rate', 0):.1%}")
            print(f"   Total predictions: {ml_stats.get('total_predictions', 0):,}")
        
        # Validation results
        validation = self.stats.get('validation_results', {})
        if validation:
            print(f"\n✅ VALIDATION STATUS")
            for check, passed in validation.items():
                status = "✅ PASSED" if passed else "❌ FAILED"
                print(f"   {check.replace('_', ' ').title()}: {status}")
    
    def create_overview_dashboard(self, figsize=(16, 12)):
        """Create comprehensive overview dashboard."""
        print("\n📊 Creating overview dashboard...")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Healthcare AI Intervention Simulation - Overview Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Risk Distribution
        if 'patient_risks' in self.data:
            ax1 = axes[0, 0]
            risks = self.data['patient_risks']['base_risk']
            ax1.hist(risks, bins=50, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Base Annual Risk')
            ax1.set_ylabel('Number of Patients')
            ax1.set_title('Patient Risk Distribution')
            ax1.grid(True, alpha=0.3)
        
        # 2. Temporal Risk Evolution
        if 'temporal_matrix' in self.data:
            ax2 = axes[0, 1]
            temporal_data = self.data['temporal_matrix']
            
            # Sample 100 patients for visualization
            n_patients = min(100, len(temporal_data))
            sample_indices = np.random.choice(len(temporal_data), n_patients, replace=False)
            
            for i in sample_indices:
                patient_risks = temporal_data.iloc[i].values
                ax2.plot(patient_risks, alpha=0.1, color='blue')
            
            # Plot mean risk over time
            mean_risks = temporal_data.mean(axis=0)
            ax2.plot(mean_risks, color='red', linewidth=2, label='Population Mean')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Risk Level')
            ax2.set_title('Temporal Risk Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Intervention Coverage
        if 'interventions' in self.data:
            ax3 = axes[0, 2]
            intervention_data = self.data['interventions']
            coverage_by_time = intervention_data.sum(axis=0)
            ax3.plot(coverage_by_time)
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Number of Interventions')
            ax3.set_title('Interventions Over Time')
            ax3.grid(True, alpha=0.3)
        
        # 4. ML Prediction Distribution
        if 'ml_predictions' in self.data:
            ax4 = axes[1, 0]
            ml_data = self.data['ml_predictions']
            
            # Assuming ml_predictions has columns for each prediction time
            prediction_cols = [col for col in ml_data.columns if 'prediction' in col.lower()]
            if prediction_cols:
                all_predictions = ml_data[prediction_cols].values.flatten()
                all_predictions = all_predictions[~np.isnan(all_predictions)]
                ax4.hist(all_predictions, bins=50, alpha=0.7, edgecolor='black')
                ax4.set_xlabel('ML Prediction Score')
                ax4.set_ylabel('Frequency')
                ax4.set_title('ML Prediction Distribution')
                ax4.grid(True, alpha=0.3)
        
        # 5. Incident Analysis
        if 'incidents' in self.data and 'counterfactuals' in self.data:
            ax5 = axes[1, 1]
            incidents = self.data['incidents'].sum(axis=0)
            counterfactuals = self.data['counterfactuals'].sum(axis=0)
            
            time_steps = range(len(incidents))
            ax5.plot(time_steps, incidents, label='Actual Incidents', linewidth=2)
            ax5.plot(time_steps, counterfactuals, label='Counterfactual', linewidth=2, linestyle='--')
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('Number of Incidents')
            ax5.set_title('Incident Comparison')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Performance Metrics Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create text summary
        if self.stats:
            sim_params = self.stats.get('simulation_parameters', {})
            int_stats = self.stats.get('intervention_statistics', {})
            outcome_stats = self.stats.get('outcome_statistics', {})
            cf_stats = self.stats.get('counterfactual_statistics', {})
            
            summary_text = f"""
SIMULATION SUMMARY

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
            
            ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        # Save the dashboard
        output_file = self.results_dir / 'analysis_dashboard.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved to {output_file}")
        
        plt.show()
        return fig
    
    def analyze_intervention_effectiveness(self):
        """Analyze intervention effectiveness in detail."""
        print("\n🎯 Analyzing intervention effectiveness...")
        
        if 'incidents' not in self.data or 'counterfactuals' not in self.data:
            print("❌ Missing incident or counterfactual data")
            return
        
        incidents = self.data['incidents']
        counterfactuals = self.data['counterfactuals']
        
        # Calculate effectiveness by time period
        actual_by_time = incidents.sum(axis=0)
        counterfactual_by_time = counterfactuals.sum(axis=0)
        effectiveness_by_time = (counterfactual_by_time - actual_by_time) / counterfactual_by_time
        
        # Calculate effectiveness by patient risk level
        if 'patient_risks' in self.data:
            risks = self.data['patient_risks']['base_risk']
            
            # Create risk quartiles
            risk_quartiles = pd.qcut(risks, 4, labels=['Low', 'Medium', 'High', 'Very High'])
            
            effectiveness_by_risk = []
            for quartile in ['Low', 'Medium', 'High', 'Very High']:
                mask = risk_quartiles == quartile
                actual_incidents = incidents.iloc[mask].sum().sum()
                counterfactual_incidents = counterfactuals.iloc[mask].sum().sum()
                
                if counterfactual_incidents > 0:
                    effectiveness = (counterfactual_incidents - actual_incidents) / counterfactual_incidents
                else:
                    effectiveness = 0
                
                effectiveness_by_risk.append({
                    'risk_group': quartile,
                    'actual_incidents': actual_incidents,
                    'counterfactual_incidents': counterfactual_incidents,
                    'effectiveness': effectiveness
                })
            
            effectiveness_df = pd.DataFrame(effectiveness_by_risk)
            print("\n📊 Effectiveness by Risk Group:")
            print(effectiveness_df.to_string(index=False))
        
        # Create effectiveness visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Time-based effectiveness
        ax1.plot(effectiveness_by_time, marker='o')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Effectiveness Rate')
        ax1.set_title('Intervention Effectiveness Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Risk-based effectiveness
        if 'patient_risks' in self.data:
            ax2.bar(effectiveness_df['risk_group'], effectiveness_df['effectiveness'])
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Risk Group')
            ax2.set_ylabel('Effectiveness Rate')
            ax2.set_title('Intervention Effectiveness by Risk Group')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save effectiveness analysis
        output_file = self.results_dir / 'intervention_effectiveness.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Effectiveness analysis saved to {output_file}")
        
        plt.show()
        
        return effectiveness_by_time, effectiveness_df if 'patient_risks' in self.data else None
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on results."""
        print("\n💡 RECOMMENDATIONS")
        print("="*40)
        
        if not self.stats:
            print("❌ No statistics available for recommendations")
            return
        
        recommendations = []
        
        # Check intervention coverage
        int_stats = self.stats.get('intervention_statistics', {})
        coverage = int_stats.get('intervention_coverage', 0)
        
        if coverage < 0.05:
            recommendations.append(
                "📈 LOW INTERVENTION COVERAGE: Consider lowering the ML threshold "
                f"(current coverage: {coverage:.1%}). Target range: 5-50%"
            )
        elif coverage > 0.5:
            recommendations.append(
                "📉 HIGH INTERVENTION COVERAGE: Consider raising the ML threshold "
                f"(current coverage: {coverage:.1%}) to focus on highest-risk patients"
            )
        
        # Check incident reduction
        cf_stats = self.stats.get('counterfactual_statistics', {})
        incident_reduction = cf_stats.get('incident_reduction', 0)
        
        if incident_reduction <= 0:
            recommendations.append(
                "⚠️  NEGATIVE EFFECTIVENESS: Interventions appear to increase incidents. "
                "Consider: (1) Adjusting intervention effectiveness parameter, "
                "(2) Improving ML model performance, (3) Different assignment strategy"
            )
        elif incident_reduction < 0.1:
            recommendations.append(
                "📊 LOW EFFECTIVENESS: Consider increasing intervention effectiveness "
                f"parameter (current reduction: {incident_reduction:.1%})"
            )
        
        # Check ML performance
        ml_stats = self.stats.get('ml_prediction_statistics', {})
        if ml_stats:
            mean_score = ml_stats.get('mean_ml_score', 0)
            if mean_score < 0.3:
                recommendations.append(
                    "🤖 LOW ML SCORES: Mean prediction score is low. Consider "
                    "adjusting target PPV/sensitivity parameters"
                )
        
        # Check validation failures
        validation = self.stats.get('validation_results', {})
        failed_checks = [check for check, passed in validation.items() if not passed]
        
        if failed_checks:
            recommendations.append(
                f"✅ VALIDATION FAILURES: {', '.join(failed_checks)}. "
                "Review simulation parameters to meet validation criteria"
            )
        
        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✅ Results look good! No major issues identified.")
        
        print(f"\n📁 All analysis files saved to: {self.results_dir}")
        
        return recommendations


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze Healthcare AI Simulation Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py outputs/baseline_simulation/
  python analyze_results.py outputs/baseline_simulation/ --dashboard-only
        """
    )
    
    parser.add_argument(
        'results_dir',
        help='Path to simulation results directory'
    )
    
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Only create dashboard, skip detailed analysis'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    try:
        analyzer = SimulationResultsAnalyzer(args.results_dir)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return 1
    
    # Run analysis
    try:
        analyzer.print_summary()
        
        if not args.dashboard_only:
            analyzer.analyze_intervention_effectiveness()
            analyzer.generate_recommendations()
        
        analyzer.create_overview_dashboard()
        
        print("\n🎉 Analysis complete!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    exit(main())