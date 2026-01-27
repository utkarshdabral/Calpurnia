"""
Decision Support System (DSS) - Main integration layer
Combines orbital mechanics, AI predictions, and maneuver optimization
into a comprehensive collision risk assessment and avoidance recommendation system.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from src.physics.converter import load_tle, propagate_orbit
from src.physics.conjunction import ConjunctionAssessment, create_default_covariance
from src.physics.maneuver import SimpleReinforcementLearner
from src.ai.residual_predictor import ResidualErrorPredictor, SpaceWeatherDataManager


class ConjunctionAssessmentDecisionSupport:
    """
    Main Decision Support System for conjunction assessment and collision avoidance.
    """
    
    def __init__(
        self,
        keep_out_sphere_km: float = 1.0,
        pc_alert_threshold: float = 1e-4,
        use_ai_correction: bool = True
    ):
        """
        Initialize DSS.
        
        Args:
            keep_out_sphere_km: Keep-out sphere radius (km)
            pc_alert_threshold: Probability of collision threshold for alerts
            use_ai_correction: Whether to use AI-based residual correction
        """
        self.ca = ConjunctionAssessment(keep_out_sphere_km)
        self.pc_alert_threshold = pc_alert_threshold
        self.use_ai_correction = use_ai_correction
        
        self.residual_predictor = ResidualErrorPredictor() if use_ai_correction else None
        self.rl_optimizer = SimpleReinforcementLearner()
        self.space_weather = SpaceWeatherDataManager()
    
    def assess_conjunction_pair(
        self,
        satellite_id_1: int,
        satellite_id_2: int,
        propagation_hours: float = 24.0,
        propagation_steps: int = 100,
        enable_maneuver_planning: bool = True
    ) -> Dict:
        """
        Complete conjunction assessment for satellite pair.
        
        Args:
            satellite_id_1: NORAD catalog ID of satellite 1
            satellite_id_2: NORAD catalog ID of satellite 2
            propagation_hours: Hours ahead to propagate
            propagation_steps: Number of propagation steps
            enable_maneuver_planning: Whether to compute optimal maneuvers
        
        Returns:
            Comprehensive assessment dictionary
        """
        from skyfield.api import Loader
        
        try:
            # Load TLEs
            name1, tle1_line1, tle1_line2 = load_tle(satellite_id_1)
            name2, tle2_line1, tle2_line2 = load_tle(satellite_id_2)
        except Exception as e:
            return {
                'error': f"Failed to load TLEs: {str(e)}",
                'status': 'failed'
            }
        
        # Set up time array
        ts = Loader('data').timescale()
        now = datetime.utcnow()
        
        time_array_seconds = np.linspace(0, propagation_hours * 3600, propagation_steps)
        times = ts.utc(
            now.year, now.month, now.day, now.hour,
            now.minute, now.second + time_array_seconds
        )
        
        # Propagate both satellites
        try:
            pos1, vel1 = propagate_orbit(tle1_line1, tle1_line2, times)
            pos2, vel2 = propagate_orbit(tle2_line1, tle2_line2, times)
        except Exception as e:
            return {
                'error': f"Failed to propagate orbits: {str(e)}",
                'status': 'failed'
            }
        
        # Apply AI residual correction if enabled
        if self.use_ai_correction and self.residual_predictor.model_trained:
            recent_residuals_1 = np.zeros((self.residual_predictor.sequence_length, 3))
            recent_residuals_2 = np.zeros((self.residual_predictor.sequence_length, 3))
            
            pos1_corrected = pos1.copy()
            pos2_corrected = pos2.copy()
            
            for i in range(len(pos1)):
                pos1_corrected[i] = self.residual_predictor.correct_sgp4_prediction(
                    pos1[i], recent_residuals_1
                )
                pos2_corrected[i] = self.residual_predictor.correct_sgp4_prediction(
                    pos2[i], recent_residuals_2
                )
            
            pos1 = pos1_corrected
            pos2 = pos2_corrected
        
        # Compute default covariances
        cov1 = create_default_covariance(sigma_position_m=150.0)  # 150m uncertainty
        cov2 = create_default_covariance(sigma_position_m=200.0)  # 200m uncertainty
        
        # Conjunction assessment
        ca_result = self.ca.assess_conjunction(
            name1, pos1, vel1, cov1,
            name2, pos2, vel2, cov2,
            time_array_seconds,
            pc_threshold=self.pc_alert_threshold,
            monte_carlo_samples=10000  # 10k samples for robust Pc estimation
        )
        
        # Build base result
        result = {
            'status': 'success',
            'timestamp_utc': now.isoformat() + 'Z',
            'propagation_hours': propagation_hours,
            'satellite_1': {
                'name': name1,
                'catalog_id': satellite_id_1,
                'tle': f"{tle1_line1}\n{tle1_line2}"
            },
            'satellite_2': {
                'name': name2,
                'catalog_id': satellite_id_2,
                'tle': f"{tle2_line1}\n{tle2_line2}"
            },
            'conjunction_assessment': ca_result
        }
        
        # Maneuver planning if requested and alert is triggered
        if enable_maneuver_planning and ca_result['alert']:
            try:
                maneuver_result = self.rl_optimizer.optimize_maneuver(
                    pos1[0], vel1[0],
                    pos2[0], vel2[0],
                    ca_result['dca_km'],
                    ca_result['probability_of_collision'],
                    burn_time_utc=now.isoformat() + 'Z',
                    n_candidates=200,
                    top_k=3
                )
                
                result['maneuver_plan'] = {
                    'recommended_maneuver': self._serialize_maneuver(
                        maneuver_result['recommended_maneuver']
                    ),
                    'top_alternatives': [
                        self._serialize_maneuver(m) for m in maneuver_result['top_k_maneuvers'][1:3]
                    ],
                    'performance_metrics': {
                        'initial_dca_km': maneuver_result['initial_dca_km'],
                        'predicted_dca_km': maneuver_result['best_predicted_dca_km'],
                        'initial_pc': maneuver_result['initial_pc'],
                        'predicted_pc': maneuver_result['best_predicted_pc'],
                        'risk_reduction_factor': (
                            maneuver_result['initial_pc'] / (maneuver_result['best_predicted_pc'] + 1e-10)
                        ),
                        'fuel_cost_kg': maneuver_result['best_fuel_cost_kg'],
                    }
                }
            except Exception as e:
                result['maneuver_plan'] = {
                    'error': f"Maneuver optimization failed: {str(e)}"
                }
        
        return result
    
    def _serialize_maneuver(self, maneuver) -> Optional[Dict]:
        """Convert ManeuverAction to serializable dict."""
        if maneuver is None:
            return None
        
        return {
            'delta_v_m_s': {
                'x': maneuver.delta_v_x,
                'y': maneuver.delta_v_y,
                'z': maneuver.delta_v_z,
                'magnitude': np.sqrt(maneuver.delta_v_x**2 + maneuver.delta_v_y**2 + maneuver.delta_v_z**2)
            },
            'burn_time_utc': maneuver.burn_time_utc,
            'fuel_cost_kg': maneuver.fuel_cost_kg,
            'predicted_dca_improvement_km': maneuver.predicted_dca_improvement,
            'risk_reduction_factor': maneuver.risk_reduction
        }
    
    def generate_explainability_report(self, assessment: Dict) -> str:
        """
        Generate human-readable explanation of assessment results.
        
        Args:
            assessment: Assessment result dictionary
        
        Returns:
            Formatted text report
        """
        if assessment['status'] != 'success':
            return f"Assessment failed: {assessment.get('error', 'Unknown error')}"
        
        ca = assessment['conjunction_assessment']
        sat1 = assessment['satellite_1']['name']
        sat2 = assessment['satellite_2']['name']
        
        # Use Monte Carlo Pc if available, else fall back to analytical
        pc_value = ca.get('probability_of_collision_monte_carlo', ca.get('probability_of_collision', 0))
        mc_info = ""
        if 'probability_of_collision_monte_carlo' in ca:
            mc_samples = ca.get('monte_carlo_total_samples', 10000)
            mc_collisions = ca.get('monte_carlo_collision_count', 0)
            mc_info = f"\n  Monte Carlo Results (n={mc_samples:,} samples):"
            mc_info += f"\n    Collision count: {mc_collisions}"
            mc_info += f"\n    Confidence interval: 95% CI ≈ ±{1.96*np.sqrt(pc_value*(1-pc_value)/mc_samples):.2e}"
        
        report = f"""
CONJUNCTION ASSESSMENT REPORT
{'='*80}
Generated: {assessment['timestamp_utc']}

OBJECT PAIR:
  Primary: {sat1} (Catalog ID: {assessment['satellite_1']['catalog_id']})
  Secondary: {sat2} (Catalog ID: {assessment['satellite_2']['catalog_id']})

COLLISION RISK SUMMARY:
  Distance of Closest Approach (DCA): {ca['dca_km']:.3f} km
  Probability of Collision (Pc): {pc_value:.2e}
  Inside Keep-Out Sphere ({ca['keep_out_sphere_km']} km): {ca['inside_keep_out']}
  ALERT STATUS: {'🚨 COLLISION RISK' if ca['alert'] else '✓ Safe'}{mc_info}

ASSESSMENT DETAILS:
  Time of Closest Approach: {ca['tca_seconds']:.1f} seconds from now
  Assessment window: {assessment['propagation_hours']} hours
  
INTERPRETATION:
"""
        
        if ca['alert']:
            report += f"""  
  ⚠️  HIGH RISK CONJUNCTION DETECTED
  
  The probability of collision ({pc_value:.2e}) exceeds the 
  threshold of {self.pc_alert_threshold:.2e}. Immediate action recommended.
  
  At Time of Closest Approach ({ca['tca_seconds']/60:.1f} minutes from now):
    - Objects will approach to {ca['dca_km']:.3f} km
    - Collision sphere radius: {ca['keep_out_sphere_km']} km
    - Risk margin: {ca['dca_km'] - ca['keep_out_sphere_km']:.3f} km
"""
            if 'maneuver_plan' in assessment and 'error' not in assessment['maneuver_plan']:
                mp = assessment['maneuver_plan']
                pm = mp['performance_metrics']
                rec = mp['recommended_maneuver']
                
                report += f"""
RECOMMENDED MANEUVER:
  Delta-V Vector: [{rec['delta_v_m_s']['x']:+.4f}, {rec['delta_v_m_s']['y']:+.4f}, {rec['delta_v_m_s']['z']:+.4f}] m/s
  Delta-V Magnitude: {rec['delta_v_m_s']['magnitude']:.4f} m/s
  Burn Time: {rec['burn_time_utc']}
  Fuel Required: {rec['fuel_cost_kg']:.2f} kg
  
PREDICTED OUTCOME:
  Current DCA: {pm['initial_dca_km']:.3f} km → After maneuver: {pm['predicted_dca_km']:.3f} km
  Current Pc: {pm['initial_pc']:.2e} → After maneuver: {pm['predicted_pc']:.2e}
  Risk Reduction Factor: {pm['risk_reduction_factor']:.1f}x safer
  DCA Improvement: {rec['predicted_dca_improvement_km']:.3f} km
"""
        else:
            report += f"""
  ✓ No collision risk detected.
  
  The closest approach distance of {ca['dca_km']:.3f} km is comfortably above the
  keep-out sphere of {ca['keep_out_sphere_km']} km. No maneuvers required.
"""
        
        report += f"""
{'='*80}
"""
        return report
    
    def export_to_json(self, assessment: Dict, filepath: str):
        """Export assessment to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(assessment, f, indent=2, default=str)
    
    def export_to_html(self, assessment: Dict, filepath: str):
        """Export assessment as HTML dashboard."""
        ca = assessment.get('conjunction_assessment', {})
        alert_class = "alert" if ca.get('alert', False) else "safe"
        alert_text = "🚨 ALERT" if ca.get('alert', False) else "✓ SAFE"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Conjunction Assessment - {assessment['satellite_1']['name']} vs {assessment['satellite_2']['name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .header {{ background: #333; color: white; padding: 15px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 15px 30px 15px 0; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .alert {{ background: #fff3cd; border-left: 5px solid #ff6b6b; padding: 15px; margin: 15px 0; }}
        .safe {{ background: #d4edda; border-left: 5px solid #28a745; padding: 15px; margin: 15px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f9f9f9; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Conjunction Assessment Report</h1>
            <p>Generated: {assessment['timestamp_utc']}</p>
        </div>
        
        <div class="{alert_class}">
            <h2>Status: {alert_text}</h2>
        </div>
        
        <h3>Objects Under Assessment</h3>
        <table>
            <tr><th>Name</th><th>Catalog ID</th></tr>
            <tr><td>{assessment['satellite_1']['name']}</td><td>{assessment['satellite_1']['catalog_id']}</td></tr>
            <tr><td>{assessment['satellite_2']['name']}</td><td>{assessment['satellite_2']['catalog_id']}</td></tr>
        </table>
        
        <h3>Risk Metrics</h3>
        <div class="metric">
            <div class="metric-label">Distance of Closest Approach</div>
            <div class="metric-value">{ca.get('dca_km', 'N/A'):.3f} km</div>
        </div>
        <div class="metric">
            <div class="metric-label">Probability of Collision</div>
            <div class="metric-value">{ca.get('probability_of_collision', 0):.2e}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Keep-Out Sphere</div>
            <div class="metric-value">{ca.get('keep_out_sphere_km', 'N/A')} km</div>
        </div>
"""
        
        if 'maneuver_plan' in assessment and 'error' not in assessment.get('maneuver_plan', {}):
            mp = assessment['maneuver_plan']
            rec = mp['recommended_maneuver']
            pm = mp['performance_metrics']
            
            html += f"""
        <h3>Recommended Maneuver</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Δv X</td><td>{rec['delta_v_m_s']['x']:+.4f} m/s</td></tr>
            <tr><td>Δv Y</td><td>{rec['delta_v_m_s']['y']:+.4f} m/s</td></tr>
            <tr><td>Δv Z</td><td>{rec['delta_v_m_s']['z']:+.4f} m/s</td></tr>
            <tr><td>Δv Magnitude</td><td>{rec['delta_v_m_s']['magnitude']:.4f} m/s</td></tr>
            <tr><td>Fuel Cost</td><td>{rec['fuel_cost_kg']:.2f} kg</td></tr>
            <tr><td>Burn Time</td><td>{rec['burn_time_utc']}</td></tr>
        </table>
        
        <h3>Predicted Outcome</h3>
        <table>
            <tr><th>Metric</th><th>Before</th><th>After</th></tr>
            <tr><td>DCA</td><td>{pm['initial_dca_km']:.3f} km</td><td>{pm['predicted_dca_km']:.3f} km</td></tr>
            <tr><td>Pc</td><td>{pm['initial_pc']:.2e}</td><td>{pm['predicted_pc']:.2e}</td></tr>
            <tr><td>Risk Reduction Factor</td><td colspan="2">{pm['risk_reduction_factor']:.1f}x</td></tr>
        </table>
"""
        
        html += """
    </div>
</body>
</html>
"""
        with open(filepath, 'w') as f:
            f.write(html)
