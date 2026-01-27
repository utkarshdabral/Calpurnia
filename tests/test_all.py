"""
Unit Tests and Usage Examples for Calpurnia
Demonstrates all major components with realistic scenarios
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from src.physics.converter import load_tle, propagate_orbit
from src.physics.conjunction import ConjunctionAssessment, create_default_covariance
from src.physics.maneuver import SimpleReinforcementLearner, ManeuverAction
from src.ai.residual_predictor import ResidualErrorPredictor, SpaceWeatherDataManager
from src.physics.dss import ConjunctionAssessmentDecisionSupport


class TestOrbitalPropagation:
    """Test SGP4 propagation accuracy"""
    
    @staticmethod
    def test_iss_propagation():
        """Test ISS propagation produces reasonable values"""
        name, tle1, tle2 = load_tle(25544)
        
        from skyfield.api import Loader
        ts = Loader('data').timescale()
        now = datetime.utcnow()
        
        times_array = np.linspace(0, 3600, 10)  # 1 hour, 10 points
        times = ts.utc(now.year, now.month, now.day, now.hour,
                       now.minute, now.second + times_array)
        
        pos, vel = propagate_orbit(tle1, tle2, times)
        
        assert pos.shape == (10, 3), f"Position shape mismatch: {pos.shape}"
        assert vel.shape == (10, 3), f"Velocity shape mismatch: {vel.shape}"
        
        # ISS orbits at ~6700-6900 km altitude (radius ~6700-6900 km from Earth center)
        dist_earth = np.linalg.norm(pos[0])
        assert 6600 < dist_earth < 7000, f"ISS distance {dist_earth} outside expected range"
        
        # ISS speed ~7.7 km/s
        speed = np.linalg.norm(vel[0])
        assert 7.0 < speed < 8.0, f"ISS speed {speed} outside expected range"
        
        print("✓ ISS propagation test passed")
    
    @staticmethod
    def test_propagation_continuity():
        """Test that propagation is smooth (no jumps)"""
        name, tle1, tle2 = load_tle(25544)
        
        from skyfield.api import Loader
        ts = Loader('data').timescale()
        now = datetime.utcnow()
        
        times_array = np.linspace(0, 3600, 100)
        times = ts.utc(now.year, now.month, now.day, now.hour,
                       now.minute, now.second + times_array)
        
        pos, vel = propagate_orbit(tle1, tle2, times)
        
        # Check differences between consecutive points
        pos_diff = np.diff(pos, axis=0)
        max_step = np.max(np.linalg.norm(pos_diff, axis=1))
        
        # In 36 seconds (~1 orbital period), expect ~270 km movement
        assert max_step < 300, f"Position step {max_step} km too large"
        
        print("✓ Propagation continuity test passed")


class TestConjunctionAssessment:
    """Test collision risk assessment"""
    
    @staticmethod
    def test_zero_separation():
        """Test DCA computation with same positions"""
        ca = ConjunctionAssessment(keep_out_sphere_radius=1.0)
        
        pos = np.array([6700, 0, 0])
        vel = np.array([0, 7, 0])
        time_secs = np.array([0, 100, 200])
        
        dca, tca, idx = ca.compute_distance_of_closest_approach(
            pos, vel, pos, vel, time_secs
        )
        
        assert dca < 1e-6, f"DCA for same position should be ~0, got {dca}"
        print("✓ Zero separation test passed")
    
    @staticmethod
    def test_parallel_motion():
        """Test with satellites moving in parallel (no collision)"""
        ca = ConjunctionAssessment(keep_out_sphere_radius=1.0)
        
        pos1 = np.array([6700, 0, 0])
        pos2 = np.array([6702, 0, 0])  # 2 km apart
        vel = np.array([0, 7, 0])  # Same velocity
        time_secs = np.array([0, 100, 200, 300])
        
        dca, tca, idx = ca.compute_distance_of_closest_approach(
            pos1, vel, pos2, vel, time_secs
        )
        
        assert 1.9 < dca < 2.1, f"DCA should remain ~2 km, got {dca}"
        print("✓ Parallel motion test passed")
    
    @staticmethod
    def test_collision_probability():
        """Test collision probability computation"""
        ca = ConjunctionAssessment(keep_out_sphere_radius=1.0)
        
        dca = 0.5  # 500m
        cov = create_default_covariance(sigma_position_m=100)
        rel_vel = np.array([0.1, 0.1, 0.1])
        
        pc = ca.compute_probability_of_collision(dca, cov, cov, rel_vel)
        
        assert 0 <= pc <= 1, f"Pc should be in [0,1], got {pc}"
        assert pc > 0.5, f"High pc expected (DCA < uncertainty), got {pc}"
        
        print("✓ Collision probability test passed")


class TestReinforcementLearning:
    """Test maneuver optimization"""
    
    @staticmethod
    def test_fuel_cost():
        """Test Tsiolkovsky fuel cost calculation"""
        rl = SimpleReinforcementLearner(spacecraft_mass=6500, isp=300)
        
        fuel_0 = rl.compute_fuel_cost(0)
        fuel_100 = rl.compute_fuel_cost(100)
        fuel_500 = rl.compute_fuel_cost(500)
        
        assert fuel_0 < 1e-6, "Zero delta-v should require ~0 fuel"
        assert fuel_100 > fuel_0, "More delta-v requires more fuel"
        assert fuel_500 > fuel_100, "Exponential growth in fuel cost"
        
        print(f"✓ Fuel cost test passed (Δv=0: {fuel_0:.2e} kg, "
              f"Δv=100: {fuel_100:.2f} kg, Δv=500: {fuel_500:.2f} kg)")
    
    @staticmethod
    def test_maneuver_generation():
        """Test candidate maneuver generation"""
        rl = SimpleReinforcementLearner(max_delta_v=0.5)
        
        candidates = rl.generate_candidate_maneuvers(n_candidates=100, seed=42)
        
        assert candidates.shape == (100, 3), f"Shape mismatch: {candidates.shape}"
        
        # Check magnitudes
        mags = np.linalg.norm(candidates, axis=1)
        assert np.all(mags <= 0.5), f"Some candidates exceed max_delta_v"
        assert np.any(mags > 0.4), f"Some large-magnitude candidates expected"
        
        print(f"✓ Maneuver generation test passed")
    
    @staticmethod
    def test_maneuver_optimization():
        """Test full optimization"""
        rl = SimpleReinforcementLearner()
        
        pos1 = np.array([6700.0, 0.0, 0.0])
        vel1 = np.array([0.0, 7.0, 0.0])
        pos2 = np.array([6705.0, 0.0, 0.0])
        vel2 = np.array([0.0, 6.9, 0.0])
        
        result = rl.optimize_maneuver(
            pos1, vel1, pos2, vel2,
            dca_initial=5.0,
            pc_initial=1e-4,
            n_candidates=50,
            top_k=3
        )
        
        assert result['recommended_maneuver'] is not None
        assert len(result['top_k_maneuvers']) == 3
        assert result['best_reward_score'] > -100  # Reasonable score
        
        print("✓ Maneuver optimization test passed")


class TestResidualPrediction:
    """Test AI residual error prediction"""
    
    @staticmethod
    def test_sequence_creation():
        """Test sequence generation for LSTM"""
        pred = ResidualErrorPredictor(sequence_length=10)
        
        sgp4_pos = np.random.randn(50, 3)
        actual_pos = sgp4_pos + np.random.randn(50, 3) * 0.1
        
        X, y = pred.create_sequences(sgp4_pos, actual_pos, sequence_length=10)
        
        assert X.shape == (40, 10, 3), f"X shape mismatch: {X.shape}"
        assert y.shape == (40, 3), f"y shape mismatch: {y.shape}"
        
        print("✓ Sequence creation test passed")
    
    @staticmethod
    def test_residual_prediction():
        """Test residual prediction"""
        pred = ResidualErrorPredictor(sequence_length=10)
        
        recent_residuals = np.random.randn(10, 3) * 0.1
        predicted = pred.predict_residual(recent_residuals, steps_ahead=6)
        
        assert predicted.shape == (6, 3), f"Shape mismatch: {predicted.shape}"
        assert np.all(np.isfinite(predicted)), "Non-finite values"
        
        print("✓ Residual prediction test passed")


class TestSpaceWeather:
    """Test space weather data management"""
    
    @staticmethod
    def test_f107_range():
        """Test F10.7 is in reasonable range"""
        sw = SpaceWeatherDataManager()
        
        f107 = sw.get_f107_daily(datetime.utcnow())
        assert 50 < f107 < 250, f"F10.7 {f107} outside typical range"
        
        print(f"✓ F10.7 test passed (value: {f107:.1f})")
    
    @staticmethod
    def test_ap_index_range():
        """Test Ap index is in reasonable range"""
        sw = SpaceWeatherDataManager()
        
        ap = sw.get_ap_index(datetime.utcnow())
        assert 0 < ap < 200, f"Ap {ap} outside typical range"
        
        print(f"✓ Ap index test passed (value: {ap:.1f})")


class TestDecisionSupportSystem:
    """Test integrated DSS"""
    
    @staticmethod
    def test_dss_initialization():
        """Test DSS can be created"""
        dss = ConjunctionAssessmentDecisionSupport(
            keep_out_sphere_km=1.5,
            pc_alert_threshold=1e-4,
            use_ai_correction=False
        )
        
        assert dss.ca is not None
        assert dss.rl_optimizer is not None
        
        print("✓ DSS initialization test passed")


# Usage Examples
def example_1_simple_propagation():
    """Example 1: Simple orbital propagation"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Orbital Propagation")
    print("="*60)
    
    from skyfield.api import Loader
    
    # Load TLE
    name, tle1, tle2 = load_tle(25544)
    print(f"Loaded: {name}")
    
    # Propagate 1 hour
    ts = Loader('data').timescale()
    now = datetime.utcnow()
    times_array = np.linspace(0, 3600, 10)
    times = ts.utc(now.year, now.month, now.day, now.hour,
                   now.minute, now.second + times_array)
    
    pos, vel = propagate_orbit(tle1, tle2, times)
    
    print(f"\nPropagated {len(pos)} positions over 1 hour")
    print(f"Initial position: {pos[0]} km")
    print(f"Initial velocity: {vel[0]} km/s")
    print(f"Initial orbital speed: {np.linalg.norm(vel[0]):.3f} km/s")


def example_2_collision_assessment():
    """Example 2: Collision probability assessment"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Collision Probability Assessment")
    print("="*60)
    
    # Create synthetic close approach
    ca = ConjunctionAssessment(keep_out_sphere_radius=2.0)
    
    pos1 = np.array([6700.0, 0.0, 0.0])
    vel1 = np.array([0.0, 7.0, 0.0])
    cov1 = create_default_covariance(100)
    
    pos2 = np.array([6702.0, 0.0, 0.0])
    vel2 = np.array([0.0, 6.95, 0.0])
    cov2 = create_default_covariance(100)
    
    time_secs = np.array([0, 100, 200, 300, 400, 500])
    
    result = ca.assess_conjunction(
        "SAT1", pos1, vel1, cov1,
        "SAT2", pos2, vel2, cov2,
        time_secs,
        pc_threshold=1e-4
    )
    
    print(f"\nConjunction Assessment Result:")
    print(f"  DCA: {result['dca_km']:.3f} km")
    print(f"  TCA: {result['tca_seconds']:.1f} seconds")
    print(f"  Pc: {result['probability_of_collision']:.2e}")
    print(f"  Alert: {result['alert']}")


def example_3_maneuver_planning():
    """Example 3: Optimal maneuver planning"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Optimal Maneuver Planning")
    print("="*60)
    
    rl = SimpleReinforcementLearner(max_delta_v=0.5)
    
    # Conjunction scenario
    pos1 = np.array([6700.0, 0.0, 0.0])
    vel1 = np.array([0.0, 7.0, 0.0])
    pos2 = np.array([6703.0, 0.0, 0.0])
    vel2 = np.array([0.0, 6.9, 0.0])
    
    print(f"\nScenario:")
    print(f"  Sat1 position: {pos1} km")
    print(f"  Sat2 position: {pos2} km (3 km separation)")
    print(f"  Initial DCA: {np.linalg.norm(pos1-pos2):.3f} km")
    
    result = rl.optimize_maneuver(
        pos1, vel1, pos2, vel2,
        dca_initial=3.0,
        pc_initial=1e-5,
        n_candidates=100,
        top_k=3
    )
    
    rec = result['recommended_maneuver']
    print(f"\nRecommended Maneuver:")
    print(f"  Δv: [{rec.delta_v_x:+.4f}, {rec.delta_v_y:+.4f}, {rec.delta_v_z:+.4f}] m/s")
    print(f"  Magnitude: {np.sqrt(rec.delta_v_x**2 + rec.delta_v_y**2 + rec.delta_v_z**2):.4f} m/s")
    print(f"  Fuel cost: {rec.fuel_cost_kg:.3f} kg")
    print(f"  DCA improvement: {rec.predicted_dca_improvement:+.3f} km")


if __name__ == "__main__":
    print("\n" + "█"*60)
    print("CALPURNIA: Unit Tests & Examples")
    print("█"*60)
    
    # Run tests
    print("\n--- RUNNING TESTS ---")
    try:
        TestOrbitalPropagation.test_iss_propagation()
        TestOrbitalPropagation.test_propagation_continuity()
    except Exception as e:
        print(f"✗ Propagation tests failed: {e}")
    
    try:
        TestConjunctionAssessment.test_zero_separation()
        TestConjunctionAssessment.test_parallel_motion()
        TestConjunctionAssessment.test_collision_probability()
    except Exception as e:
        print(f"✗ Conjunction tests failed: {e}")
    
    try:
        TestReinforcementLearning.test_fuel_cost()
        TestReinforcementLearning.test_maneuver_generation()
        TestReinforcementLearning.test_maneuver_optimization()
    except Exception as e:
        print(f"✗ RL tests failed: {e}")
    
    try:
        TestResidualPrediction.test_sequence_creation()
        TestResidualPrediction.test_residual_prediction()
    except Exception as e:
        print(f"✗ Residual prediction tests failed: {e}")
    
    try:
        TestSpaceWeather.test_f107_range()
        TestSpaceWeather.test_ap_index_range()
    except Exception as e:
        print(f"✗ Space weather tests failed: {e}")
    
    try:
        TestDecisionSupportSystem.test_dss_initialization()
    except Exception as e:
        print(f"✗ DSS tests failed: {e}")
    
    # Run examples
    print("\n--- RUNNING EXAMPLES ---")
    try:
        example_1_simple_propagation()
    except Exception as e:
        print(f"✗ Example 1 failed: {e}")
    
    try:
        example_2_collision_assessment()
    except Exception as e:
        print(f"✗ Example 2 failed: {e}")
    
    try:
        example_3_maneuver_planning()
    except Exception as e:
        print(f"✗ Example 3 failed: {e}")
    
    print("\n" + "█"*60)
    print("✓ All tests and examples completed!")
    print("█"*60)
