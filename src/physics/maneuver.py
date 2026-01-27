"""
Avoidance Maneuver Planning using Reinforcement Learning.
Optimizes delta-v burns to maximize safety while minimizing fuel consumption.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ManeuverAction:
    """Represents a proposed maneuver action."""
    delta_v_x: float  # m/s
    delta_v_y: float  # m/s
    delta_v_z: float  # m/s
    burn_time_utc: str
    fuel_cost_kg: float
    predicted_dca_improvement: float  # km
    risk_reduction: float  # Fraction (0 to 1)


class SimpleReinforcementLearner:
    """
    Simplified RL-based maneuver optimizer.
    Uses reward function balancing collision avoidance vs fuel efficiency.
    """
    
    def __init__(
        self,
        max_delta_v: float = 0.5,  # m/s
        spacecraft_mass: float = 6500.0,  # kg (typical LEO satellite)
        isp: float = 300.0  # specific impulse in seconds
    ):
        """
        Initialize RL maneuver optimizer.
        
        Args:
            max_delta_v: Maximum delta-v per burn (m/s)
            spacecraft_mass: Spacecraft mass (kg)
            isp: Specific impulse of thruster (seconds)
        """
        self.max_delta_v = max_delta_v
        self.spacecraft_mass = spacecraft_mass
        self.isp = isp  # seconds
        self.g0 = 9.81  # m/s²
    
    def compute_fuel_cost(self, delta_v_magnitude: float) -> float:
        """
        Compute fuel cost for a maneuver using Tsiolkovsky rocket equation.
        
        Args:
            delta_v_magnitude: Magnitude of delta-v (m/s)
        
        Returns:
            Fuel mass required (kg)
        """
        # Tsiolkovsky: Δm = m * (exp(Δv / (Isp * g0)) - 1)
        exponent = delta_v_magnitude / (self.isp * self.g0)
        if exponent > 100:  # Avoid overflow
            return self.spacecraft_mass
        fuel_mass = self.spacecraft_mass * (np.exp(exponent) - 1)
        return fuel_mass
    
    def reward_function(
        self,
        dca_initial: float,
        dca_predicted: float,
        fuel_cost: float,
        pc_initial: float,
        pc_predicted: float,
        alpha_safety: float = 0.7,
        alpha_fuel: float = 0.3
    ) -> float:
        """
        Reward function balancing safety and fuel efficiency.
        
        Args:
            dca_initial: Initial DCA (km)
            dca_predicted: Predicted DCA after maneuver (km)
            fuel_cost: Fuel cost (kg)
            pc_initial: Initial probability of collision
            pc_predicted: Predicted probability of collision
            alpha_safety: Weight for safety (0 to 1)
            alpha_fuel: Weight for fuel efficiency (0 to 1)
        
        Returns:
            Reward score (higher is better)
        """
        # Safety reward: improvement in DCA and collision probability
        dca_improvement = max(0, dca_predicted - dca_initial) / (dca_initial + 1e-3)
        pc_improvement = max(0, pc_initial - pc_predicted)
        
        safety_reward = dca_improvement + pc_improvement * 100  # Weight PC more heavily
        
        # Fuel efficiency reward: penalize excessive fuel consumption
        max_fuel = self.compute_fuel_cost(self.max_delta_v)
        fuel_reward = 1.0 - (fuel_cost / max_fuel)  # Normalized to [0, 1]
        
        # Combined reward
        total_reward = alpha_safety * safety_reward + alpha_fuel * fuel_reward
        
        return total_reward
    
    def generate_candidate_maneuvers(
        self,
        n_candidates: int = 100,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate candidate maneuvers by sampling from action space.
        
        Args:
            n_candidates: Number of candidate maneuvers
            seed: Random seed
        
        Returns:
            Array of candidate delta-v vectors (n_candidates, 3) in m/s
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Sample from sphere of radius max_delta_v
        candidates = np.random.randn(n_candidates, 3)
        magnitudes = np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Random radius up to max_delta_v
        radii = np.random.uniform(0, self.max_delta_v, (n_candidates, 1))
        candidates = candidates / (magnitudes + 1e-10) * radii
        
        return candidates
    
    def propagate_with_maneuver(
        self,
        pos_sat1_initial: np.ndarray,
        vel_sat1_initial: np.ndarray,
        pos_sat2_initial: np.ndarray,
        vel_sat2_initial: np.ndarray,
        delta_v: np.ndarray,
        propagation_time: float = 3600.0  # seconds
    ) -> Tuple[float, float]:
        """
        Simple linear propagation with maneuver applied at t=0.
        (In production, use numerical integration with perturbations)
        
        Args:
            pos_sat1_initial: Initial position of satellite 1 (km)
            vel_sat1_initial: Initial velocity of satellite 1 (km/s)
            pos_sat2_initial: Initial position of satellite 2 (km)
            vel_sat2_initial: Initial velocity of satellite 2 (km/s)
            delta_v: Delta-v maneuver (km/s, converted from m/s input)
            propagation_time: Propagation time (seconds)
        
        Returns:
            (dca_predicted, pc_predicted)
        """
        # Apply maneuver to satellite 1
        vel_sat1_modified = vel_sat1_initial + delta_v
        
        # Linear propagation to TCA (~collision time)
        pos_sat1_at_tca = pos_sat1_initial + vel_sat1_modified * propagation_time
        pos_sat2_at_tca = pos_sat2_initial + vel_sat2_initial * propagation_time
        
        # Compute new DCA
        dca_predicted = np.linalg.norm(pos_sat1_at_tca - pos_sat2_at_tca)
        
        # Simple PC estimate: exponential decay with distance
        pc_predicted = np.exp(-dca_predicted / 2.0) * 0.1
        
        return dca_predicted, pc_predicted
    
    def optimize_maneuver(
        self,
        pos_sat1_initial: np.ndarray,
        vel_sat1_initial: np.ndarray,
        pos_sat2_initial: np.ndarray,
        vel_sat2_initial: np.ndarray,
        dca_initial: float,
        pc_initial: float,
        burn_time_utc: Optional[str] = None,
        n_candidates: int = 200,
        top_k: int = 5
    ) -> Dict:
        """
        Optimize maneuver selection using RL approach.
        
        Args:
            pos_sat1_initial: Initial position of satellite 1 (km)
            vel_sat1_initial: Initial velocity of satellite 1 (km/s)
            pos_sat2_initial: Initial position of satellite 2 (km)
            vel_sat2_initial: Initial velocity of satellite 2 (km/s)
            dca_initial: Initial DCA (km)
            pc_initial: Initial probability of collision
            burn_time_utc: Proposed burn time (UTC)
            n_candidates: Number of candidate maneuvers to evaluate
            top_k: Return top-k best maneuvers
        
        Returns:
            Dictionary with optimization results and recommended maneuvers
        """
        if burn_time_utc is None:
            burn_time_utc = datetime.utcnow().isoformat() + "Z"
        
        # Generate candidates
        candidates = self.generate_candidate_maneuvers(n_candidates)
        
        rewards = []
        predictions = []
        
        for delta_v_m_s in candidates:
            delta_v_km_s = delta_v_m_s / 1000.0  # Convert to km/s
            fuel_cost = self.compute_fuel_cost(np.linalg.norm(delta_v_m_s))
            
            dca_pred, pc_pred = self.propagate_with_maneuver(
                pos_sat1_initial, vel_sat1_initial,
                pos_sat2_initial, vel_sat2_initial,
                delta_v_km_s
            )
            
            reward = self.reward_function(
                dca_initial, dca_pred,
                fuel_cost, pc_initial, pc_pred
            )
            
            rewards.append(reward)
            predictions.append({
                'delta_v': delta_v_m_s,
                'fuel_cost': fuel_cost,
                'dca_predicted': dca_pred,
                'pc_predicted': pc_pred,
                'reward': reward
            })
        
        # Sort by reward
        sorted_indices = np.argsort(rewards)[::-1]
        
        best_maneuvers = []
        for idx in sorted_indices[:top_k]:
            pred = predictions[idx]
            dv = pred['delta_v']
            
            maneuver = ManeuverAction(
                delta_v_x=float(dv[0]),
                delta_v_y=float(dv[1]),
                delta_v_z=float(dv[2]),
                burn_time_utc=burn_time_utc,
                fuel_cost_kg=float(pred['fuel_cost']),
                predicted_dca_improvement=float(pred['dca_predicted'] - dca_initial),
                risk_reduction=float((pc_initial - pred['pc_predicted']) / (pc_initial + 1e-10))
            )
            best_maneuvers.append(maneuver)
        
        best = best_maneuvers[0] if best_maneuvers else None
        
        return {
            'recommended_maneuver': best,
            'top_k_maneuvers': best_maneuvers,
            'initial_dca_km': dca_initial,
            'initial_pc': pc_initial,
            'best_predicted_dca_km': predictions[sorted_indices[0]]['dca_predicted'],
            'best_predicted_pc': predictions[sorted_indices[0]]['pc_predicted'],
            'best_fuel_cost_kg': predictions[sorted_indices[0]]['fuel_cost'],
            'best_reward_score': float(rewards[sorted_indices[0]])
        }
