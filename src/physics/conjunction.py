"""
Conjunction Assessment Risk Analysis (CARA) - Core orbital mechanics
Computes Distance of Closest Approach (DCA) and probability of collision.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
from scipy.optimize import minimize_scalar


class ConjunctionAssessment:
    """Performs conjunction assessment between two satellites."""
    
    def __init__(self, keep_out_sphere_radius: float = 1.0):
        """
        Initialize conjunction assessment.
        
        Args:
            keep_out_sphere_radius: Radius of keep-out sphere in km (default: 1km)
        """
        self.keep_out_sphere_radius = keep_out_sphere_radius
    
    def compute_distance_of_closest_approach(
        self,
        pos1: np.ndarray,
        vel1: np.ndarray,
        pos2: np.ndarray,
        vel2: np.ndarray,
        time_seconds: np.ndarray
    ) -> Tuple[float, float, int]:
        """
        Compute Distance of Closest Approach (DCA) using linear relative motion model.
        
        Args:
            pos1: Satellite 1 position (km) - shape (3,) or (N, 3)
            vel1: Satellite 1 velocity (km/s) - shape (3,) or (N, 3)
            pos2: Satellite 2 position (km) - shape (3,) or (N, 3)
            vel2: Satellite 2 velocity (km/s) - shape (3,) or (N, 3)
            time_seconds: Time array (seconds) - shape (N,)
        
        Returns:
            dca: Distance of closest approach (km)
            tca: Time of closest approach (seconds from first epoch)
            tca_index: Index in time array nearest to TCA
        """
        # Handle both single and multiple time steps
        if pos1.ndim == 1:
            pos1 = pos1.reshape(1, -1)
            vel1 = vel1.reshape(1, -1)
            pos2 = pos2.reshape(1, -1)
            vel2 = vel2.reshape(1, -1)
        
        # Relative position and velocity at first epoch
        rel_pos_0 = pos1[0] - pos2[0]
        rel_vel = vel1[0] - vel2[0]
        
        # Time of Closest Approach (TCA) - solve for when relative velocity is perpendicular to relative position
        # r_rel(t) = r0 + v*t
        # TCA: d(|r_rel|^2)/dt = 0 => 2*r0·v + 2*v·v*t = 0
        denom = np.dot(rel_vel, rel_vel)
        
        if denom < 1e-10:  # Parallel or zero relative velocity
            tca = time_seconds[0]
            dca = np.linalg.norm(rel_pos_0)
        else:
            tca_normalized = -np.dot(rel_pos_0, rel_vel) / denom
            tca = time_seconds[0] + tca_normalized
            
            # Clamp TCA to measurement window
            tca = max(time_seconds[0], min(tca, time_seconds[-1]))
            
            # Compute relative position at TCA
            t_offset = tca - time_seconds[0]
            rel_pos_tca = rel_pos_0 + rel_vel * t_offset
            dca = np.linalg.norm(rel_pos_tca)
        
        # Find closest index in time array
        tca_index = np.argmin(np.abs(time_seconds - tca))
        
        return dca, tca, tca_index
    
    def compute_probability_of_collision(
        self,
        dca: float,
        covariance1: np.ndarray,
        covariance2: np.ndarray,
        relative_velocity: np.ndarray
    ) -> float:
        """
        Compute probability of collision using Covariance-based collision assessment.
        
        Args:
            dca: Distance of closest approach (km)
            covariance1: 3x3 position covariance of satellite 1 (km²)
            covariance2: 3x3 position covariance of satellite 2 (km²)
            relative_velocity: 3D relative velocity vector (km/s)
        
        Returns:
            Pc: Probability of collision (0 to 1)
        """
        from scipy.stats import norm
        
        # Combined covariance
        combined_cov = covariance1 + covariance2
        
        # Effective combined standard deviation
        # (Assumes spherical covariance; for accurate results, use 3D Mahalanobis distance)
        try:
            sigma = np.sqrt(np.trace(combined_cov) / 3)  # Mean of eigenvalues
        except:
            sigma = 1.0  # Fallback to 1 km std dev
        
        # Collision probability (Gaussian approximation)
        # Pc ≈ Φ((R + σ*k - dca) / (σ*sqrt(k²+1)))
        # where k ≈ 3-4, R is collision radius
        
        if sigma < 1e-6:
            pc = 1.0 if dca < self.keep_out_sphere_radius else 0.0
        else:
            # Foster's formula approximation
            k = 3.0  # Number of standard deviations for high confidence
            x = (self.keep_out_sphere_radius - dca) / sigma
            pc = norm.cdf(x)
            pc = max(0.0, min(1.0, pc))  # Clamp to [0, 1]
        
        return pc
    
    def assess_conjunction(
        self,
        name1: str,
        pos1: np.ndarray,
        vel1: np.ndarray,
        cov1: np.ndarray,
        name2: str,
        pos2: np.ndarray,
        vel2: np.ndarray,
        cov2: np.ndarray,
        time_seconds: np.ndarray,
        pc_threshold: float = 1e-4,
        monte_carlo_samples: int = 10000
    ) -> Dict:
        """
        Complete conjunction assessment for two satellites with Monte Carlo simulation.
        
        Args:
            name1: Name of satellite 1
            pos1: Position of satellite 1 (km)
            vel1: Velocity of satellite 1 (km/s)
            cov1: Position covariance of satellite 1 (km²)
            name2: Name of satellite 2
            pos2: Position of satellite 2 (km)
            vel2: Velocity of satellite 2 (km/s)
            cov2: Position covariance of satellite 2 (km²)
            time_seconds: Time array (seconds)
            pc_threshold: Threshold for collision probability alert
            monte_carlo_samples: Number of Monte Carlo samples for uncertainty quantification
        
        Returns:
            Dictionary with conjunction assessment results
        """
        dca, tca, tca_index = self.compute_distance_of_closest_approach(
            pos1, vel1, pos2, vel2, time_seconds
        )
        
        rel_vel = vel1[0] if vel1.ndim > 1 else vel1 - vel2
        pc = self.compute_probability_of_collision(dca, cov1, cov2, rel_vel)
        
        # Perform Monte Carlo simulation for uncertainty quantification
        pc_monte_carlo, collision_samples = self.monte_carlo_collision_probability(
            pos1, vel1, cov1,
            pos2, vel2, cov2,
            time_seconds,
            n_samples=monte_carlo_samples
        )
        
        alert = pc_monte_carlo >= pc_threshold
        
        return {
            "satellite_1": name1,
            "satellite_2": name2,
            "dca_km": float(dca),
            "tca_seconds": float(tca),
            "tca_index": int(tca_index),
            "probability_of_collision": float(pc),
            "probability_of_collision_monte_carlo": float(pc_monte_carlo),
            "monte_carlo_collision_count": int(collision_samples),
            "monte_carlo_total_samples": monte_carlo_samples,
            "alert": bool(alert),
            "keep_out_sphere_km": self.keep_out_sphere_radius,
            "inside_keep_out": dca < self.keep_out_sphere_radius,
        }
    
    def monte_carlo_collision_probability(
        self,
        pos1: np.ndarray,
        vel1: np.ndarray,
        cov1: np.ndarray,
        pos2: np.ndarray,
        vel2: np.ndarray,
        cov2: np.ndarray,
        time_seconds: np.ndarray,
        n_samples: int = 10000
    ) -> Tuple[float, int]:
        """
        Compute collision probability using Monte Carlo simulation.
        
        Args:
            pos1: Position of satellite 1 (km)
            vel1: Velocity of satellite 1 (km/s)
            cov1: Position covariance of satellite 1 (km²)
            pos2: Position of satellite 2 (km)
            vel2: Velocity of satellite 2 (km/s)
            cov2: Position covariance of satellite 2 (km²)
            time_seconds: Time array (seconds)
            n_samples: Number of Monte Carlo samples
        
        Returns:
            (probability_of_collision, number_of_collisions)
        """
        collision_count = 0
        
        # Generate Monte Carlo samples
        for _ in range(n_samples):
            # Sample positions from covariance distributions
            pos1_sample = np.random.multivariate_normal(pos1[0], cov1)
            pos2_sample = np.random.multivariate_normal(pos2[0], cov2)
            
            # Compute DCA for this sample
            dca_sample, _, _ = self.compute_distance_of_closest_approach(
                pos1_sample, vel1[0], pos2_sample, vel2[0], time_seconds
            )
            
            # Check if collision occurs (within keep-out sphere)
            if dca_sample < self.keep_out_sphere_radius:
                collision_count += 1
        
        pc_monte_carlo = collision_count / n_samples
        return pc_monte_carlo, collision_count


def create_default_covariance(sigma_position_m: float = 100.0) -> np.ndarray:
    """
    Create default spherical position covariance matrix.
    
    Args:
        sigma_position_m: Standard deviation in meters
    
    Returns:
        3x3 covariance matrix in km²
    """
    sigma_km = sigma_position_m / 1000.0
    return np.eye(3) * (sigma_km ** 2)
