from skyfield.api import Loader, EarthSatellite, wgs84
from skyfield.data import hipparcos
import os
from datetime import datetime, timedelta
import numpy as np

DATA_DIR = "data"
SATELLITE_ID = 25544

def load_tle(catalog_id: int) -> tuple:
    """Load TLE data and return satellite name and lines."""
    file_path = os.path.join(DATA_DIR, f"tle_{catalog_id}.txt")

    if not os.path.exists(file_path):
        from src.utils.data_fetcher import fetch_tle
        fetch_tle(catalog_id)

    # Read TLE lines directly from file
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Parse TLE format: name, line1, line2
    if len(lines) < 2:
        raise ValueError(f"Invalid TLE format in {file_path}")
    
    # Handle both 2-line and 3-line TLE format
    if lines[0].startswith('1 '):
        name = "SATELLITE"
        tle_line1, tle_line2 = lines[0], lines[1]
    else:
        name = lines[0]
        tle_line1, tle_line2 = lines[1], lines[2]
    
    return name, tle_line1, tle_line2

def propagate_orbit(tle_line1: str, tle_line2: str, times):
    """Propagate satellite position using SGP4 via Skyfield."""
    from skyfield.api import EarthSatellite
    
    satellite = EarthSatellite(tle_line1, tle_line2)
    positions = []
    velocities = []
    
    for t in times:
        astrometric = satellite.at(t)
        position = astrometric.position.km
        velocity = astrometric.velocity.km_per_s
        positions.append(position)
        velocities.append(velocity)
    
    return np.array(positions), np.array(velocities)

def calculate_residual_error(predicted_positions, actual_positions):
    """Calculate residual error between predicted and actual positions."""
    return actual_positions - predicted_positions

if __name__ == "__main__":
    # Example usage
    name, tle_line1, tle_line2 = load_tle(SATELLITE_ID)
    ts = Loader('data').timescale()

    # Current time and next hour
    now = datetime.utcnow()
    times = ts.utc(now.year, now.month, now.day, now.hour, range(0, 60, 10))  # Every 10 minutes

    positions, velocities = propagate_orbit(tle_line1, tle_line2, times)
    print(f"Satellite: {name}")
    print(f"Propagated positions shape: {positions.shape}")
    print(f"Position at first epoch: {positions[0]} km")
    print(f"Velocity at first epoch: {velocities[0]} km/s")
