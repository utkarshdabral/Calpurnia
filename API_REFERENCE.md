# Calpurnia API Reference & Quick Start

## Quick Start: 30 Seconds

```python
from src.physics.dss import ConjunctionAssessmentDecisionSupport

# Initialize
dss = ConjunctionAssessmentDecisionSupport(keep_out_sphere_km=2.0)

# Assess conjunction
result = dss.assess_conjunction_pair(25544, 25543)  # ISS vs debris

# Get report
print(dss.generate_explainability_report(result))

# Export
dss.export_to_json(result, "assessment.json")
dss.export_to_html(result, "dashboard.html")
```

---

## Core API

### 1. Orbital Propagation

#### `load_tle(catalog_id: int) -> Tuple[str, str, str]`

Load satellite TLE data.

```python
from src.physics.converter import load_tle

name, tle1, tle2 = load_tle(25544)
print(name)  # "ISS (ZARYA)"
```

**Arguments:**
- `catalog_id`: NORAD catalog ID (e.g., 25544 for ISS)

**Returns:**
- `(name, tle_line1, tle_line2)` tuple

---

#### `propagate_orbit(tle_line1, tle_line2, times) -> Tuple[np.ndarray, np.ndarray]`

Propagate satellite orbit using SGP4.

```python
from src.physics.converter import propagate_orbit
from skyfield.api import Loader

ts = Loader('data').timescale()
times = ts.utc(2026, 1, 26, 12, 0, range(0, 3600, 10))

positions, velocities = propagate_orbit(tle1, tle2, times)
# positions: (N, 3) in km
# velocities: (N, 3) in km/s
```

**Arguments:**
- `tle_line1`, `tle_line2`: TLE strings
- `times`: Skyfield time array

**Returns:**
- `(positions, velocities)` as numpy arrays

---

### 2. Conjunction Assessment

#### `ConjunctionAssessment(keep_out_sphere_radius=1.0)`

Initialize conjunction assessor.

```python
from src.physics.conjunction import ConjunctionAssessment, create_default_covariance

ca = ConjunctionAssessment(keep_out_sphere_radius=2.0)
```

---

#### `assess_conjunction(name1, pos1, vel1, cov1, name2, pos2, vel2, cov2, time_seconds) -> Dict`

Complete conjunction assessment.

```python
result = ca.assess_conjunction(
    "ISS", pos1, vel1, cov1,
    "DEBRIS", pos2, vel2, cov2,
    time_array
)

print(result['dca_km'])  # 0.543 km
print(result['probability_of_collision'])  # 2.3e-5
print(result['alert'])  # True/False
```

**Returns Dictionary:**
```python
{
    'satellite_1': str,
    'satellite_2': str,
    'dca_km': float,  # Distance of closest approach
    'tca_seconds': float,  # Time of closest approach
    'tca_index': int,  # Index in time array
    'probability_of_collision': float,  # 0 to 1
    'alert': bool,  # pc >= threshold?
    'inside_keep_out': bool,  # dca < keep_out_radius?
}
```

---

### 3. Reinforcement Learning Maneuver Optimization

#### `SimpleReinforcementLearner(max_delta_v=0.5, spacecraft_mass=6500, isp=300)`

Initialize RL optimizer.

```python
from src.physics.maneuver import SimpleReinforcementLearner

rl = SimpleReinforcementLearner(
    max_delta_v=0.5,  # m/s max burn
    spacecraft_mass=6500,  # kg
    isp=300  # specific impulse, seconds
)
```

---

#### `optimize_maneuver(pos_sat1, vel_sat1, pos_sat2, vel_sat2, dca_initial, pc_initial, n_candidates=200) -> Dict`

Find optimal collision avoidance maneuver.

```python
result = rl.optimize_maneuver(
    pos_sat1, vel_sat1,
    pos_sat2, vel_sat2,
    dca_initial=2.5,  # km
    pc_initial=5e-4,  # probability
    n_candidates=200,
    top_k=3
)

rec = result['recommended_maneuver']
print(f"Δv: {rec.delta_v_x:.3f}, {rec.delta_v_y:.3f}, {rec.delta_v_z:.3f} m/s")
print(f"Fuel: {rec.fuel_cost_kg:.2f} kg")
print(f"Risk reduction: {rec.risk_reduction:.2%}")
```

**Returns Dictionary:**
```python
{
    'recommended_maneuver': ManeuverAction,  # Best option
    'top_k_maneuvers': List[ManeuverAction],  # Top-k options
    'initial_dca_km': float,
    'best_predicted_dca_km': float,
    'initial_pc': float,
    'best_predicted_pc': float,
    'best_fuel_cost_kg': float,
    'best_reward_score': float
}
```

---

### 4. AI Residual Prediction

#### `ResidualErrorPredictor(sequence_length=24, prediction_horizon=6)`

Initialize LSTM residual predictor.

```python
from src.ai.residual_predictor import ResidualErrorPredictor

pred = ResidualErrorPredictor(
    sequence_length=24,  # hours of history
    prediction_horizon=6  # hours to predict
)
```

---

#### `train_lstm_mock(sgp4_positions, actual_positions, epochs=50) -> Dict`

Train on historical data.

```python
history = pred.train_lstm_mock(
    sgp4_positions,  # (N, 3) in km
    actual_positions,  # (N, 3) in km
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

print(history['final_train_loss'])
print(history['final_val_loss'])
```

---

#### `correct_sgp4_prediction(sgp4_position, recent_residuals) -> np.ndarray`

Apply AI correction to SGP4 position.

```python
corrected_pos = pred.correct_sgp4_prediction(
    sgp4_prediction,  # (3,) km
    recent_residuals  # (sequence_length, 3) km
)
```

---

### 5. Decision Support System

#### `ConjunctionAssessmentDecisionSupport(keep_out_sphere_km=1.0, pc_alert_threshold=1e-4, use_ai_correction=True)`

Main integration layer.

```python
from src.physics.dss import ConjunctionAssessmentDecisionSupport

dss = ConjunctionAssessmentDecisionSupport(
    keep_out_sphere_km=2.0,
    pc_alert_threshold=1e-5,
    use_ai_correction=True
)
```

---

#### `assess_conjunction_pair(satellite_id_1, satellite_id_2, propagation_hours=24) -> Dict`

End-to-end assessment.

```python
result = dss.assess_conjunction_pair(
    satellite_id_1=25544,  # ISS
    satellite_id_2=25543,  # Debris
    propagation_hours=24,
    propagation_steps=100,
    enable_maneuver_planning=True
)

# Access results
ca = result['conjunction_assessment']
mp = result.get('maneuver_plan', {})
```

---

#### `generate_explainability_report(assessment: Dict) -> str`

Human-readable report.

```python
report = dss.generate_explainability_report(result)
print(report)
```

**Output Example:**
```
CONJUNCTION ASSESSMENT REPORT
================================================================================
Generated: 2026-01-26T15:22:00Z

OBJECT PAIR:
  Primary: ISS (ZARYA) (Catalog ID: 25544)
  Secondary: DEBRIS (Catalog ID: 25543)

COLLISION RISK SUMMARY:
  Distance of Closest Approach (DCA): 0.543 km
  Probability of Collision (Pc): 2.30e-05
  Inside Keep-Out Sphere (2.0 km): False
  ALERT STATUS: ✓ Safe

RECOMMENDED MANEUVER:
  Delta-V Vector: [+0.2450, -0.1560, +0.0890] m/s
  Δv Magnitude: 0.305 m/s
  Burn Time: 2026-01-26T15:22:30Z
  Fuel Required: 0.68 kg
  
PREDICTED OUTCOME:
  Current DCA: 0.543 km → After maneuver: 3.241 km
  Current Pc: 2.30e-05 → After maneuver: 1.20e-06
  Risk Reduction Factor: 19.2x safer
================================================================================
```

---

#### `export_to_json(assessment, filepath)`

Export to JSON.

```python
dss.export_to_json(result, "assessment.json")
```

---

#### `export_to_html(assessment, filepath)`

Export to HTML dashboard.

```python
dss.export_to_html(result, "dashboard.html")
```

---

## Data Structures

### ManeuverAction

```python
@dataclass
class ManeuverAction:
    delta_v_x: float  # m/s
    delta_v_y: float  # m/s
    delta_v_z: float  # m/s
    burn_time_utc: str
    fuel_cost_kg: float
    predicted_dca_improvement: float  # km
    risk_reduction: float  # 0 to 1
```

---

## Configuration

### Keep-Out Sphere

Controls collision risk threshold.

```python
# Conservative (more maneuvers)
dss = ConjunctionAssessmentDecisionSupport(keep_out_sphere_km=5.0)

# Aggressive (fewer maneuvers)
dss = ConjunctionAssessmentDecisionSupport(keep_out_sphere_km=0.5)
```

---

### Alert Threshold

Controls probability of collision alert level.

```python
# Strict (alert at higher probabilities)
pc_threshold = 1e-4  # Alert if Pc > 0.01%

# Relaxed (alert only at very high risk)
pc_threshold = 1e-5  # Alert if Pc > 0.001%
```

---

### AI Correction

Enable/disable LSTM residual prediction.

```python
# With AI correction (higher accuracy, requires training)
dss = ConjunctionAssessmentDecisionSupport(use_ai_correction=True)

# Without AI (fast SGP4-only mode)
dss = ConjunctionAssessmentDecisionSupport(use_ai_correction=False)
```

---

## Common Use Cases

### Use Case 1: Quick Risk Check

```python
from src.physics.dss import ConjunctionAssessmentDecisionSupport

dss = ConjunctionAssessmentDecisionSupport()
result = dss.assess_conjunction_pair(25544, 25543)

if result['conjunction_assessment']['alert']:
    print("🚨 COLLISION ALERT!")
else:
    print("✓ Safe")
```

### Use Case 2: Get Maneuver Recommendation

```python
mp = result['maneuver_plan']
rec = mp['recommended_maneuver']

print(f"Execute {rec['delta_v_m_s']['magnitude']:.3f} m/s burn at {rec['burn_time_utc']}")
```

### Use Case 3: Generate Report

```python
report = dss.generate_explainability_report(result)
print(report)
dss.export_to_html(result, "collision_report.html")
```

### Use Case 4: Batch Assessment

```python
debris_catalog = [25543, 25544, 25545, 25546]
iss_id = 25544

for debris_id in debris_catalog:
    if debris_id == iss_id:
        continue
    
    result = dss.assess_conjunction_pair(iss_id, debris_id)
    
    if result['conjunction_assessment']['alert']:
        print(f"Alert: {debris_id}")
        dss.export_to_json(result, f"alert_{debris_id}.json")
```

---

## Error Handling

### TLE Not Found

```python
try:
    result = dss.assess_conjunction_pair(99999, 25544)
except Exception as e:
    print(f"Error: {e}")
```

### Invalid Covariance

```python
# Ensure covariance is positive definite
cov = create_default_covariance(sigma_position_m=150)
assert np.all(np.linalg.eigvals(cov) > 0)
```

---

## Performance Tips

1. **Batch Processing**: Reuse DSS instance for multiple assessments
2. **Reduce Steps**: Use `propagation_steps=50` for speed (default: 100)
3. **Disable AI**: Use `use_ai_correction=False` for real-time (faster by 50%)
4. **Cache TLEs**: Load TLEs once, reuse for multiple conjunction pairs

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| TLE too old | Refresh from Celestrak |
| High DCA but Alert | Check keep-out sphere setting |
| Residual model poor | Retrain with more data |
| Maneuver unrealistic | Increase spacecraft mass or ISP |

---

## Examples Directory

Run examples:
```bash
python main_demo.py           # 4-part full demo
python tests/test_all.py      # Unit tests + examples
```

---

## Support & Further Reading

- **Full Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Submission Summary**: See [SUBMISSION.md](SUBMISSION.md)
- **Original README**: See [README.md](README.md)
- **Requirements**: See [requirements.txt](requirements.txt)

---

*Last Updated: 2026-01-26*
