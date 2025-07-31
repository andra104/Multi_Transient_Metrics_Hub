# M Dwarf Flare‑Only Light Curve Model

## Overview
This model simulates **short‑duration flares** on M dwarfs **without including a quiescent baseline in the light curve output**.  
It is designed for fast‑evolving events lasting from **30 minutes to 4 hours**, where the brightness quickly rises and then drops instantly back to the quiescent level.  

The model supports:
- **Variable flare durations** per simulated event.
- **Randomized quiescent brightness** based on empirical distributions.
- **Randomized flare amplitude** between **0.7 and 1.0 magnitudes** brighter than the star’s quiescent level.
- **Instant drop‑off** after peak brightness — no extended fade tail.

---

## Light Curve Structure

Each flare consists of:
1. **Rise Phase**  
   Duration: **15–30 %** of total flare duration.  
   Steep, nearly linear brightening toward peak.  

2. **Peak**  
   Single point at **t = 0**.  

3. **Instant Drop**  
   Immediately after the peak, the light curve returns to quiescent level in one step.  

---

## Parameter Ranges

| Parameter | Description | Range / Value |
|-----------|-------------|---------------|
| **Total Duration** | Time from flare start to return to quiescent | `0.02–0.17` days (≈ 30 min – 4 hr) |
| **Rise Fraction** | Fraction of total duration spent in rise | `0.15–0.3` |
| **Quiescent Magnitude (u)** | Random draw per flare | `17.5–20.5` |
| **Quiescent Magnitude (g)** | Random draw per flare | `16.5–19.5` |
| **Quiescent Magnitude (r)** | Random draw per flare | `15.0–18.0` |
| **Quiescent Magnitude (i)** | Random draw per flare | `13.0–15.5` |
| **Quiescent Magnitude (z)** | Random draw per flare | `12.0–13.5` |
| **Quiescent Magnitude (y)** | Random draw per flare | `11.5–12.7` |
| **Flare Amplitude** | Brightness increase above quiescent | `0.7–1.0` mag |
| **Rise Rate** | Steepening of rise to peak | `5–15` mag/day |
| **Fade** | Instant return to quiescent | N/A |

---

## Output Data

Each simulated light curve stores:
- **`ph`** – Phase/time array for the flare (days relative to peak)  
- **`mag`** – Magnitude array corresponding to the flare evolution  
- **`rise_time_days`** – Duration of rise phase (days)  
- **`fade_time_days`** – Duration of fade phase (days) — near zero for instant drop events  

---

## Example Shape



## 2. Extinction and Distance

- **Galactic Extinction**: Optional; applied using the SFD dust map (`dustmaps.sfd.SFDQuery`) with extinction coefficients from `rubin_sim.phot_utils.DustValues`.
- **Distance Modulus**: Applied for each injected event based on its simulated distance.
- **Population distances**: Drawn from a volumetric distribution with configurable min/max bounds.

---

## 3. Detection Logic

Detection is currently not entirely determined as option A and B are too stringent. But, functionally, detection would be defined as meeting **any** of the options we would impliment:

| **Option** | **Condition** |
|------------|---------------|
| **A** | ≥3 detections ≥3σ **and** ≥1 ≥5σ, all within **0.5 days** |
| **B** | ≥2 detections ≥5σ, separated by ≥15 minutes |
| **C** | ≥3 detections ≥3σ within **3 days**, ≥1 ≥5σ |
| **D** | ≥2 detections ≥5σ in any two epochs |

---

## 4. Characterization Logic (yet to be fully implemented) 

A flare is **characterized** if:
- ≥4 detections above 0.5σ (minimum sampling)
....TBD?

---

## 5. Metrics Implemented

| Metric Name | Purpose | Key Criteria |
|-------------|---------|--------------|
| `Detect_Metric` | Discovery efficiency | Any of Options A–D above |
| `MDwarfFlareCharacterizeMetric` | Morphological classification | Peak counting logic above |

Both metrics record detailed **observation records** (`obs_records`) per event for later inspection.

---

## 6. How to Run

Example workflow:

```python
from local_MDwarfFlares_metric import get_multi_metrics
import shared_utils

# Load or generate light curve templates
lc_model = shared_utils.load_or_generate_templates(
    LC, "output/MDwarfFlares_templates.pkl", generate_new=True
)

# Load or generate population
slicer = shared_utils.load_or_generate_population(
    use_extinction=True,
    t_start=1, t_end=3652,
    z_min=0.00226, z_max=0.3,
    rate_density=1e-9,
    pop_file="output/MDwarfFlares_population.pkl",
    generate_new=True
)

# Run detection and characterization metrics
multi_metrics = get_multi_metrics(lc_model, include=['detect', 'characterize'])
shared_utils.run_multi_metrics(
    multi_metrics, slicer,
    cadences=['baseline_v4.3.1_10yrs'],
    db_dir="path/to/opsim_dbs",
    storage_dir="output/MDwarfFlares",
    summary_filename="output/MDwarfFlares_summary.csv"
)
