# M Dwarf Flare Metric for LSST Cadence Evaluation

## Overview
This metric simulates and evaluates the detectability and classification of **M Dwarf Flares** in Rubin Observatory LSST survey cadences.  
It models synthetic flare light curves based on empirical quiescent magnitude distributions, injects them across the sky, and measures how often LSST would detect and classify these short-lived events.

The M Dwarf flare metric supports:
- **Detection efficiency**: Fraction of flares meeting minimum discovery criteria.
- **Characterization**: Events with enough time sampling to distinguish between classical and complex (multi-peaked) flare morphologies. TBD

The design follows observed M Dwarf flare properties from:
- **UltraCoolSheet**: Empirical quiescent magnitude distributions.
- **Stellar flare surveys**: Typical amplitudes, durations, and decay behaviors.

---

## 1. Light Curve Model

M Dwarf flares are modeled with:
- **Constant pre-flare phase** (~7 days before peak)
- **Fast rise** (<1 hour to peak)
- **Sharp peak** (t = 0)
- **Fading tail** (~1.5 days to quiescence) -> Needs to be shorter.

**Current Issue:** Needs to be only a .7-1 mag difference between quiescence and peak. Very fast return to quiescence needs to be less than hour. Need a better law to govern the light curve. 

Quiescent magnitudes are drawn from filter-specific ranges (empirical UltraCoolSheet values):

| Filter | Quiescent Mag Range |
|--------|--------------------|
| u | 17.5 – 20.5 |
| g | 16.5 – 19.5 |
| r | 15.0 – 18.0 |
| i | 13.0 – 15.5 |
| z | 12.0 – 13.5 |
| y | 11.5 – 12.7 |

**Flare amplitude**: Brightens by `Δmag` (default 5.0 mag) from quiescence.  
**Rise/fade rates**: Drawn from filter-specific empirical/fitted ranges.

---

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
- Then tested for **complexity**:
  - **Complex flare**: ≥2 peaks above 1.5σ separated by ≥0.1 days
  - **Classical flare**: Otherwise

Returns (yet to be implemented):
- `1.0` -> Complex flare
- `0.5` -> Classical flare
- `0.0` -> Not characterizable

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
