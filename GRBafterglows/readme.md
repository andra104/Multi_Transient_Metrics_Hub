# GRB Afterglow Metric for LSST Cadence Evaluation

## Overview
This metric simulates and evaluates the detectability and characterization of **Gamma-Ray Burst (GRB) afterglows** in Rubin Observatory LSST survey cadences.  
It models synthetic GRB light curves based on empirical distributions, injects them across the sky, and measures how often LSST would detect, characterize, and trigger spectroscopic follow-up for these events.

The GRB afterglow metric supports:
- **Detection efficiency**: Fraction of events meeting minimum discovery criteria.
- **Characterization**: Events with enough multi-band temporal coverage to constrain light curve evolution.
- **Spectroscopic triggerability**: Events bright enough and early enough for rapid spectroscopic follow-up.

The design follows empirical GRB afterglow properties from:
- **Zeh et al. (2005)**: Temporal decay slopes and jet break behavior.
- **Cenko et al. (2009)**: Peak brightness distributions.
- **Observed population**: Typical timescales, brightness, and spectral slopes.

---

## 1. Light Curve Model

GRB afterglows are modeled as **power-law decays with a jet break**:

`m(t) = m_0 + 2.5 * α * log10(t / t_0)`

- **Pre-break**: Shallow decay slope (`α = 1.5`).
- **Post-break**: Steeper decline (multiplied decay rate after the jet break time).
- **Jet break**: Randomized between `1–5` days.
- **Peak brightness**: Drawn from a range `(-31.6, -18.47)` mag (absolute, Rc band).

The **reference light curve** is generated in **Rc-band** from Cenko et al. 2009 survey catalog, then transformed into LSST `u, g, r, i, z, y` bands using a **spectral index**:

`F_ν ∝ ν^β,   β = -0.75`

Conversion is handled by the `apply_spectral_index()` function in `shared_utils`.

---

## 2. Extinction and Cosmology

- **Galactic Extinction**: Optional; applied using the SFD dust map (`dustmaps.sfd.SFDQuery`) with extinction coefficients from `rubin_sim.phot_utils.DustValues`.
- **Distance Modulus**: Applied for each injected event based on its simulated distance.
- **Population distances**: Drawn from a volumetric distribution with configurable min/max redshift or distance bounds.

---

## 3. Detection Logic

Detection is defined as **either**:
1. **Intra-night detection**: ≥2 detections in the **same filter**, ≥30 minutes apart.
2. **Epoch-based detection**: ≥2 observing epochs, second epoch with ≥2 filters detected.

Events must have **SNR ≥ 5** in detections used for triggering.

---

## 4. Characterization Logic

A GRB is considered **characterized** if:
- ≥4 detections with **SNR ≥ 5**.
- At least **3 distinct filters**.
- Temporal coverage spanning **≥3 days and ≤14 days**.

This ensures enough temporal and color coverage for basic modeling and classification.

---

## 5. Spectroscopic Triggerability Logic

A GRB is **triggerable** for spectroscopy if:
- It is **detected**.
- **Within 2 days of peak**:
  - Brightness `< 21` mag in at least one filter.
  - Both detections have **SNR ≥ 5**.

This provides a realistic estimate of events that could be followed up with large telescopes before fading.

---

## 6. Metrics Implemented

| Metric Name | Purpose | Key Criteria |
|-------------|---------|--------------|
| `Detect_Metric` | Discovery efficiency | ≥2 detections in same filter ≥30 min apart OR ≥2 epochs w/ color info |
| `GRBAfterglowCharacterizeMetric` | Photometric characterization | ≥4 SNR≥5 detections, ≥3 filters, ≥3–14 days baseline |
| `GRBAfterglowSpecTriggerableMetric` | Spectroscopic triggerability | Bright <21 mag within 2 days, SNR≥5 |

All metrics record detailed **observation records** (`obs_records`) per event for later inspection.

---

## 7. How to Run

Example workflow (see provided notebook):

```python
from local_GRBafterglows_metric import (
    get_output_paths, load_or_generate_templates,
    load_or_generate_population, get_multi_metrics
)
import shared_utils

# Configure output paths
paths = get_output_paths("GRBafterglows")

# Load or generate light curve templates
lc_model = load_or_generate_templates(paths['templates_file'], generate_new=True)

# Load or generate population
slicer = load_or_generate_population(
    t_start=1, t_end=3652,
    d_min=None, d_max=None,
    z_min=0.00226, z_max=0.3,
    rate_density=1e-8,
    pop_file=paths['pop_file'],
    generate_new=True
)

# Run detection and characterization metrics
multi_metrics = get_multi_metrics(lc_model, include=['detect', 'characterize', 'spec_trigger'])
shared_utils.run_multi_metrics(multi_metrics, slicer, cadences=['baseline_v4.3.1_10yrs'])
