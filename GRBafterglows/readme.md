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

A GRB afterglow is considered **detected** if the survey captures **at least two high-significance observations in the same filter** that meet all of the following criteria:

1. **Signal-to-noise ratio (SNR) ≥ 5** for both detections.
2. **Minimum separation** of `0.5 hours` (`0.05 / 24` hours) between the first and second detection in that filter.
3. **Maximum allowed gap** (`MAXGAP`) of `1 day` between any two detections in that filter.

The logic proceeds as follows:
- Loop over each LSST filter (`u, g, r, i, z, y`) that has been observed for the event.
- Select only the observation times in that filter where `SNR ≥ 5`.
- Check whether there are **at least two** such detections.
- If yes, verify that:
  - The **time span** between the earliest and latest qualifying detections in that filter is ≥ 0.5 hours.
  - The **shortest gap** between consecutive detections does not exceed `1 day`.
- If both conditions are satisfied in **any filter**, the event is flagged as **detected**.

**This ensures that detections are not due to a single spurious point,  
and that there is enough temporal spacing to confirm a real astrophysical transient.**
---

## 4. Characterization Logic

A GRB afterglow is considered **characterized** if it satisfies all of the following conditions:

1. **It has already been detected** according to the detection logic.
2. At least **4 observations** with `SNR ≥ 5` exist for the event.
3. These qualifying observations span at least **3 distinct LSST filters**.
4. The **time baseline** between the earliest and latest qualifying detections is:
   - At least `3 days` (ensuring temporal coverage of the evolving light curve), and
   - At most `14 days` (ensuring the sampling is relevant to the rapid-evolution phase of GRB afterglows).

The logic proceeds as follows:
- Use the same detection procedure described in the Detection Logic section to confirm the event is real.
- Select only the observations with `SNR ≥ 5`.
- Count how many of these observations exist (`≥ 4 required`).
- Check how many **unique filters** are represented (`≥ 3 required`).
- Compute the total **duration** (`latest_time - earliest_time`) of these observations.
- Accept the event as **characterized** if both the minimum and maximum duration constraints are met.

This ensures that an event is not just discovered but is followed with **sufficient multi-band and temporal coverage**  
to allow meaningful modeling of its color evolution and decay rate, both of which are key to distinguishing GRB afterglows from other fast-evolving transients

---

## 5. Spectroscopic Triggerability Logic

A GRB afterglow is considered **spectroscopically triggerable** if it meets the following criteria:

1. **It has already been detected** according to the detection logic.
2. The event has at least **two detections in the same filter** with:
   - `SNR ≥ 5` for both measurements.
   - Brightness `< 21 mag` in that filter for at least one of the measurements.
3. At least one of these bright detections occurs **within 2 days of the modeled peak time**.

The logic proceeds as follows:
- Confirm the event passes the **detection** test.
- Sort the event’s observations in time.
- Loop over each filter observed for the event.
- Select only the observations in that filter with `SNR ≥ 5`.
- Require **at least two** such observations in the same filter to avoid spurious triggers.
- Among those detections, identify the ones brighter than `21 mag`.
- If any bright detection occurs **within ±2 days** of the modeled peak time,  
  the event is marked as **spectroscopically triggerable**.

This logic identifies events that are not only detectable but **bright and early enough** for rapid spectroscopic follow-up,  
increasing the likelihood that follow-up telescopes can capture valuable early-time afterglow spectra before significant fading occurs.


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
