## LFBOTs

All of these metrics build on the same core model: a set of synthetic light curves generated from 
power-law decay models inspired by Zeh et al. (2005), applied within realistic extinction (using 
DustValues) and distance moduli. Each metric stores full per-event observation records for 
deeper analysis or visualization.

### Light Curve Modeling

- **Peak absolute magnitude range**: (-21.5, -20) 
- **Rise power law index α**: (-2.5, -0.25)
- **Fade power law index α**: (0.15, 0.45) 
- **Light curve model**:  
  `m(t) = m₀ + 2.5 × α × log₁₀(t / t₀) `  
  - **Rise time**: 1 day
  - **Post-jet break slope**: α = 10  
- **Filter treatment**: Light curve parameters between filters are currently uncorrelated


---

### Population Modeling

- **Volumetric rate**: `420e-9 Mpc⁻³ yr⁻¹`  
- **Distance range**: 10 to 1000 Mpc

---

### Detection Criteria

**Primary**  
- Require ≥2 distinct epochs of detection (≥30 minutes separation, ≤6 days total span)
- with at least one epoch showing detections in ≥2 different filters (to establish color and luminosity) 

**Fallback**  
- Fallback: If color information is not available, require ≥3 epochs (≥30 minutes separation, ≤6 days total span) to track fading behavior consistent with LFBOT timescales.
- (with the current cadence, color information should almost always be available)

---

### Characterization Criteria (Preliminary)

This metric checks whether Rubin captured enough of the light curve to inform follow-up and 
help us understand the transient’s behavior. 

We define photometric characterization as: 

- At least 4 detections with SNR ≥ 3 
- These detections span ≥ 3 days in the observer frame 
 
This setup is intentionally minimal but meaningful. It doesn’t require multi-filter coverage (since 
LFBOTs are mostly blue), and it doesn’t assume we’ve caught the peak, just that we’ve seen it 
evolve. 

In short: If Rubin can capture at least four data points over a few days above SNR=3, we can 
trace how the event fades and alert follow-up teams quickly. That’s what this metric tracks. 

---

### Other Metrics

_(To be specified)_