## LFBOTs

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

### Other Metrics

_(To be specified)_
