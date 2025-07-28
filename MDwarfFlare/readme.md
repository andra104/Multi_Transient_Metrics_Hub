### M Dwarf Flare Metric Implementation 

The MDwarfFlareMetric class is designed to simulate and analyze M Dwarf flares, accounting for both resolved (detectable flares) and unresolved (transient flares) scenarios.

**Key Components of the Code**

**M Dwarf Flare Light Curves:**

The MDwarfFlareLC class generates synthetic light curves for M Dwarf flares in the g, r, i, z, y filters based on realistic rise/fade rates and peak magnitudes.

Light curves are generated for num_samples (100 by default) using a normal distribution for rise and fade rates.

**MDwarf Flare Detection:**

Resolved Detection: A flare is detected if its signal-to-noise ratio (SNR) exceeds certain thresholds (5σ and 3σ). If the detected points are within a single observation night (separated by less than 0.5 days), it is classified as detected.

Unresolved Detection: For flares without a visible counterpart, at least two detections are required, separated by more than 15 minutes, to avoid detecting moving objects.

**Characterization:**

After detecting a flare, its characterization is performed. A flare is classified as:

**Single:** If the flare has a single peak.

**Complex:** If the flare has multiple peaks (with a separation of at least 0.1 days).

**Galactic Latitude Cut:**

A Galactic latitude cut (|b| < 30°) filters flares to simulate only those near the Galactic plane, where M Dwarfs are most common. This is done using the function equatorialFromGalactic, which converts Galactic coordinates to Equatorial coordinates.

**Slicer Generation:**

The slicer (MDwarfFlareSlicer) is responsible for generating flare positions and characteristics. This includes applying the Galactic latitude filter and generating flare properties such as peak_time, distance, and file_index.

**Plotting the Results:**
The MetricBundleGroup class groups multiple metrics (detection, classical, complex, unresolved) and plots the results using a Healpix SkyMap.

The script also provides a mechanism for saving the flare detection and classification results into CSV files (outfile for summary efficiency and typefile for flare classifications).