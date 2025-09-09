from rubin_sim.maf.metrics import BaseMetric

#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues

import sys
import os
sys.path.append(os.path.abspath(".."))
from shared_utils import equatorialFromGalactic, uniform_sphere_degrees, inject_uniform_healpix, apply_spectral_index, evaluate

import matplotlib.pyplot as plt 
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import Galactic, ICRS as ICRSFrame
from astropy.coordinates import SkyCoord
#from rubin_sim.phot_utils import SFDMap
import astropy.units as u
import healpy as hp
from astropy.cosmology import z_at_value
from scipy.stats import truncnorm
import numpy as np
import glob
import os

import pickle 
from pathlib import Path

DEBUG = False
# MAXGAP = 1

FILTER_CENTRAL_FREQS = {
    'u': 8.088e14,
    'g': 6.293e14,
    'r': 4.844e14,
    'i': 3.979e14,
    'z': 3.461e14,
    'y': 3.080e14,
}

# # ------------------------------------------------------
# # Light Curve Model
# # ------------------------------------------------------
# class LC:
#     """
#     Generate synthetic light curves for M Dwarf Flares.

#     Light curves include a constant pre-flare phase, a fast rise, a sharp peak,
#     and a fading tail. Time is represented as days from peak (t=0), and all filters
#     use band-specific rise and fade rates, with quiescent magnitudes drawn from
#     empirical distributions (UltraCoolSheet).
#     """
#     def __init__(self, num_samples=100, num_lightcurves=1000, load_from=None, delta_mag=5.0):
#         self.data = []
#         self.filts = ["u", "g", "r", "i", "z", "y"]
#         self.delta_mag = delta_mag
#         self.t_grid = None  # unified time array
#         rng = np.random.default_rng(42)

#         # Quiescent Magnitude Distributions (UltraCoolSheet)
#         # Quiescent Magnitude Ranges (empirically motivated)
#         QUIESCENT_MAG_RANGES = {
#             'u': (17.5, 20.5),
#             'g': (16.5, 19.5),
#             'r': (15.0, 18.0),
#             'i': (13.0, 15.5),
#             'z': (12.0, 13.5),
#             'y': (11.5, 12.7)
#         }


#         if load_from and os.path.exists(load_from):
#             with open(load_from, 'rb') as f:
#                 obj = pickle.load(f)
#             self.data = obj['lightcurves']
#             self.t_grid = obj['t_grid']
#             print(f"Loaded LFBOT templates from {load_from}")
#             return

#         # RILEY: in g as an example, a typical flare on a mid range dwarf would have a brightness change of 0.7-1 mag, then would return to quiescence in less than hour. 
#         #RILEY: emperical flare template 
        

#         # Rise and fade rates - u is fake
#         #Rise: Quiescent to Peak brightness
#         #Fade: Peak to Return to quiescence
#         self.rise_rates = {'u': (0.01, 4.5), 'g': (0.009, 3.89), 'r': (0.005, 2.15), 'i': (0.0027, 0.415),
#                            'z': (0.002, 0.113), 'y': (0.0014, 0.051)}
#         self.fade_rates = {'u': (0.005, 1.0), 'g': (0.006, 0.79), 'r': (0.003, 0.44), 'i': (0.0019, 0.085),
#                            'z': (0.001, 0.023), 'y': (0.001, 0.01)}

#         #self.rise_rates['u'] = (0.01, 4.5) #fake
#         #self.fade_rates['u'] = (0.005, 1.0) #fake


#         # --- Time Grid Construction ---
#         # Units: days from peak.
#         # Pre-flare: Up to 7 days before peak
#         # Rise: Short ramp-up (~1.2 hours before peak)
#         # Fade: Return to quiescence over ~1.5 days
#         # Combines all phases into a unified time axis

#         t_quiescent = np.linspace(-7.0, -0.05, num_samples // 5)
#         t_rise = np.linspace(-0.05, 0, num_samples // 5)
#         t_fade = np.linspace(0.01, 1.5, num_samples) #1.5 is correct 
#         self.t_grid = np.concatenate([t_quiescent, t_rise, [0], t_fade])

#         for _ in range(num_lightcurves):
#             lc = {}
#             for f in self.filts:
#                 #Draw a random baseline brightness from range
#                 qmin, qmax = QUIESCENT_MAG_RANGES[f]
#                 quiescent = rng.uniform(qmin, qmax)
                
#                 # Flares brighten the star by delta_mag magnitudes (default = 5 mag) (could be the problem)
#                 peak_mag = quiescent - delta_mag
#                 rise = rng.uniform(*self.rise_rates[f])
#                 fade = rng.uniform(*self.fade_rates[f])

#                 # Quiescent: Constant magnitude
#                 # Rise: Steep brightening
#                 # Peak: Single point at t=0
#                 # Fade: Slowly dims back toward quiescence
                
#                 mag_quiescent = np.full_like(t_quiescent, quiescent)
#                 mag_rise = peak_mag - rise * (t_rise - np.min(t_rise)) / np.ptp(t_rise)
#                 mag_peak = np.full((1,), peak_mag)
#                 mag_fade = peak_mag + fade * (np.log10(1 + t_fade))

#                 #Joins all phases into one magnitude array
#                 mag_f = np.concatenate([mag_quiescent, mag_rise, mag_peak, mag_fade])

#                 #ph: Phase/time values
#                 #mag: Magnitudes at each phase
#                 lc[f] = {'ph': self.t_grid, 'mag': mag_f}

#             self.data.append(lc)

#     def interp(self, t, filtername, lc_indx=0):
#         # Retrieves interpolated magnitude for time(s) t
#         # lc_indx: Which flare template to use
#         # Outside valid range, return 99 (treated as no data)
#         return np.interp(t,
#                          self.data[lc_indx][filtername]['ph'],
#                          self.data[lc_indx][filtername]['mag'],
#                          left=99, right=99)

#version 2 with quiescent (obvious)
# class LC:
#     """
#     Flare-only light curves for M Dwarf flares.
#     - No quiescent baseline in the curve (only flare)
#     - Peak amplitude: 0.7–1.0 mag brighter than quiescent
#     - Duration: ~1 hour total (rise + fade)
#     """
#     def __init__(self, num_samples=50, num_lightcurves=1000, load_from=None):
#         self.data = []
#         self.filts = ["u", "g", "r", "i", "z", "y"]
#         self.t_grid = None
#         rng = np.random.default_rng(42)

#         # Quiescent magnitude ranges (reference only)
#         QUIESCENT_MAG_RANGES = {
#             'u': (17.5, 20.5),
#             'g': (16.5, 19.5),
#             'r': (15.0, 18.0),
#             'i': (13.0, 15.5),
#             'z': (12.0, 13.5),
#             'y': (11.5, 12.7)
#         }

#         # Load from pickle if provided
#         if load_from and os.path.exists(load_from):
#             with open(load_from, 'rb') as f:
#                 obj = pickle.load(f)
#             self.data = obj['lightcurves']
#             self.t_grid = obj['t_grid']
#             print(f"Loaded flare templates from {load_from}")
#             return

#         # Rise and fade rates tuned for ~1 hour duration
#         self.rise_rates = {
#             'u': (5.0, 15.0), 'g': (4.0, 12.0), 'r': (3.0, 10.0),
#             'i': (2.0, 8.0),  'z': (1.5, 6.0),  'y': (1.0, 5.0)
#         }
#         self.fade_rates = {
#             'u': (8.0, 20.0), 'g': (7.0, 18.0), 'r': (6.0, 16.0),
#             'i': (5.0, 14.0), 'z': (4.0, 12.0), 'y': (3.0, 10.0)
#         }

#         # Time grid: ~1 hour (0.0417 days)
#         t_rise = np.linspace(-0.005, 0, num_samples // 4)     # ~7 minutes
#         t_fade = np.linspace(0.0007, 0.0417, num_samples)     # ~1 hour fade
#         self.t_grid = np.concatenate([t_rise, [0], t_fade])

#         for _ in range(num_lightcurves):
#             lc = {}
#             for f in self.filts:
#                 # Quiescent magnitude (not plotted, only for amplitude reference)
#                 qmin, qmax = QUIESCENT_MAG_RANGES[f]
#                 quiescent = rng.uniform(qmin, qmax)

#                 # Peak magnitude: 0.7–1.0 mag brighter than quiescent
#                 peak_mag = quiescent - rng.uniform(0.7, 1.0)

#                 rise = rng.uniform(*self.rise_rates[f])
#                 fade = rng.uniform(*self.fade_rates[f])

#                 # Rise portion
#                 mag_rise = peak_mag - rise * (t_rise - np.min(t_rise)) / np.ptp(t_rise)
#                 mag_peak = np.full((1,), peak_mag)
#                 mag_fade = peak_mag + fade * (np.log10(1 + t_fade))

#                 mag_f = np.concatenate([mag_rise, mag_peak, mag_fade])

#                 lc[f] = {
#                     'ph': self.t_grid,
#                     'mag': mag_f,
#                     'quiescent': quiescent
#                 }

#             self.data.append(lc)

#     def interp(self, t, filtername, lc_indx=0):
#         return np.interp(
#             t,
#             self.data[lc_indx][filtername]['ph'],
#             self.data[lc_indx][filtername]['mag'],
#             left=99, right=99
#         )

# class LC: #with drop set 
#     """
#     Flare-only light curves for M Dwarf flares.
#     - Instant drop back to quiescent after peak
#     - Peak amplitude: 0.7–1.0 mag brighter than quiescent
#     - Duration: rise (~minutes) + instant drop
#     """
#     def __init__(self, num_samples=50, num_lightcurves=1000, load_from=None):
#         self.data = []
#         self.filts = ["u", "g", "r", "i", "z", "y"]
#         rng = np.random.default_rng(42)

#         QUIESCENT_MAG_RANGES = {
#             'u': (17.5, 20.5),
#             'g': (16.5, 19.5),
#             'r': (15.0, 18.0),
#             'i': (13.0, 15.5),
#             'z': (12.0, 13.5),
#             'y': (11.5, 12.7)
#         }

#         if load_from and os.path.exists(load_from):
#             with open(load_from, 'rb') as f:
#                 obj = pickle.load(f)
#             self.data = obj['lightcurves']
#             self.t_grid = obj['t_grid']
#             print(f"Loaded flare templates from {load_from}")
#             return

#         # Time grid: steep rise, peak, instant drop
#         t_rise = np.linspace(-0.005, 0, num_samples // 4)  # ~7 min rise
#         t_drop = np.array([0.0007])  # instant drop step
#         self.t_grid = np.concatenate([t_rise, [0], t_drop])  # shared for all templates

#         for _ in range(num_lightcurves):
#             lc = {}
#             for f in self.filts:
#                 qmin, qmax = QUIESCENT_MAG_RANGES[f]
#                 quiescent = rng.uniform(qmin, qmax)
#                 peak_mag = quiescent - rng.uniform(0.7, 1.0)

#                 # Rise time from start to peak
#                 rise_time_days = t_rise[-1] - t_rise[0]
#                 fade_time_days = 0.0  # instant drop model


#                 rise_rate = rng.uniform(5, 15)
#                 mag_rise = peak_mag - rise_rate * (t_rise - np.min(t_rise)) / np.ptp(t_rise)

#                 mag_peak = np.array([peak_mag])
#                 mag_quiescent = np.array([quiescent])

#                 mag_f = np.concatenate([mag_rise, mag_peak, mag_quiescent])
#                 lc[f] = {'ph': self.t_grid, 'mag': mag_f, 'rise_time_days': rise_time_days, 'fade_time_days': fade_time_days}

#             self.data.append(lc)

#     def interp(self, t, filtername, lc_indx=0):
#         return np.interp(
#             t,
#             self.data[lc_indx][filtername]['ph'],
#             self.data[lc_indx][filtername]['mag'],
#             left=99, right=99
#         )

#9/3-9/4
class LC:
    """
    M-dwarf flare light curves with shared per-event parameters:
      - m_quiescent_ref : quiescent mag in a reference band (default r)
      - delta_mag_ref   : peak brightening (mag) in the reference band
      - fade_time       : days to return to quiescent
      - gamma           : fade curve shape (0.5–0.9 = fast early fade)
      - beta_quiescent  : quiescent SED slope (Fν ∝ ν^β) to get per-filter quiescent mags
      - amp_index       : amplitude color law exponent; Δmag_f ∝ (ν_f/ν_ref)^{amp_index}

    Support is post-peak only (t>0). Pre-peak samples return NaN.
    """

    def __init__(self, num_lightcurves=1000, load_from=None, ref_filter="r"):
        import numpy as np, pickle, os

        self.filts = ["u", "g", "r", "i", "z", "y"]
        self.data = []
        self.ref_filter = ref_filter

        # Shared bounds used by evaluate() to understand support
        # (we model up to 12 hours after peak)
        self.t_grid = np.geomspace(1e-5, 0.5, 60)

        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                obj = pickle.load(f)
            self.data = obj['lightcurves']
            self.t_grid = obj.get('t_grid', self.t_grid)
            print(f"Loaded flare templates from {load_from}")
            return

        rng = np.random.default_rng(42)

        # Parameter priors (tweak as desired)
        # Quiescent (anchor in r-band to match your earlier ranges)
        MQUI_R_RANGE   = (15.0, 18.0)    # mag in r
        DELTA_MAG_REF  = (0.7, 1.0)      # mag brightening at peak in ref band
        FADE_TIME_RANGE= (0.02, 0.17)    # days (≈30 min – 4 hr)
        GAMMA_RANGE    = (0.5, 0.9)      # shape of fade
        BETA_Q_RANGE   = (-1.3, -0.7)    # red SED (M dwarfs): Fν ∝ ν^β, β ~ -1
        AMP_INDEX_RANGE= (0.5, 1.0)      # stronger amplitude at higher ν

        nu_ref = FILTER_CENTRAL_FREQS[self.ref_filter]

        for _ in range(num_lightcurves):
            # Shared per-event parameters
            m_quiescent_ref = rng.uniform(*MQUI_R_RANGE)
            delta_mag_ref   = rng.uniform(*DELTA_MAG_REF)
            fade_time       = rng.uniform(*FADE_TIME_RANGE)
            gamma           = rng.uniform(*GAMMA_RANGE)
            beta_quiescent  = rng.uniform(*BETA_Q_RANGE)
            amp_index       = rng.uniform(*AMP_INDEX_RANGE)

            # Event-specific time grid (up to ~3×fade_time, capped at 0.5 d)
            t_end = min(3.0 * fade_time, 0.5)
            t_ev  = np.geomspace(1e-5, t_end, 48)

            lc = {}
            for f in self.filts:
                # Quiescent per filter from a simple power-law SED
                m_quiescent_f = apply_spectral_index(
                    m_quiescent_ref, f, ref_filter=self.ref_filter, beta=beta_quiescent
                )

                # Amplitude color law: bigger Δ in blue
                nu_f = FILTER_CENTRAL_FREQS[f]
                amp_scale = (nu_f / nu_ref) ** amp_index
                delta_mag_f = delta_mag_ref * amp_scale
                # Optionally clamp to keep things sane
                delta_mag_f = float(np.clip(delta_mag_f, 0.3, 1.8))

                m_peak_f = m_quiescent_f - delta_mag_f

                # Fade curve (power-law-ish in time fraction)
                #9/5 tail-free (recommended)
                eps  = 1e-6  # days; keep us just shy of fade_time
                mask = t_ev < (fade_time - eps)
                t_use = t_ev[mask]
                
                # if fade_time is so short that mask is empty, keep a single peak sample
                if t_use.size == 0:
                    t_use = np.array([1e-5])
                    mag_use = np.array([m_peak_f])
                else:
                    frac    = t_use / fade_time            # in (0,1)
                    mag_use = m_peak_f + delta_mag_f * (frac ** gamma)
                
                lc[f] = {
                    "ph":  t_use,          # no samples at/after fade_time
                    "mag": mag_use,
                    "quiescent_mag": m_quiescent_f,
                    "delta_mag":     delta_mag_f,
                    "fade_time":     fade_time,
                    "gamma":         gamma,
                    "m_quiescent_ref": m_quiescent_ref,
                    "delta_mag_ref":   delta_mag_ref,
                    "beta_quiescent":  beta_quiescent,
                    "amp_index":       amp_index,
                    "ref_filter":      self.ref_filter,
                }


            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        """
        Post-peak, log-time interpolation (like GRB).
        Pre-peak (t<=0) → NaN.
        """
        import numpy as np

        ph  = np.asarray(self.data[lc_indx][filtername]["ph"],  dtype=float)  # >0
        mag = np.asarray(self.data[lc_indx][filtername]["mag"], dtype=float)

        t_arr = np.asarray(t, dtype=float)
        # Map t<=0 → NaN. Clamp >0 to small floor to avoid log10(0)
        t_pos = np.where(t_arr > 0, t_arr, np.nan)
        t_pos = np.clip(t_pos, 1e-5, None)

        xp = np.log10(ph)
        x  = np.log10(t_pos)

        out = np.interp(x, xp, mag, left=np.nan, right=np.nan)
        out[~np.isfinite(t_arr) | (t_arr <= 0)] = np.nan
        return out



# ------------------------------------------------------
# Base MDwarfFlare Metric
# ------------------------------------------------------
#9/3 updated for new format 
class Base_Metric(BaseMetric):
    def __init__(self, metricName='BaseMDwarfFlareMetric', 
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night', mjd0=60980.5, outputLc=False, badval=-666,
                 filter_include=None,
                 load_from="MDwarfFlares_templates.pkl",
                 lc_model=None, use_extinction=True, use_kcorrect=False, k_correct_type=None, k_correct_arg=None,
                 # NEW: diagnostic sampling controls
                 diag_store=True,
                 diag_sample_rate=0.01,       # Bernoulli keep prob for each obs row
                 diag_per_event_cap=30,       # hard cap per event after sampling
                 diag_min_snr=None,           # e.g., 3 or 5 to gate by SNR
                 diag_max_mag=None,           # e.g., 24 to gate by brightness
                 **kwargs):
        """
        Parameters
        ----------
        lc_model : LC or None
            M Dwarf Flare model object. If None, load from file.
        """
        self.diag_store = diag_store
        self.diag_sample_rate = float(diag_sample_rate)
        self.diag_per_event_cap = int(diag_per_event_cap)
        self.diag_min_snr = None if diag_min_snr is None else float(diag_min_snr)
        self.diag_max_mag = None if diag_max_mag is None else float(diag_max_mag)
        self._rng = np.random.default_rng(12345)  # stable, cheap
        
        if lc_model is not None:
            self.lc_model = lc_model
        else:
            self.lc_model = LC(load_from=load_from)

        self.ax1 = DustValues().ax1  # From rubin_sim.phot_utils
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.mjd0 = mjd0
        self.outputLc = outputLc
        self.filter_include = filter_include
        self.use_extinction = use_extinction
        self.use_kcorrect = use_kcorrect
        self.k_correct_type = k_correct_type
        self.k_correct_arg = k_correct_arg
        self.extinction_printed = False

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName,
                         units='Detection Efficiency',
                         badval=badval, **kwargs)



    def detect(self, filters, snr, times, obs_record):
        """
        Detection Criteria:

        Option A:  ≥3 detections ≥3σ  AND  ≥1 detection ≥5σ, All within 0.5 days
        
        Option B:  ≥2 detections ≥5σ  AND  separated by ≥15 min (0.0104 days)
        
        If either condition is met, detected = True
        Else, detected = False
        """
        detected = False
    
        filters = np.array(filters)
        snr = np.array(snr)
        times = np.array(times)
    
        # idx_3sigma = snr >= 3
        idx_5sigma = snr >= 5
    
        # # Option A: 3x3σ detections with at least one 5σ, all within 0.5 days - VERY STRINGENT
        # if np.sum(idx_3sigma) >= 3:
        #     t_detected = times[idx_3sigma]
        #     if np.ptp(t_detected) <= 0.5 and np.any(idx_5sigma):
        #         detected = True
    
        # # Option B: truly transient fallback — ≥2x5σ detections ≥15 min apart - VERY STRINGENT
        # elif np.sum(idx_5sigma) >= 2:
        #     t_detected = times[idx_5sigma]
        #     if np.ptp(t_detected) >= 0.0104:  # 15 minutes
        #         detected = True

        # # Option C: Short-timescale detection (relaxed) 
        # # ≥3 detections ≥3σ within 3 days, at least one ≥5σ
        # if np.sum(idx_3sigma) >= 3:
        #     t_detected = times[idx_3sigma]
        #     if np.ptp(t_detected) <= 3.0 and np.any(idx_5sigma):  # was 0.5 days
        #         detected = True
    
        # # Option D: Strong detections over survey timescale 
        # # ≥2 detections ≥5σ in ANY two epochs (no strict same-night requirement)
        # elif np.sum(idx_5sigma) >= 2:
        #     detected = True 

        #shar sept
        if np.sum(idx_5sigma)>0:
            detected=True
    
        return detected

# --------------------------------------------
# Detection Metric for M Dwarf Flares
# --------------------------------------------
class Detect_Metric(Base_Metric):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'Detect')
        self.use_kcorrect = False #shar hardcoded
        self.obs_records = {}  # <-- NEW: to store all detected event records individually
        self.parent_instance = Base_Metric()


    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)
    
        if obs_record is None:
            # print("OBSRECORD IS NONE SHAR")
            return self.badval
    
        if self.filter_include is not None:
            # print("filter include stuff shar")
            keep = np.isin(filters, self.filter_include)
            snr = snr[keep]
            filters = filters[keep]
            times = times[keep]
            for k in ['mjd_obs', 'mag_obs', 'snr_obs', 'filter']:
                if isinstance(obs_record[k], np.ndarray):
                    obs_record[k] = obs_record[k][keep]
 
        detected = self.parent_instance.detect(filters, snr, times, obs_record)
    
        detected_mask = snr >= 5
        first_det_mjd = np.nan
        last_det_mjd = np.nan
        #rise_time = np.nan
        fade_time = np.nan

        per_filter_min_mag = {}
        for f in np.unique(obs_record['filter']):
            fmask = (obs_record['filter'] == f)
            if np.any(fmask):
                vals = np.asarray(obs_record['mag_obs'][fmask], dtype=float)
                good = np.isfinite(vals) & (vals < 90)   # drop sentinels if any
                per_filter_min_mag[f] = float(np.nanmin(vals[good])) if np.any(good) else np.nan
        obs_record['per_filter_min_mag'] = per_filter_min_mag
    
        if np.any(detected_mask):
            first_det_mjd = obs_record['mjd_obs'][detected_mask].min()
            last_det_mjd = obs_record['mjd_obs'][detected_mask].max()
            #rise_time = first_det_mjd - (self.mjd0 + slice_point['peak_time'])
            fade_time = last_det_mjd - (self.mjd0 + slice_point['peak_time'])
    
        peak_index = np.argmin(obs_record['mag_obs'])
        peak_mjd = obs_record['mjd_obs'][peak_index] if len(obs_record['mjd_obs']) > 0 else np.nan
        peak_mag = obs_record['mag_obs'][peak_index] if len(obs_record['mag_obs']) > 0 else np.nan
    
        obs_record.update({
            'first_det_mjd': first_det_mjd,
            'last_det_mjd': last_det_mjd,
            #'rise_time_days': rise_time,
            # 'fade_time_days': fade_time,
            'sid': slice_point['sid'],
            'file_indx': slice_point['file_indx'],
            'ra': slice_point['ra'],
            'dec': slice_point['dec'],
            'distance_Mpc': slice_point['distance'],
            'peak_mjd_observed': peak_mjd,
            'peak_mag_observed': peak_mag,
            'ebv': slice_point['ebv'],
            'peak_time': slice_point['peak_time'],
            'detected': bool(detected),
            'mag_obs': obs_record.get('mag_obs', np.array([])).tolist(),
            'snr_obs': obs_record.get('snr_obs', np.array([])).tolist(),
            'mjd_obs': obs_record.get('mjd_obs', np.array([])).tolist(),
            # 'theta_obs': slice_point['theta_obs'],
            'filter': obs_record.get('filter', np.array([])).tolist(),
            # 'rise_time_model_days': slice_point['rise_time_days'], #shar commenting out cause they done work right now
            # 'fade_time_model_days': slice_point['fade_time_days'],

            'distance_modulus': 5 * np.log10(slice_point['distance'] * 1e6) - 5
        })    

        self.obs_records[slice_point['sid']] = obs_record
        self.latest_obs_record = obs_record if detected else None
    
        return 1.0 if detected else 0.0


# --------------------------------------------
# Characterization Metric for M Dwarf Flares
# --------------------------------------------
class MDwarfFlareSilverMetric(Base_Metric):
    """
    M Dwarf Flare Silver Characterization Metric

    Can we get two data points, the first one within 1/e timescales? 
    """
    def __init__(self, **kwargs):
        use_extinction = kwargs.pop('use_extinction', True)
        super().__init__(**kwargs, use_extinction=use_extinction)
        self.metricName = kwargs.get('metricName', 'MDwarf_Silver')
        self.obs_records = {}
        self.parent_instance = Base_Metric(use_extinction=use_extinction)

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)
        detected = self.parent_instance.detect(filters, snr, times, obs_record)

        if np.sum(detected)<2: #shar sept
            return 0.0

        #todo: implement 1/e timescale criterion
        # #this is in survey time not mjd
        t_start   = self.lc_model.data[slice_point['file_indx']][f]['ph'][0] + slice_point['peak_time'] 
        t_end   = self.lc_model.data[slice_point['file_indx']][f]['ph'][:-1] + slice_point['peak_time']
        duration = t_end - t_start

        times_good = times[snr > 5]
        if np.any(times_good) < (t_start + duration/np.e):
            return 1.0
            

        return 0.0
        
# --------------------------------------------
# Gold Metric for M Dwarf Flares
# --------------------------------------------
class MDwarfFlareGoldMetric(Base_Metric): #shar sept
    """
    M Dwarf Flare Characterization Metric

    Can we get two data points in the same filter, the first one within 1/e timescales? 
    """
    def __init__(self, **kwargs):
        use_extinction = kwargs.pop('use_extinction', True)
        super().__init__(**kwargs, use_extinction=use_extinction)
        self.metricName = kwargs.get('metricName', 'MDwarf_Gold')
        self.obs_records = {}
        self.parent_instance = Base_Metric(use_extinction=use_extinction)

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)
        detected = self.parent_instance.detect(filters, snr, times, obs_record)

        if np.sum(detected)<2: #shar sept
            return 0.0
            
        t_start   = self.lc_model.data[slice_point['file_indx']][f]['ph'][0] + slice_point['peak_time'] 
        t_end   = self.lc_model.data[slice_point['file_indx']][f]['ph'][:-1] + slice_point['peak_time']
        duration = t_end - t_start
        
        for f in np.unique(filters):
            mask = filters == f
            snr_in_filter = snr[mask]
            if np.sum(snr_in_filter>5)<2:
                return 0.0
            times_in_filter = times[mask]
            observed_detection_times = times_in_filter[snr_in_filter >= 5]
            if np.any(observed_detection_times) < (t_start + duration/np.e):
                return 1.0
            

        return 0.0

# --------------------------------------------
# Multi_Metric Standardized Call
# --------------------------------------------
def get_multi_metrics(lc_model, include=None, use_extinction=True):
    """
    Return a list of metrics. `include` can be a list of metric names to include.
    """
    all_metrics = {
        'detect': Detect_Metric(lc_model=lc_model, use_extinction=use_extinction,use_kcorrect=False),
        'silver': MDwarfFlareSilverMetric(lc_model=lc_model, use_extinction=use_extinction,use_kcorrect=False),
        'gold': MDwarfFlareGoldMetric(lc_model=lc_model, use_extinction=use_extinction,use_kcorrect=False)
    }

    if include is None:
        return list(all_metrics.values())
    else:
        return [all_metrics[name] for name in include if name in all_metrics]

