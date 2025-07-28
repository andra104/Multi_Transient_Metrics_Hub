from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues

import sys
import os
sys.path.append(os.path.abspath(".."))
from shared_utils import equatorialFromGalactic, uniform_sphere_degrees, inject_uniform_healpix, apply_spectral_index, evaluate

from rubin_sim.maf.utils import m52snr
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import Galactic, ICRS as ICRSFrame
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
#from rubin_sim.phot_utils import SFDMap
import astropy.units as u
import healpy as hp
from astropy.cosmology import z_at_value
from scipy.stats import truncnorm
import numpy as np
import glob
import os

import pickle 

DEBUG = False
MAXGAP = 1

# ------------------------------------------------------
# Light Curve Model
# ------------------------------------------------------
class LC:
    """
    Generate synthetic light curves for M Dwarf Flares.

    Light curves include a constant pre-flare phase, a fast rise, a sharp peak,
    and a fading tail. Time is represented as days from peak (t=0), and all filters
    use band-specific rise and fade rates, with quiescent magnitudes drawn from
    empirical distributions (UltraCoolSheet).
    """
    def __init__(self, num_samples=100, num_lightcurves=1000, load_from=None, delta_mag=5.0):
        self.data = []
        self.filts = ["u", "g", "r", "i", "z", "y"]
        self.delta_mag = delta_mag
        self.t_grid = None  # unified time array
        rng = np.random.default_rng(42)

        # Quiescent Magnitude Distributions (UltraCoolSheet)
        # Quiescent Magnitude Ranges (empirically motivated)
        QUIESCENT_MAG_RANGES = {
            'u': (17.5, 20.5),
            'g': (16.5, 19.5),
            'r': (15.0, 18.0),
            'i': (13.0, 15.5),
            'z': (12.0, 13.5),
            'y': (11.5, 12.7)
        }


        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                obj = pickle.load(f)
            self.data = obj['lightcurves']
            self.t_grid = obj['t_grid']
            print(f"Loaded LFBOT templates from {load_from}")
            return


        # Rise and fade rates - us is fake
        self.rise_rates = {'u': (0.01, 4.5), 'g': (0.009, 3.89), 'r': (0.005, 2.15), 'i': (0.0027, 0.415),
                           'z': (0.002, 0.113), 'y': (0.0014, 0.051)}
        self.fade_rates = {'u': (0.005, 1.0), 'g': (0.006, 0.79), 'r': (0.003, 0.44), 'i': (0.0019, 0.085),
                           'z': (0.001, 0.023), 'y': (0.001, 0.01)}

        #self.rise_rates['u'] = (0.01, 4.5) #fake
        #self.fade_rates['u'] = (0.005, 1.0) #fake


        # --- Time Grid Construction ---
        t_quiescent = np.linspace(-7.0, -0.05, num_samples // 5)
        t_rise = np.linspace(-0.05, 0, num_samples // 5)
        t_fade = np.linspace(0.01, 1.5, num_samples) #1.5 is correct 

        self.t_grid = np.concatenate([t_quiescent, t_rise, [0], t_fade])

        for _ in range(num_lightcurves):
            lc = {}
            for f in self.filts:
                qmin, qmax = QUIESCENT_MAG_RANGES[f]
                quiescent = rng.uniform(qmin, qmax)
                peak_mag = quiescent - delta_mag
                rise = rng.uniform(*self.rise_rates[f])
                fade = rng.uniform(*self.fade_rates[f])

                mag_quiescent = np.full_like(t_quiescent, quiescent)
                mag_rise = peak_mag - rise * (t_rise - np.min(t_rise)) / np.ptp(t_rise)
                mag_peak = np.full((1,), peak_mag)
                mag_fade = peak_mag + fade * (np.log10(1 + t_fade))

                mag_f = np.concatenate([mag_quiescent, mag_rise, mag_peak, mag_fade])

                lc[f] = {'ph': self.t_grid, 'mag': mag_f}

            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        return np.interp(t,
                         self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)

# ------------------------------------------------------
# Base MDwarfFlare Metric
# ------------------------------------------------------
class Base_Metric(BaseMetric):
    def __init__(self, metricName='Base_Metric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=60980.5, outputLc=False, badval=-666,
                 filter_include=None, load_from="MDwarfFlares_templates.pkl", use_extinction=True,
                 lc_model=None, **kwargs):

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
        self.extinction_printed = False

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName, units='Detection Efficiency', badval=badval, **kwargs)


    def detect(self, filters, snr, times, obs_record):
        detected = False
    
        filters = np.array(filters)
        snr = np.array(snr)
        times = np.array(times)
    
        idx_3sigma = snr >= 3
        idx_5sigma = snr >= 5
    
        # Option A: 3×3σ detections with at least one 5σ, all within 0.5 days
        if np.sum(idx_3sigma) >= 3:
            t_detected = times[idx_3sigma]
            if np.ptp(t_detected) <= 0.5 and np.any(idx_5sigma):
                detected = True
    
        # Option B: truly transient fallback — ≥2×5σ detections ≥15 min apart
        elif np.sum(idx_5sigma) >= 2:
            t_detected = times[idx_5sigma]
            if np.ptp(t_detected) >= 0.0104:  # 15 minutes
                detected = True
    
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
            'fade_time_days': fade_time,
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
            'theta_obs': slice_point['theta_obs'],
            'filter': obs_record.get('filter', np.array([])).tolist(),
            'distance_modulus': 5 * np.log10(slice_point['distance'] * 1e6) - 5
        })    

        self.obs_records[slice_point['sid']] = obs_record
        self.latest_obs_record = obs_record if detected else None
    
        return 1.0 if detected else 0.0


# --------------------------------------------
# Characterization Metric for M Dwarf Flares
# --------------------------------------------
class MDwarfFlareCharacterizeMetric(Base_Metric):
    """
    M Dwarf Flare Characterization Metric

    This metric evaluates whether Rubin observations of an M Dwarf flare contain
    sufficient information to distinguish between classical and complex (multi-peaked) profiles.

    Characterization logic:
    - Identify flare detections above 0.5σ (used to define flare start/stop time).
    - Require at least 4 such detections for any characterization.
    - Then raise the threshold to 1.5σ and count peaks:
        - If there are ≥2 peaks separated by ≥0.1 days -> complex flare
        - Otherwise → classical flare

    Returns:
        1.0 -> complex flare
        0.5 -> classical flare
        0.0 -> not characterizable
    """
    def __init__(self, **kwargs):
        use_extinction = kwargs.pop('use_extinction', True)
        super().__init__(**kwargs, use_extinction=use_extinction)
        self.metricName = kwargs.get('metricName', 'MDwarf_Characterize')
        self.obs_records = {}
        self.parent_instance = Base_Metric(use_extinction=use_extinction)

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)
        detected = self.parent_instance.detect(filters, snr, times, obs_record)

        if not detected:
            return 0.0

        # Step 1: Require ≥4 points above 0.5σ
        above_half_sigma = snr >= 0.5
        if np.sum(above_half_sigma) < 4:
            return 0.0

        # Step 2: Check for complexity via ≥2 peaks above 1.5σ, separated by ≥0.1 day
        above_onefive_sigma = snr >= 1.5
        t_peak = times[above_onefive_sigma]
        if len(t_peak) >= 2 and np.ptp(t_peak) >= 0.1:
            return 1.0  # Complex flare

        return 0.5  # Classical flare


# --------------------------------------------
# Multi_Metric Standardized Call
# --------------------------------------------
def get_multi_metrics(lc_model, include=None, use_extinction=True):
    """
    Return a list of metrics. `include` can be a list of metric names to include.
    """
    all_metrics = {
        'detect': Detect_Metric(lc_model=lc_model, use_extinction=use_extinction),
        'characterize': MDwarfFlareCharacterizeMetric(lc_model=lc_model, use_extinction=use_extinction),
    }

    if include is None:
        return list(all_metrics.values())
    else:
        return [all_metrics[name] for name in include if name in all_metrics]


