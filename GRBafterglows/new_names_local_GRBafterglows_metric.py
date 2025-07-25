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



# -------------------------------------------
# Generate Single Light Curve from Rc Parameters
# -------------------------------------------
def generate_grb_lc_from_rc(mag_peak_rc, alpha, t_jetbreak):
    """
    Generate a GRB light curve based on peak Rc mag, decay slope, and jet break.
    """
    t_grid = np.array([0.01, t_jetbreak, 100])  # days
    mag_grid = np.zeros_like(t_grid)

    # Enforce initial point is the peak
    mag_grid[0] = mag_peak_rc

    # Pre-break power-law decay
    mag_grid[1] = mag_grid[0] + 2.5 * alpha * np.log10(t_jetbreak / 0.01) 

    # Post-break segment: steeper decline
    t_end = t_grid[2]
    new_decay = 10 * (np.log10(t_end + 1)) - 10 * (np.log10(t_jetbreak + 1))
    mag_grid[2] = mag_grid[1] + new_decay

    return t_grid, mag_grid
    
# --------------------------------------------
# Power-law GRB afterglow model based on Zeh et al. (2005) - Rc reference 
# --------------------------------------------
class LC:
    """
    Simulate GRB afterglow light curves with Rc reference band using power-law + break.
    Stores light curves in ugrizy using spectral index from Rc.
    Peak mag range is now determined from Cenko et al. (2009) with redshift and k-correction 

    Light curves follow:
        m(t) = m_0 + 2.5 * alpha * log10(t/t_0)
    where alpha is the temporal slope (decay), t is time (days),
    and m_0 is the peak magnitude (from Zeh et al. 2005).

    The light curve begins at peak magnitude, and the decay is positive (fading).
    """
    def __init__(self, num_lightcurves=1000, load_from=None):
        self.filts = ['u', 'g', 'r', 'i', 'z', 'y']
        self.data = []
        self.t_grid = None  # Set per-lightcurve

        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                obj = pickle.load(f)
            self.data = obj['lightcurves']
            self.t_grid = obj['t_grid']
            print(f"Loaded GRB afterglow templates from {load_from}")
            return

        rng = np.random.default_rng(42)
        peak_mag_range = (-31.6, -18.47) 
        # peak_mag_range = (-25.1,-25) #shar

        for _ in range(num_lightcurves):
            # --- Draw intrinsic Rc properties
            mag_peak_rc = rng.uniform(*peak_mag_range)

            a, b = (.5 - 1.5) / .5, (1.7 - 1.5) / .5
            trunc_alpha = truncnorm(a=a, b=b, loc=1.5, scale=.5)
            alpha_fade = trunc_alpha.rvs(random_state=rng)

            t_jetbreak = rng.uniform(1, 5)  # Days
            
            # --- Build Rc light curve
            t_vals, mag_rc = generate_grb_lc_from_rc(mag_peak_rc, alpha_fade, t_jetbreak)
            self.t_grid = t_vals

            # --- Project to other filters using spectral index
            lc = {}
            for f in self.filts:
                mag_filt = apply_spectral_index(mag_rc, f, ref_filter="Rc", beta=-0.75)
                lc[f] = {"ph": t_vals, "mag": mag_filt}
            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        return np.interp( #interpolate 
            t,
            self.data[lc_indx][filtername]["ph"],
            self.data[lc_indx][filtername]["mag"],
            left=99, right=99,
        )
        




# --------------------------------------------
# Base GRB Metric with extinction and SNR
# --------------------------------------------
class Base_Metric(BaseMetric):
    def __init__(self, metricName='BaseGRBAfterglowMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=60980.5, outputLc=False, badval=-666,
                 filter_include=None,
                 load_from="GRBAfterglow_templates.pkl",
                 lc_model=None, use_extinction=True,  # <-- NEW
                 **kwargs):
        """
        Parameters
        ----------
        lc_model : LC or None
            Shared GRB light curve model object. If None, load from file.
        """
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
        super().__init__(col=cols, metric_name=metricName,
                         units='Detection Efficiency',
                         badval=badval, **kwargs)




    def detect(self, filters, snr, times, obs_record):
        detected = False        
        # -------- Detection Logic --------
        # Option A: 2 detections in same filter ≥30min apart
        
        
        # f='g' #use this to test just one filter
        # mask = filters == f
        # times_in_filter = times[mask]
        # snr_in_filter = snr[mask]
        # observed_detection_times = times_in_filter[snr_in_filter >= 5]
        # if len(observed_detection_times)>=2: #require 2+ detections
        #     if np.ptp(observed_detection_times) >= .5 / 24 and np.diff(np.sort(observed_detection_times)).min() <= MAXGAP :
        #         detected = True
                  
        # if len(observed_detection_times)>=1: #use this if you want to test just one detection
        #     detected = True 
        
        MAXGAP = 1
            
        for f in np.unique(filters):
            mask = filters == f
            times_in_filter = times[mask]
            snr_in_filter = snr[mask]
            observed_detection_times = times_in_filter[snr_in_filter >= 5]
            if len(observed_detection_times)>=2: #require 2+ detections
                if np.ptp(observed_detection_times) >= .5 / 24 and np.diff(np.sort(observed_detection_times)).min() <= MAXGAP :
                    detected = True
                    break  
            
            # if len(observed_detection_times)>=1: #use this if you want to test just one detection
            #     detected = True 
            #     break    
                
        return detected

#shar removed betterdetect cause we don't use it
    

# --------------------------------------------
# Unified Detection metric
# --------------------------------------------
class Detect_Metric(Base_Metric):
    """ 

    Option A: ≥2 detections in a single filter, ≥30 minutes apart
    
    Option B: ≥2 epochs, second has ≥2 filters; first can be a non-detection
    
    This is an “either/or” detection logic. 
    
    This event is detected if it passes either the intra-night multi-detection or the epoch-based detection criteria.
    
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
            'sid_duplicate': slice_point['sid'],
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
# Characterization metric — extended multi-band follow-up
# --------------------------------------------
class GRBAfterglowCharacterizeMetric(Base_Metric):
    """
    Characterization metric for GRB Afterglows.

    This metric tests whether the transient can be sufficiently characterized for follow-up
    science goals. An event is considered 'characterized' if it meets two criteria:
    
    (1) At least 4 observations with signal-to-noise ratio (SNR) ≥ 3.
    (2) Among those detections, the observations span at least 3 different filters 
        and cover a duration of at least 3 days.

    These thresholds are motivated by the need to capture the transient's color evolution 
    and fading behavior across multiple bands and epochs, which are key for identifying
    and classifying GRB afterglows compared to other fast-evolving transients.
    
    This design ensures that events classified as 'characterized' have sufficient
    multi-band and temporal information to allow basic modeling and comparison to 
    theoretical GRB afterglow light curves.
    """
    def __init__(self, **kwargs):
        use_extinction = kwargs.pop('use_extinction', True)
        super().__init__(**kwargs, use_extinction=use_extinction)
        self.metricName = kwargs.get('metricName', 'GRB_Characterize')
        self.obs_records = {}  # <-- NEW: to store all detected event records individually
        self.parent_instance = Base_Metric(use_extinction=use_extinction)
        
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)
        detected = self.parent_instance.detect(filters, snr, times, obs_record)
        if detected:
            good = snr >= 3
            if np.sum(good) < 4:
                return 0.0
            n_filters = len(np.unique(filters[good]))
            duration = np.ptp(times[good])
            if n_filters >= 3 and duration >= 3:
                return 1.0
        return 0.0

# --------------------------------------------
# Spectroscopic Triggerability Metric
# Detects if ≥2 filters are triggered within 0.5 days of peak
# --------------------------------------------
class GRBAfterglowSpecTriggerableMetric(Base_Metric):
    """
    Spectroscopic triggerability metric for GRB Afterglows.

    This metric evaluates whether a GRB afterglow would be suitable for rapid spectroscopic follow-up.
    An event is considered triggerable if:
    (0.5) it is detected
    (1) At least one filter shows brightness < 21 mag,
    (2) It rises faster than 0.3 mag/day in that filter,
    (3) Both detections used to assess this have SNR >= 5.
    """
    def __init__(self, **kwargs):
        use_extinction = kwargs.pop('use_extinction', True)
        super().__init__(load_from="GRBAfterglow_templates.pkl", **kwargs, use_extinction=use_extinction)
        self.metricName = kwargs.get('metricName', 'GRB_SpecTrigger')
        self.parent_instance = Base_Metric(use_extinction=use_extinction)

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)
        
        if obs_record is None or len(obs_record['mjd_obs']) < 2:
            return 0.0
        detected = self.parent_instance.detect(filters, snr, times, obs_record)
        if detected!=True:
            return 0.0
        # Sort by time
        sorted_idx = np.argsort(obs_record['mjd_obs'])
        for key in obs_record:
            if isinstance(obs_record[key], np.ndarray):
                obs_record[key] = obs_record[key][sorted_idx]

        mjd = obs_record['mjd_obs']
        mags = obs_record['mag_obs']
        snrs = obs_record['snr_obs']
        filts = obs_record['filter']

        for f in np.unique(filts):
            f_mask = (filts == f)
            if np.sum(f_mask) < 2:
                continue

            good = f_mask & (snrs >= 5)
            if np.sum(good) < 2:
                continue

            t = mjd[good]
            m = mags[good]

            # Check rise rate
            delta_mag = np.diff(m)
            delta_time = np.diff(t)
            rise_rate = delta_mag / delta_time  # Positive = fading, Negative = rising

            #if np.any(rise_rate < -0.3) and np.any(m < 21): #we don't have rise rates atm
            #if np.any(np.abs(rise_rate) > 0.3) and np.any(m < 21): # triggers if any rapid brightness change (fading or rising) AND magnitude is bright enough
            if np.any(m < 21): #assume rise rate is always that fast and detected.  

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
        'detect': Detect_Metric(lc_model=lc_model, use_extinction=use_extinction),
        'characterize': GRBAfterglowCharacterizeMetric(lc_model=lc_model, use_extinction=use_extinction),
        'spec_trigger': GRBAfterglowSpecTriggerableMetric(lc_model=lc_model, use_extinction=use_extinction),
    }

    if include is None:
        return list(all_metrics.values())
    else:
        return [all_metrics[name] for name in include if name in all_metrics]











    
