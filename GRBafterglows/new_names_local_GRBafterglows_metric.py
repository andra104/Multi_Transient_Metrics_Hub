from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues

import sys
import os
sys.path.append(os.path.abspath(".."))
from shared_utils import equatorialFromGalactic, uniform_sphere_degrees, inject_uniform_healpix

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

    
# --------------------------------------------
# Power-law GRB afterglow model based on Zeh et al. (2005)
# --------------------------------------------
class LC:
    """
    Simulate GRB afterglow light curves using a power-law model.

    Light curves follow:
        m(t) = m_0 + 2.5 * alpha * log10(t/t_0)
    where alpha is the temporal slope (decay), t is time (days),
    and m_0 is the peak magnitude (from Zeh et al. 2005).

    The light curve begins at peak magnitude, and the decay is positive (fading).
    """
    def __init__(self, num_samples=100, num_lightcurves=1000, load_from=None):
        """
        Parameters
        ----------
        num_samples : int
            Number of time points to sample in the light curve (log-uniformly spaced).
        load_from : str or None
            If provided and valid, loads light curve templates from a pickle file.
        """
        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                data = pickle.load(f)
            self.data = data['lightcurves']
            self.filts = list(self.data[0].keys())
            print(f"Loaded GRB afterglow templates from {load_from}")
            return

        self.data = []
        self.filts = ["u", "g", "r", "i", "z", "y"]
        self.t_grid = np.logspace(-2, 2, num_samples) # 0.1 to 100 days
        #fbb adjusted to .01 because .1 day is actually significantly after peak already

        # decay_slope_range = (0.5, 2.5) #not using this anymore
        peak_mag_range = (-24, -22)

        rng = np.random.default_rng(42)
        for _ in range(num_lightcurves):
            lc = {}
            for f in self.filts:
                m0 = rng.uniform(*peak_mag_range)
                #alpha_fade = rng.uniform(*decay_slope_range)
                
                # Parameters of truncted normal: mean = 1.3, std = 0.3, range = [0.5, 2.5]
                # a, b = (0.5 - 1.3) / 0.3, (2.5 - 1.3) / 0.3
                # trunc_alpha = truncnorm(a=a, b=b, loc=1.3, scale=0.3)

                #shar - this section works okay
                # mean = 4, std = 1, range = [2.5, 5]
                # a, b = (2.5 - 4) / .5, (5 - 4) / .5
                # trunc_alpha = truncnorm(a=a, b=b, loc=4, scale=.5)

                t_jetbreak = np.logspace(0.01, 0.7, 100) #jet break timing (3 times faster decay) between 1 and 5 days
                
                jetbreak = np.random.choice(t_jetbreak) # FBB added May 21
                
                #shar numbers based off Zeh 2005 (different Zeh 2005 though)
                a, b = (.5 - 1.5) / .5, (1.7 - 1.5) / .5
                trunc_alpha = truncnorm(a=a, b=b, loc=1.5, scale=.5)
                
                alpha_fade = trunc_alpha.rvs(random_state=rng)

                adjusted_m0 = m0 - 2.5 * alpha_fade * np.log10(self.t_grid )[0]
                #shar adjusting so that the first mag is in our peak range
                mag = adjusted_m0 + 2.5 * alpha_fade * np.log10(self.t_grid )

                 # FBB now replace tail with post-jet break
                mask = self.t_grid > jetbreak #only dates after the jet break
                _ = [True if mask[i+1] else False for i,m in enumerate(mask[:-1])] 
                mask[:-1] = _

                new_decay = 10 * (np.log10(self.t_grid[mask] + 1)) 
                if mask[0] == True:
                    mask[0] = False
                mag[mask] =  new_decay - new_decay[0] + mag[~mask][-1]

                lc[f] = {'ph': self.t_grid, 'mag': mag}
            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        """
        Interpolate the light curve for the given filter and index at times `t`.

        Parameters
        ----------
        t : array_like
            Times relative to peak (days).
        filtername : str
            LSST filter name (u, g, r, i, z, y).
        lc_indx : int
            Index of the light curve in the template set.

        Returns
        -------
        magnitudes : array_like
            Interpolated magnitudes, clipped at 99 for out-of-range.
        """
        if lc_indx >= len(self.data):
            print(f"Warning: lc_indx {lc_indx} out of bounds, using last template.")
            lc_indx = len(self.data) - 1
        return np.interp(t,
                         self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)

# --------------------------------------------------
# Light Curve Template Generator (Separate from Population)
# --------------------------------------------------
def generate_Templates(
    num_samples=100, num_lightcurves=1000,
    save_to="GRBAfterglow_templates.pkl"
):
    """
    Generate synthetic GRB afterglow light curve templates and save to file.
    """
    # if os.path.exists(save_to):
    #     print(f"Found existing GRB afterglow templates at {save_to}. Not regenerating.")
    #     return
    #shar - i want this to be a keyword argument not a default

    lc_model = LC(num_samples=num_samples, num_lightcurves=num_lightcurves,
                              load_from=None)
    with open(save_to, "wb") as f:
        pickle.dump({'lightcurves': lc_model.data}, f)
    print(f"Saved synthetic GRB light curve templates to {save_to}")

# --------------------------------------------
# Light Curve Loader (used in scripts)
# --------------------------------------------
def load_or_generate_templates(templates_file="GRBAfterglow_templates.pkl",
                               num_samples=100, num_lightcurves=1000,
                               generate_new=False):
    """
    Load GRB light curve templates from a file, or generate and save new ones.

    Parameters
    ----------
    templates_file : str
        Path to the .pkl file containing light curves.
    num_samples : int
        Number of time samples in each light curve.
    num_lightcurves : int
        Number of unique light curve templates to simulate.
    generate_new : bool
        Whether to generate and save new templates.

    Returns
    -------
    LC instance
        The loaded or newly generated light curve model.
    """
    if generate_new or not os.path.exists(templates_file):
        print(f"[INFO] Generating {num_lightcurves} light curve templates.")
        generate_Templates(num_samples=num_samples,
                           num_lightcurves=num_lightcurves,
                           save_to=templates_file)
    else:
        print(f"[INFO] Loading light curve templates from {templates_file}.")
    return LC(load_from=templates_file)


# --------------------------------------------
# Base GRB Metric with extinction and SNR
# --------------------------------------------
class Base_Metric(BaseMetric):
    def __init__(self, metricName='BaseGRBAfterglowMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=59853.5, outputLc=False, badval=-666,
                 filter_include=None,
                 load_from="GRBAfterglow_templates.pkl",
                 lc_model=None,  # <-- NEW
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

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName,
                         units='Detection Efficiency',
                         badval=badval, **kwargs)



    
    def evaluate_grb(self, dataSlice, slice_point, return_full_obs=True):
        """
        Evaluate GRB light curve at the location and time of the slice point.
        Apply extinction, distance modulus, and optional filter inclusion.
        """
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros(t.size)
    
        for f in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == f)
            mags[infilt] = self.lc_model.interp(t[infilt], f, slice_point['file_indx'])
            mags[infilt] += self.ax1[f] * slice_point['ebv']
            mags[infilt] += 5 * np.log10(slice_point['distance'] * 1e6) - 5
    
        snr = m52snr(mags, dataSlice[self.m5Col])
        filters = dataSlice[self.filterCol]
        times = t
    
        if return_full_obs:
            obs_record = {
                'mjd_obs': dataSlice[self.mjdCol],
                'mag_obs': mags,
                'snr_obs': snr,
                'filter': filters,
                # NO 'detected' YET -- will be set later if detected!
            }
            
            return snr, filters, times, obs_record
        return snr, filters, times, None

    def detect(self, filters, snr, times, obs_record):
        detected = False        
        # -------- Detection Logic --------
        # Option A: 2 detections in same filter ≥30min apart
        for f in np.unique(filters):
            mask = filters == f
            if np.sum(snr[mask] >= 5) >= 2:        
                if np.ptp(times[mask]) >= 0.5 / 24 and np.diff(np.sort(times[mask])).min() <= MAXGAP :
                    detected = True
                    break
        
        return detected

    def betterdetect(self, filters, snr, times, obs_record):
        

        mask = snr >= 5
        t_detect = times[snr >= 5]
        detected = False
        if len(t_detect) > 2: # more than 2 detections
            if len(np.unique(filters[mask])) >= 2 : #more than 2 filters
                day1 = (times >= times[mask].min() + 2/24) * (times <= times[mask].min() + 1)
                
                
        
                if np.ptp(t_detect) >= 0.5 / 24 : #two detections in >30 min
                    if len(times[day1]) > 2: #three detections in 1 night
                        #np.diff(np.sort(times[day1])).max() <= MAXGAP: 
                        #not the right logic: 
                        #print(times[day1])
                        detected = True
        # Option B: ≥2 epochs, second has ≥2 filters; first can be a non-detection
        return detected
    

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
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)
    
        if obs_record is None:
            return self.badval
    
        if self.filter_include is not None:
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
            'peak_mjd': peak_mjd,
            'peak_mag': peak_mag,
            'ebv': slice_point['ebv'],
            'peak_time': slice_point['peak_time'],
            'detected': bool(detected),
            'mjd_obs': obs_record.get('mjd_obs', np.array([])),
            'mag_obs': obs_record.get('mag_obs', np.array([])),
            'snr_obs': obs_record.get('snr_obs', np.array([])),
            'filter': obs_record.get('filter', np.array([]))
        })    

        self.obs_records[slice_point['sid']] = obs_record
        self.latest_obs_record = obs_record if detected else None
    
        return 1.0 if detected else 0.0


class GRBAfterglowBetterDetectMetric(Base_Metric):
    """ 

    Option A: ≥2 detections in a single filter, ≥30 minutes apart
    
    Option B: ≥2 epochs, second has ≥2 filters; first can be a non-detection
    
    This is an “either/or” detection logic. 
    
    This event is detected if it passes either the intra-night multi-detection or the epoch-based detection criteria.
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'GRB_BetterDetect')
        self.obs_records = {}  # <-- NEW: to store all detected event records individually
        self.parent_instance = Base_Metric()

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)
        
        if self.filter_include is not None:
            keep = np.isin(filters, self.filter_include)
            snr = snr[keep]
            filters = filters[keep]
            times = times[keep]
            for k in ['mjd_obs', 'mag_obs', 'snr_obs', 'filter']:
                if isinstance(obs_record[k], np.ndarray):
                    obs_record[k] = obs_record[k][keep]

        # -------- Detection Logic --------
        
        detected = False
    
        # Option A: 2 detections in same filter ≥30min apart

        #detected = self.parent_instance.detect(filters, snr, times, obs_record)
        #if not detected:
        detected = self.parent_instance.betterdetect(filters, snr, times, obs_record)
        
        
        # -------- Save Detection Metadata --------
    
        if detected:
            
            detected_mask = snr >= 5 # FBB didnt you have this cut earlier too? why doubling it?
            obs_record['detected'] = bool(np.any(detected_mask))
            #self.latest_obs_record = obs_record

            # Calculate and fade times
            first_det_mjd = np.nan
            last_det_mjd = np.nan
            fade_time = np.nan
        
            if np.any(detected_mask):
                first_det_mjd = obs_record['mjd_obs'][detected_mask].min()
                last_det_mjd = obs_record['mjd_obs'][detected_mask].max()
                fade_time = last_det_mjd - (self.mjd0 + slice_point['peak_time'])
        
            peak_index = np.argmin(obs_record['mag_obs'])
            peak_mjd = obs_record['mjd_obs'][peak_index]
            peak_mag = obs_record['mag_obs'][peak_index]

            # Update obs_record with full metadata

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
                'peak_mjd': peak_mjd,
                'peak_mag': peak_mag,
                'ebv': slice_point['ebv'],
                'peak_time': slice_point['peak_time'],
                'detected': bool(detected),
                'mjd_obs': obs_record.get('mjd_obs', np.array([])),
                'mag_obs': obs_record.get('mag_obs', np.array([])),
                'snr_obs': obs_record.get('snr_obs', np.array([])),
                'filter': obs_record.get('filter', np.array([]))
            })    
        
            # Save this full event

            self.obs_records[slice_point['sid']] = obs_record
            self.latest_obs_record = obs_record if detected else None
            return 1.0
        else:
            self.latest_obs_record = None
            return 0.0


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
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'GRB_Characterize')
        self.obs_records = {}  # <-- NEW: to store all detected event records individually
        self.parent_instance = Base_Metric()
        
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)
        detected = self.parent_instance.detect(filters, snr, times, obs_record)
        if not detected:
            detected = self.parent_instance.betterdetect(filters, snr, times, obs_record)
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

    (1) At least one filter shows brightness < 21 mag,
    (2) It rises faster than 0.3 mag/day in that filter,
    (3) Both detections used to assess this have SNR >= 5.
    """
    def __init__(self, **kwargs):
        super().__init__(load_from="GRBAfterglow_templates.pkl", **kwargs)
        self.metricName = kwargs.get('metricName', 'GRB_SpecTrigger')
        self.parent_instance = Base_Metric()

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)
        if obs_record is None or len(obs_record['mjd_obs']) < 2:
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

            #if np.any(rise_rate < -0.3) and np.any(m < 21): #we don't have fade rates atm
            if np.any(np.abs(rise_rate) > 0.3) and np.any(m < 21): # triggers if any rapid brightness change (fading or rising) AND magnitude is bright enough

                return 1.0

        return 0.0

# --------------------------------------------
# Color Evolution Metric
# Detects ≥2 epochs with multi-color information to constrain synchrotron cooling
# --------------------------------------------
class GRBAfterglowColorEvolveMetric(Base_Metric):
    """
    Color evolution detection metric for GRB Afterglows.

    This metric assesses whether an event shows measurable color (spectral) evolution over time.
    An event satisfies the criterion if:

    (1) At least 4 observations are detected with SNR ≥ 3,
    (2) These observations cluster into ≥ 2 distinct epochs (grouped at 0.5-day resolution),
    (3) Each epoch includes detections in at least 2 different filters.

    The detection of color evolution is critical for constraining synchrotron cooling breaks,
    energy injection episodes, and jet structure in GRB afterglows. These requirements
    are based on observational constraints for detecting chromatic breaks described in Zeh et al. (2005)
    and adapted to Rubin's cadence characteristics.

    By requiring multi-color detections across epochs, this metric distinguishes genuine 
    evolving afterglows from static or non-evolving fast transients.
    """
    def __init__(self, **kwargs):
        super().__init__(load_from="GRBAfterglow_templates.pkl", **kwargs)
        self.metricName = kwargs.get('metricName', 'GRB_ColorEvolution')
        self.obs_records = {}  # <-- NEW: to store all detected event records individually

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)

        detected = (snr >= 3)
        if np.sum(detected) < 4:
            return 0.0
        
        # Group by rounded times to cluster into epochs
        t_epoch = np.round(times[detected] * 2) / 2  # bin to 0.5-day resolution
        f_epoch = filters[detected]

        epoch_colors = {}
        for t, f in zip(t_epoch, f_epoch):
            epoch_colors.setdefault(t, set()).add(f)

        multi_color_epochs = [e for e in epoch_colors.values() if len(e) >= 2]

        if len(multi_color_epochs) >= 2:
            return 1.0
        return 0.0

# --------------------------------------------
# Historical Non-Detection Match Metric
# Checks whether the transient would stand out against deep coadds
# --------------------------------------------
class GRBAfterglowHistoricalMatchMetric(Base_Metric):
    """
    Archival non-detection match metric for GRB Afterglows.

    This metric checks whether the transient would stand out as a new source compared
    to deep archival imaging. An event passes if:

    (1) Any portion of its light curve is brighter than the assumed archival coadd depth 
        (default = 27.0 magnitudes).

    This logic is based on the expectation that GRB afterglows have no persistent
    optical counterparts prior to the event. The archival depth value reflects Rubin’s
    expected Wide-Fast-Deep coadded survey depth.

    This metric was designed to filter out background variable sources such as AGNs,
    variable stars, or other contaminants that could otherwise mimic a GRB-like transient
    in photometric detection pipelines.
    """
    def __init__(self, coaddDepth=27.0, **kwargs):
        """
        Parameters
        ----------
        coaddDepth : float
            Simulated archival limiting magnitude.
        """
        self.coaddDepth = coaddDepth
        super().__init__(load_from="GRBAfterglow_templates.pkl", **kwargs)
        self.metricName = kwargs.get('metricName', 'GRB_HistoricalMetric')
        self.obs_records = {}  # <-- NEW: to store all detected event records individually

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)
        # Check if any detection is brighter than the archival depth
        mags = np.zeros(times.size)
        for f in np.unique(filters):
            mask = filters == f
            mags[mask] = self.lc_model.interp(times[mask], f, slice_point['file_indx'])
            mags[mask] += self.ax1[f] * slice_point['ebv']
            mags[mask] += 5 * np.log10(slice_point['distance'] * 1e6) - 5

        if np.any(mags < self.coaddDepth):
            return 1.0  # Would stand out above archival image
        return 0.0

# --------------------------------------------
# Multi_Metric Standardized Call
# --------------------------------------------
def get_multi_metrics(lc_model, include=None):
    """
    Return a list of metrics. `include` can be a list of metric names to include.
    """
    all_metrics = {
        'detect': Detect_Metric(lc_model=lc_model),
        'better_detect': GRBAfterglowBetterDetectMetric(lc_model=lc_model),
        'characterize': GRBAfterglowCharacterizeMetric(lc_model=lc_model),
        'spec_trigger': GRBAfterglowSpecTriggerableMetric(lc_model=lc_model),
        'color_evolve': GRBAfterglowColorEvolveMetric(lc_model=lc_model),
        'historical': GRBAfterglowHistoricalMatchMetric(lc_model=lc_model)
    }

    if include is None:
        return list(all_metrics.values())
    else:
        return [all_metrics[name] for name in include if name in all_metrics]


# --------------------------------------------
# GRB volumetric rate model (on-axis ≈ 10⁻⁹ Mpc⁻³ yr⁻¹)
# --------------------------------------------
def sample_grb_rate_from_volume(t_start, t_end, d_min, d_max, rate_density=1e-8): #1e-8 to account for dirty fireball and off axis, 1e-9 without
    """
    Estimate the number of GRBs from comoving volume and volumetric rate.

    Parameters
    ----------
    t_start : float
        Start of the time window (days).
    t_end : float
        End of the time window (days).
    d_min : float
        Minimum luminosity distance in Mpc.
    d_max : float
        Maximum luminosity distance in Mpc.
    rate_density : float
        Volumetric GRB rate in events/Mpc^3/yr.

    Returns
    -------
    int
        Expected number of GRBs in the survey.
    """
    years = (t_end - t_start) / 365.25
    z_min = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
    z_max = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)

    V = cosmo.comoving_volume(z_max).to(u.Mpc**3).value - cosmo.comoving_volume(z_min).to(u.Mpc**3).value
    return np.random.poisson(rate_density * V * years)

# --------------------------------------------
# Population Loader (used in scripts)
# --------------------------------------------
def load_or_generate_population(t_start=1, t_end=3652, seed=42,
                                d_min=10, d_max=1000,
                                num_lightcurves=1000,
                                gal_lat_cut=None, rate_density=1e-8,
                                pop_file="GRB_population_fixedpop.pkl",
                                generate_new=False,
                                make_debug_plots=False):
    """
    Load GRB population from a saved file or generate a new one.

    Parameters
    ----------
    t_start : float
        Start time in days since survey start.
    t_end : float
        End time in days since survey start.
    seed : int
        RNG seed.
    d_min, d_max : float
        Minimum and maximum luminosity distances (Mpc).
    num_lightcurves : int
        Number of templates available.
    gal_lat_cut : float or None
        Optional minimum Galactic latitude (deg).
    rate_density : float
        Volumetric rate in Mpc⁻³ yr⁻¹.
    pop_file : str
        Path to save or load population.
    generate_new : bool
        If True, regenerate population and overwrite.
    make_debug_plots : bool
        If True, show debug histograms.

    Returns
    -------
    UserPointsSlicer
        Slicer with populated slice_points metadata.
    """
    if generate_new or not os.path.exists(pop_file):
        print(f"[INFO] Generating GRB population and saving to {pop_file}")
        slicer = generate_PopSlicer(t_start=t_start, t_end=t_end,
                                    d_min=d_min, d_max=d_max,
                                    seed=seed,
                                    num_lightcurves=num_lightcurves,
                                    gal_lat_cut=gal_lat_cut,
                                    rate_density=rate_density,
                                    save_to=pop_file,
                                    make_debug_plots=make_debug_plots)
    else:
        print(f"[INFO] Loading GRB population from {pop_file}")
        slicer = generate_PopSlicer(load_from=pop_file)

    return slicer

    
# --------------------------------------------
# GRB population generator
# --------------------------------------------
def generate_PopSlicer(t_start=1, t_end=3652, seed=42,
                         d_min=10, d_max=1000, num_lightcurves=1000, gal_lat_cut=None, rate_density=1e-8,
                         load_from=None, save_to=None, make_debug_plots=True):
    """
    Generate a population of GRB afterglows with realistic extinction and sky distribution.

    Parameters
    ----------
    gal_lat_cut : float or None
        Optional Galactic latitude cut (e.g., 15 deg).
    load_from : str or None
        If set, load slice_points from this pickle file.
    save_to : str or None
        If set, save the slice_points to this pickle file.
    make_debug_plots : True or anything else
        if true, will plot various distributions and print some stuff
    """
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        print(f"Loaded GRB population from {load_from}")
        return slicer

    rng = np.random.default_rng(seed)
    n_events = sample_grb_rate_from_volume(t_start, t_end, d_min, d_max, rate_density=rate_density)
    print(f"Simulated {n_events} GRB events using rate_density = {rate_density:.1e}")

    
    #ra, dec = uniform_sphere_degrees(n_events, seed=seed) #returns degrees
    nside = 64  # Or 128 if you want higher resolution
    ra, dec = inject_uniform_healpix(nside=nside, n_events=n_events, seed=seed)

    #print(f"[CHECK] Dec range: {dec.min():.2f} to {dec.max():.2f} (expected ~[-90, 90])")

    dec = np.clip(dec, -89.9999, 89.9999)
    #dec_rad = np.radians(dec)
    
    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0) #returns radians 
    #print(f"Print 10 = {ra[:10],dec[:10]}")
    #print(f" Value = {slicer.slice_points}")
    #slicer.slice_points['ra'] = ra
    #slicer.slice_points['dec'] = dec_rad  # Correct assignment
    if make_debug_plots==True:
        plt.hist(slicer.slice_points['ra'], bins=50)
        plt.xlabel("RA [rad]")
        plt.title("Injected GRB Population – RA Distribution")
        plt.grid(True)
        plt.show()
        
        plt.hist(slicer.slice_points['dec'], bins=50)
        plt.xlabel("Dec [rad]")
        plt.title("Injected GRB Population – Dec Distribution")
        plt.grid(True)
        plt.show()


    distances = rng.uniform(d_min, d_max, n_events)
    peak_times = rng.uniform(t_start, t_end, n_events)
    file_indx = rng.integers(0, num_lightcurves, len(ra))

    #print(t_start, t_end, n_events)
    if make_debug_plots==True:  
        plt.hist(peak_times,  bins=50)
        plt.xlabel("peak time")
        plt.title("Peak Time")
        plt.grid(True)
        plt.show()
    
        plt.hist(distances,  bins=50)
        plt.xlabel("distance")
        plt.title("Distance Distribution")
        plt.grid(True)
        plt.show()


    
    #print(f"[DEBUG] dec sample before SkyCoord: {dec[:5]}")
    #print(f"[DEBUG] dec units? min={np.min(dec):.2f}, max={np.max(dec):.2f}")
    
        #print(f"[DEBUG]Print 5 sample before SkyCoord - ra,dec: {slicer.slice_points}")
        print("[DEBUG 7]: Do you see me")


    #coords = SkyCoord(ra=slicer.slice_points['ra'] * u.deg, dec=slicer.slice_points['dec'] * u.deg, frame='icrs') - this code just labels them as deg. u.deg doesn't convert them. 

    coords = SkyCoord(ra=np.degrees(slicer.slice_points['ra']) * u.deg, dec=np.degrees(slicer.slice_points['dec']) * u.deg, frame='icrs') #this line correctly converts them and labels them
    if make_debug_plots==True:     
        print(f"[DEBUG] coords.dec[:5]: {coords.dec[:5]}")
        print(f"[DEBUG] coords.dec.unit: {coords.dec.unit}")

        plt.hist(coords.ra, bins=50)
        plt.xlabel("RA [deg]")
        plt.title("SkyCoord RA Distribution")
        plt.grid(True)
        plt.show()
        
        plt.hist(coords.dec, bins=50)
        plt.xlabel("Dec [deg]")
        plt.title("SkyCoord Dec Distribution")
        plt.grid(True)
        plt.show()

    sfd = SFDQuery()
    ebv_vals = sfd(coords)

    if gal_lat_cut is not None:
        b = coords.galactic.b.deg
        mask = np.abs(b) > gal_lat_cut
        ra, dec = ra[mask], dec[mask]
        distances = distances[mask]
        peak_times = peak_times[mask]
        file_indx = file_indx[mask]
        ebv_vals = ebv_vals[mask]
        coords = coords[mask]


    

    #slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)
    #slicer.slice_points['ra'] = ra
    #slicer.slice_points['dec'] = dec
    slicer.slice_points['distance'] = distances
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    slicer.slice_points['gall'] = coords.galactic.l.deg
    slicer.slice_points['galb'] = coords.galactic.b.deg

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)
        print(f"Saved GRB population to {save_to}")

    return slicer

# --------------------------------------------
# Standardized GRB storage paths (used in scripts)
# --------------------------------------------
def get_output_paths(case_label="GRBafterglows"):
    """
    Generate standardized output filenames and directory paths for this science case.

    Parameters
    ----------
    case_label : str
        Short name for this science case (used to define subfolder).
        Examples: 'GRBafterglows', 'KNe', 'LFBOTs', etc.

    Returns
    -------
    dict
        Dictionary with standardized paths:
            - 'case_label'
            - 'storage_dir'
            - 'templates_file'
            - 'pop_file'
    """
    # Force base_dir to be .../Multi_Transient_Metrics_Hub/output
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

    storage_dir = os.path.join(base_dir, case_label)
    templates_file = os.path.join(storage_dir, f"{case_label}_templates.pkl")
    pop_file = os.path.join(storage_dir, f"{case_label}_population.pkl")

    os.makedirs(storage_dir, exist_ok=True)
    return {
        'case_label': case_label,
        'storage_dir': storage_dir,
        'templates_file': templates_file,
        'pop_file': pop_file
    }



    
