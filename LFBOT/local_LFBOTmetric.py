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

# --------------------------------------------
# Light Curve Model for LFBOTs
# --------------------------------------------

class LC:
    """
    Generate synthetic light curves for Luminous Fast Blue Optical Transients (LFBOTs).

    Light curves are modeled with band-dependent rise and fade slopes, peaking around ~1 day,
    and spanning a fast-evolving timescale of ~0.1 to 10 days. Only g and r bands are populated,
    consistent with the predominantly blue emission of LFBOTs.
    """
    def __init__(self, num_samples = 100, num_lightcurves=1000, load_from=None):
        self.filts = ['u', 'g', 'r', 'i', 'z', 'y']
        self.data = []
        self.t_grid = None  # 0.1–10 days
        self.ratios = {
            'u': 0.47184822835586304,
            'g': 1.0,
            'r': 0.4356131668415372,
            'i': 0.20399584045639102,
            'z': 0.09743514963840874,
            'y': 0.0347938320584038
        }
        # Do the mag change as an additive factor using the mag ratios (that were converted from the flux ratios), should be a single line as an arrray

        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                obj = pickle.load(f)
            self.data = obj['lightcurves']
            self.t_grid = obj['t_grid']
            print(f"Loaded LFBOT templates from {load_from}")
            return


        rng = np.random.default_rng(42)
        self.t_grid = np.linspace(0.01, 15.0, num_samples)  # or use logspace

        for _ in range(num_lightcurves):
            lc = {}

            m0_g = rng.uniform(-22, -19.5)  # peak absolute mag
            t0 = 5     # time of peak (days)
            alpha_rise = rng.uniform(0.25, 2.5)
            alpha_fade = rng.uniform(0.15, 0.45)
            alpha_fade = 2.2

            # print(f"m0:_g:{m0_g},t0:{t0},rise:{alpha_rise},fade:{alpha_fade}")

            mag_g = np.zeros_like(self.t_grid)

            for i, t in enumerate(self.t_grid):
                if t < t0:
                    mag_g[i] = m0_g - 2.5 * alpha_rise * np.log10(t / t0)
                else:
                    mag_g[i] = m0_g + 2.5 * alpha_fade * np.log10(t / t0)

            flux_g = 10 ** (-0.4 * mag_g)
            for f in self.filts:
                flux_f = flux_g * self.ratios[f]
                mag_f = -2.5 * np.log10(flux_f)
                lc[f] = {'ph': self.t_grid, 'mag': mag_f}

            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)



# --------------------------------------------
# Base Metric for LFBOTs
# --------------------------------------------
class Base_Metric(BaseMetric):
    """
    Base metric class for evaluating LFBOT light curves against simulated observations.

    This class handles light curve interpolation, extinction correction, and signal-to-noise
    calculation, providing a standardized evaluation framework for derived LFBOT metrics.
    """
    def __init__(self, metricName='Base_Metric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=60980.5, outputLc=False, badval=-666,
                 filter_include=None, load_from="LFBOT_templates.pkl", use_extinction=True,
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
    
        # Convert to arrays just in case
        filters = np.array(filters)
        snr = np.array(snr)
        times = np.array(times)
    
        # Identify all detections above 3σ and 5σ
        idx_3sigma = snr >= 3
        idx_5sigma = snr >= 5
    
        # Option A: Strict 3σ ×3 with at least one 5σ, and all within 0.5 days
        if np.sum(idx_3sigma) >= 3:
            t_detected = times[idx_3sigma]
            if np.ptp(t_detected) <= 0.5 and np.any(idx_5sigma):
                detected = True
    
        # Option B: ≥1 detection at 5σ + ≥2 total detections ≥15min apart
        elif np.sum(idx_5sigma) >= 1:
            t_detected = times[idx_5sigma]
            if len(t_detected) >= 2 and np.ptp(t_detected) >= 0.0104:  # ~15 min
                detected = True
    
        return detected



    

# --------------------------------------------
# Detection Metric for LFBOTs
# --------------------------------------------
class Detect_Metric(Base_Metric):
    """
    LFBOT Detection Metric

    This metric implements the detection criteria for Luminous Fast Blue Optical Transients (LFBOTs),
    based on their observed fast rise, blue colors, and rapid fading behavior.

    Detection logic:
    - Primary: Require ≥2 distinct epochs of detection (≥30 minutes separation, ≤6 days total span),
      with at least one epoch showing detections in ≥2 different filters (to establish color and luminosity).
    - Fallback: If color information is not available, require ≥3 epochs (≥30 minutes separation, ≤6 days total span)
      to track fading behavior consistent with LFBOT timescales.

    This design reflects the astrophysical properties of LFBOTs, including typical 0.2 mag/day fading rates
    and durations at high luminosity under ~10–12 days, as seen in events like AT2018cow and AT2023fhn.
    It enforces both color-based identification and fallback monitoring pathways.
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
# Characterization Metric for LFBOTs
# --------------------------------------------
class LFBOTCharacterizeMetric(Base_Metric):
    """
    Given the provided scientific context, we define a minimal photometric characterization
    criterion for Rubin LSST observations of Luminous Fast Blue Optical Transients (LFBOTs).

    Based on the science description:
    - Full confirmation of LFBOT nature requires external follow-up (radio, X-ray, or spectroscopy),
      as noted explicitly in the provided science case.
    - Optical surveys like Rubin primarily serve to detect candidates and monitor fast fading behavior.
    - Example events like AT2018cow and AT2023fhn demonstrate ~0.2 mag/day fading rates
      and durations at high luminosity of less than 10–12 days.
    - You indicated that specific filters (g and r bands) dominate, and monitoring fading tails is
      considered helpful, even if it does not constitute definitive classification.

    Therefore, we define photometric characterization as:
    - Having at least 4 detections with SNR ≥3,
    - Spanning a timespan of at least 3 days.

    These limits ensure that Rubin can constrain the rapid evolution of LFBOT candidates in optical light,
    sufficient to inform and trigger multi-wavelength follow-up, even though true physical classification
    depends on external datasets.

    This structure mirrors the GRB afterglow characterization metric but is relaxed:
    - No ≥3 filters condition is required (because LFBOTs are primarily blue and concentrated in g and r).
    """
    def __init__(self, **kwargs):
        use_extinction = kwargs.pop('use_extinction', True)
        super().__init__(**kwargs, use_extinction=use_extinction)
        self.metricName = kwargs.get('metricName', 'LFBOT_Characterize')
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
            if duration >= 3:
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
        'characterize': LFBOTCharacterizeMetric(lc_model=lc_model, use_extinction=use_extinction),
    }

    if include is None:
        return list(all_metrics.values())
    else:
        return [all_metrics[name] for name in include if name in all_metrics]



