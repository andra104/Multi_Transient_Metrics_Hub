from rubin_sim.maf.metrics import BaseMetric

#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues

import sys
import os
sys.path.append(os.path.abspath(".."))
from shared_utils import equatorialFromGalactic, uniform_sphere_degrees, inject_uniform_healpix, apply_spectral_index, evaluate, compare_flux_diff_to_error

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
            alpha_rise = rng.uniform(0.5, 2.5)
            alpha_fade = rng.uniform(0.5, 1)
            # alpha_fade = 2.2

            # print(f"m0:_g:{m0_g},t0:{t0},rise:{alpha_rise},fade:{alpha_fade}")

            mag_g = np.zeros_like(self.t_grid)

            for i, t in enumerate(self.t_grid):
                if t < t0:
                    mag_g[i] = m0_g - 2.5 * alpha_rise * np.log10(t / t0)
                    # mag_g[i] = m0_g - (np.exp(t)-np.exp(t0))
                else:
                    mag_g[i] = m0_g + 2.5 * alpha_fade * np.log10(t / t0)

            flux_g = 10 ** (-0.4 * mag_g)
            for f in self.filts:
                flux_f = flux_g * self.ratios[f]
                mag_f = -2.5 * np.log10(flux_f)
                lc[f] = {'ph': self.t_grid, 'mag': mag_f}

            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        """Linear interpolation over t (supports t<0); NaN outside support."""
        ph  = np.asarray(self.data[lc_indx][filtername]['ph'], dtype=float)
        mag = np.asarray(self.data[lc_indx][filtername]['mag'], dtype=float)
        t_arr = np.asarray(t, dtype=float)
        out = np.interp(t_arr, ph, mag, left=np.nan, right=np.nan)
        out[(t_arr < ph.min()) | (t_arr > ph.max())] = np.nan
        return out



# --------------------------------------------
# Base Metric for LFBOTs
# --------------------------------------------
class Base_Metric(BaseMetric):
    def __init__(self, metricName='BaseLFBOTsMetric', 
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night', mjd0=60980.5, outputLc=False, badval=-666,
                 filter_include=None,
                 load_from="LFBOT_templates.pkl",
                 lc_model=None, use_extinction=True, use_kcorrect=True, k_correct_type=None, k_correct_arg=None,
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
            Shared GRB light curve model object. If None, load from file.
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
        """Two-stage logic tuned for fast, blue LFBOTs with *pre-peak* emphasis.
    
        Primary (Fed):
          PRE-PEAK gate A — color: ≥2 detections with SNR≥10 within 2 hours in *different* filters,
          PRE-PEAK gate B — variability: within any *single* filter pre-peak, at least two SNR≥5 points
                               whose flux change exceeds 3σ (compare_flux_diff_to_error).
        If both A and B pass -> detected. Else -> not detected.
        """
        filters = np.asarray(filters)
        snr = np.asarray(snr, dtype=float)
        times = np.asarray(times, dtype=float)
    
        # times is relative (days since peak) from shared_utils.evaluate
        pre = times < 5 #shar sept remember should be t0 but currently that's 5
        if np.sum(pre) < 2:
            return False
    
        f_pre = filters[pre]
        s_pre = snr[pre]
        t_pre = times[pre]
    
        # Gate A: pre-peak color within 2 hours
        good = s_pre >= 5
        if np.sum(good) >= 2:
            tg, fg = t_pre[good], f_pre[good]
            color_ok = any((0 < abs(tg[j]-tg[i]) < (2/24)) and (fg[i] != fg[j])
                           for i in range(len(tg)) for j in range(i+1, len(tg)))
            if not color_ok:
                return False
        else:
            return False
    
        # Gate B: pre-peak variability significance in any single filter
        for f in np.unique(f_pre):
            mask = (f_pre == f) & (s_pre >= 5)
            if np.sum(mask) >= 2:
                m = np.asarray(obs_record['mag_obs'], float)[pre][mask]
                s = s_pre[mask]
                for i in range(m.size):
                    sig = compare_flux_diff_to_error(m, m[i], s, s[i], return_bool=True)
                    # if np.size(sig) and np.nanmax(np.atleast_1d(sig)) > 3.0:
                    if np.any(sig==True):
                        return True
        return False


    

# --------------------------------------------
# Detection Metric for LFBOTs
# --------------------------------------------
class Detect_Metric(Base_Metric):
    """
    This chart string is outdated: 
    
    LFBOT Detection Metric

    This metric implements the detection criteria for Luminous Fast Blue Optical Transients (LFBOTs),
    based on their observed fast rise, blue colors, and rapid fading behavior.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'Detect')
        self.obs_records = {}  # <-- NEW: to store all detected event records individually
        #self.parent_instance = Base_Metric() #causes regen/load
        
        # 9/11 
        self.parent_instance = None  # don't create another Base_Metric



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
 
        #detected = self.parent_instance.detect(filters, snr, times, obs_record) #initiates again
        detected = Base_Metric.detect(self, filters, snr, times, obs_record) # doesn't create another Base_Metric

    
        detected_mask = snr >= 5
        first_det_mjd = np.nan
        last_det_mjd = np.nan
        #rise_time = np.nan
        fade_time = np.nan

        # 8/25
        # per_filter_min_mag = {}
        # for f in np.unique(obs_record['filter']):
        #     fmask = (obs_record['filter'] == f)
        #     if np.any(fmask):
        #         per_filter_min_mag[f] = float(np.min(obs_record['mag_obs'][fmask]))
        # obs_record['per_filter_min_mag'] = per_filter_min_mag  # small dict
        #8/28
        per_filter_min_mag = {}
        for f in np.unique(obs_record['filter']):
            fmask = (obs_record['filter'] == f)
            if np.any(fmask):
                vals = np.asarray(obs_record['mag_obs'][fmask], dtype=float)
                good = np.isfinite(vals) & (vals < 90)   # drop sentinels if any
                per_filter_min_mag[f] = float(np.nanmin(vals[good])) if np.any(good) else np.nan
        obs_record['per_filter_min_mag'] = per_filter_min_mag

        # --- flux-change significance diagnostics (saved) 9/11 ---
        f_arr = np.asarray(obs_record['filter'])
        m_arr = np.asarray(obs_record['mag_obs'], dtype=float)
        s_arr = np.asarray(obs_record['snr_obs'], dtype=float)
        
        sig_by_f = {}
        any_gt3 = False
        for f in np.unique(f_arr):
            mask = (f_arr == f) & np.isfinite(m_arr) & (s_arr >= 5)
            if np.sum(mask) >= 2:
                m = m_arr[mask]; s = s_arr[mask]
                sig_max = -np.inf
                for i in range(m.size):
                    sig = compare_flux_diff_to_error(m, m[i], s, s[i], return_bool=False)
                    if np.size(sig):
                        sig_max = max(sig_max, float(np.nanmax(np.atleast_1d(sig))))
                sig_by_f[f] = (np.nan if sig_max == -np.inf else sig_max)
                any_gt3 = any_gt3 or (np.isfinite(sig_by_f[f]) and sig_by_f[f] > 3)
            else:
                sig_by_f[f] = np.nan
        
        obs_record['max_flux_change_sigma'] = sig_by_f         # dict per filter
        obs_record['any_flux_change_gt3'] = bool(any_gt3)      # single boolean
    
        if np.any(detected_mask):
            first_det_mjd = obs_record['mjd_obs'][detected_mask].min()
            last_det_mjd = obs_record['mjd_obs'][detected_mask].max()
            #rise_time = first_det_mjd - (self.mjd0 + slice_point['peak_time'])
            fade_time = last_det_mjd - (self.mjd0 + slice_point['peak_time'])
    
        mags = np.asarray(obs_record['mag_obs'], dtype=float)
        mjds = np.asarray(obs_record['mjd_obs'], dtype=float)
        
        finite = np.isfinite(mags) & (mags < 90)  # drop NaNs and 99-style sentinels
        if np.any(finite):
            i_rel = np.nanargmin(mags[finite])     # index within finite subset
            i_abs = np.flatnonzero(finite)[i_rel]  # map back to original index
            peak_mag = float(mags[i_abs])
            peak_mjd = float(mjds[i_abs])
        else:
            peak_mag = np.nan
            peak_mjd = np.nan

    
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
            # 'theta_obs': slice_point['theta_obs'],
            'filter': obs_record.get('filter', np.array([])).tolist(),
            'distance_modulus': slice_point.get('distance_modulus')
        })    

        self.obs_records[slice_point['sid']] = obs_record
        self.latest_obs_record = obs_record if detected else None
        
    
        return 1.0 if detected else 0.0
'''
# --------------------------------------------
# Characterization Metric for LFBOTs
# --------------------------------------------
class LFBOTCharacterizeMetric(Base_Metric):
    """
    Minimal photometric characterization: require >3 SNR≥5 detections spanning ≥3 days,
    with at least one filter exhibiting >3σ flux change across the sequence. No ≥3 filters
    condition (blue SED concentrated in g/r). This stays aligned with fast-evolving nature.
    """
    def __init__(self, **kwargs):
        use_extinction = kwargs.pop('use_extinction', True)
        use_kcorrect = kwargs.pop('use_kcorrect', True)
        super().__init__(**kwargs, use_extinction=use_extinction, use_kcorrect=use_kcorrect)
        self.metricName = kwargs.get('metricName', 'LFBOT_Characterize')
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)

        good = (snr >= 5)
        if np.sum(good) < 4:
            return 0.0
        t = np.asarray(obs_record['mjd_obs'])[good]
        if (np.nanmax(t) - np.nanmin(t)) < 3.0:
            return 0.0

        f = np.asarray(obs_record['filter'])
        m = np.asarray(obs_record['mag_obs'], float)
        s = np.asarray(obs_record['snr_obs'], float)
        for band in np.unique(f):
            mask = (f == band) & good
            if np.sum(mask) >= 2:
                for i in range(np.sum(mask)):
                    sig = compare_flux_diff_to_error(m[mask], m[mask][i], s[mask], s[mask][i], return_bool=False)
                    if np.size(sig) and np.nanmax(np.atleast_1d(sig)) > 3.0:
                        return 1.0
        return 0.0
'''

# --------------------------------------------
# Multi_Metric Standardized Call
# --------------------------------------------

def get_multi_metrics(lc_model, include=None, use_extinction=True, use_kcorrect=True, k_correct_type=None, k_correct_arg=None):
    """Return a list of metrics. `include` can filter by names."""
    all_metrics = {
        'detect': Detect_Metric(lc_model=lc_model, use_extinction=use_extinction, use_kcorrect=use_kcorrect, k_correct_type=k_correct_type, k_correct_arg=k_correct_arg),
        # 'characterize': LFBOTCharacterizeMetric(lc_model=lc_model, use_extinction=use_extinction, use_kcorrect=use_kcorrect, k_correct_type=k_correct_type, k_correct_arg=k_correct_arg),
    }
    if include is None:
        return list(all_metrics.values())
    return [all_metrics[name] for name in include if name in all_metrics]

