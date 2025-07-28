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

    
# --------------------------------------------
# Bulla KNe Light Curve Model Files
# --------------------------------------------

def get_filename(inj_params_list):
    """Given kilonova parameters, get the filename from the grid of models
    developed by M. Bulla

    Parameters
    ----------
    inj_params_list : list of dict
        parameters for the kilonova model such as
        mass of the dynamical ejecta (mej_dyn), mass of the disk wind ejecta
        (mej_wind), semi opening angle of the cylindrically-symmetric ejecta
        fan ('phi'), and viewing angle ('theta'). For example
        inj_params_list = [{'mej_dyn': 0.005,
              'mej_wind': 0.050,
              'phi': 30,
              'theta': 25.8}]
    """
    # Get files, model grid developed by M. Bulla
    datadir = get_data_dir()
    file_list = glob.glob(os.path.join(datadir, 'maf', 'bns', '*.dat'))
 
    params = {}
    matched_files = []
    for filename in file_list:
        key = filename.replace(".dat","").split("/")[-1]
        params[key] = {}
        params[key]["filename"] = filename
        keySplit = key.split("_")
        # Binary neutron star merger models
        if keySplit[0] == "nsns":
            mejdyn = float(keySplit[2].replace("mejdyn",""))
            mejwind = float(keySplit[3].replace("mejwind",""))
            phi0 = float(keySplit[4].replace("phi",""))
            theta = float(keySplit[5])
            params[key]["mej_dyn"] = mejdyn
            params[key]["mej_wind"] = mejwind
            params[key]["phi"] = phi0
            params[key]["theta"] = theta
        # Neutron star--black hole merger models
        elif keySplit[0] == "nsbh":
            mej_dyn = float(keySplit[2].replace("mejdyn",""))
            mej_wind = float(keySplit[3].replace("mejwind",""))
            phi = float(keySplit[4].replace("phi",""))
            theta = float(keySplit[5])
            params[key]["mej_dyn"] = mej_dyn
            params[key]["mej_wind"] = mej_wind
            params[key]["phi"] = phi
            params[key]["theta"] = theta
    for key in params.keys():
        for inj_params in inj_params_list:
            match = all([np.isclose(params[key][var],inj_params[var]) for var in inj_params.keys()])
            if match:
                matched_files.append(params[key]["filename"])
                print(f"Found match for {inj_params}")
    print(f"Found matches for {len(matched_files)}/{len(inj_params_list)} \
          sets of parameters")

    return matched_files


class LC:
    def __init__(self, num_samples=100, num_lightcurves=1000,
                 file_list=None, load_from=None):
        """
        Kilonova light curve loader using pre-computed Bulla model files.

        Parameters
        ----------
        num_samples : ignored
            Present only for API consistency with GRBAfterglowLC.
        num_lightcurves : ignored
            Present only for API consistency with GRBAfterglowLC.
        file_list : list of str or None
            List of Bulla `.dat` files. If None, loads all from data dir.
        load_from : str or None
            Path to a .pkl file with preloaded lightcurve templates.
        """
        if load_from is not None and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                obj = pickle.load(f)
            self.data = obj['lightcurves']
            self.filts = list(self.data[0].keys())
            print(f"Loaded KN light curve templates from {load_from}")
            return

        if file_list is None:
            datadir = get_data_dir()
            file_list = glob.glob(os.path.join(datadir, 'maf', 'bns', '*.dat'))

        self.filts = ["u", "g", "r", "i", "z", "y"]
        self.t_grid = None  # Set per-lightcurve

        magidxs = [1, 2, 3, 4, 5, 6]
        self.data = []

        for filename in file_list:
            mag_ds = np.loadtxt(filename)
            t = mag_ds[:, 0]
            new_dict = {}
            for filt, magidx in zip(self.filts, magidxs):
                new_dict[filt] = {'ph': t, 'mag': mag_ds[:, magidx]}
            self.data.append(new_dict)

    def interp(self, t, filtername, lc_indx=0):
        """
        Interpolate the light curve in the given filter at times `t`.

        Parameters
        ----------
        t : array_like
            Times in days relative to peak.
        filtername : str
            LSST filter (u, g, r, i, z, y).
        lc_indx : int
            Index of the light curve to use.

        Returns
        -------
        magnitudes : array_like
            Interpolated magnitudes. Returns 99 outside valid range.
        """
        if lc_indx >= len(self.data):
            if not hasattr(self, '_warned_once'):
                print(f"[WARNING] Some lc_indx values exceeded number of templates. Using last template.")
                self._warned_once = True
            lc_indx = len(self.data) - 1

        return np.interp(t,
                         self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)

# --------------------------------------------
# Base KNe Metric with extinction and SNR
# --------------------------------------------
class Base_Metric(BaseMetric):
    def __init__(self, metricName='BaseKNeMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=60980.5, outputLc=False, badval=-666,
                 filter_include=None,
                 load_from="kne_templates_used.pkl", use_extinction=True,
                 lc_model=None,
                 **kwargs):
        """
        Base class for kilonova metrics using Bulla light curves.

        Parameters
        ----------
        lc_model : KN_lc or None
            Shared kilonova light curve model. If None, loads from file.
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



    def detect(self, filters, snr, times, obs_record, min_dt=0.0105, min_fade=0.3, max_rise=-1.0):
            """
            Apply ZTFReST Simple detection logic to define a kilonova detection.
    
            Detection requires:
            - ≥2 SNR ≥ 5 detections
            - Same filter
            - Separated by ≥15 minutes
            - Brightening or fading rate ≥ threshold
            """
        
            detected, _, _ = self._ztfrest_simple_logic(
                filters, snr, times, obs_record,
                min_dt=min_dt, min_fade=min_fade, max_rise=max_rise
            )
            return detected

    def _ztfrest_simple_logic(self, filters, snr, times, obs_record, min_dt=0.0105, min_fade=0.3, max_rise=-1.0):
        
        for f in np.unique(filters):
            mask = (filters == f) & (snr >= 5)
            if np.sum(mask) < 2:
                continue
    
            t_f = times[mask]
            if np.max(t_f) - np.min(t_f) < min_dt:
                continue
    
            mag_order = np.argsort(t_f)
            t_f = t_f[mag_order]
            mag_f = obs_record['mag_obs'][mask][mag_order]
            snr_f = obs_record['snr_obs'][mask][mag_order]
            mag_unc_f = 2.5 * np.log10(1. + 1. / snr_f)
    
            idx_max = np.argmax(mag_f)
            idx_min = np.argmin(mag_f)
            if idx_max == idx_min:
                continue
    
            dt = np.abs(t_f[idx_max] - t_f[idx_min])
            if dt < min_dt:
                continue
    
            brightening = mag_f[idx_min] + mag_unc_f[idx_min]
            fading = mag_f[idx_max] - mag_unc_f[idx_max]
    
            if brightening < fading:
                slope = (mag_f[idx_max] - mag_f[idx_min]) / dt
                if slope >= min_fade or slope <= max_rise:
                    return True, slope, f
        return False, None, None

# --------------------------------------------
# ZTFReST-Based Detection Metric for Kilonovae
# --------------------------------------------
class Detect_Metric(Base_Metric):
    """ 
    Detection metric for kilonovae based on ZTFReST simple logic.

    Requires:
    - ≥2 SNR ≥ 5 detections in the same filter
    - Separated by ≥15 minutes
    - A rise rate ≥ 1 mag/day or fade rate ≥ 0.3 mag/day
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'Detect')
        self.obs_records = {}  # stores full observation records
        self.parent_instance = Base_Metric()

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)

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
        fade_time = np.nan

        if np.any(detected_mask):
            first_det_mjd = obs_record['mjd_obs'][detected_mask].min()
            last_det_mjd = obs_record['mjd_obs'][detected_mask].max()
            fade_time = last_det_mjd - (self.mjd0 + slice_point['peak_time'])

        peak_index = np.argmin(obs_record['mag_obs'])
        peak_mjd = obs_record['mjd_obs'][peak_index] if len(obs_record['mjd_obs']) > 0 else np.nan
        peak_mag = obs_record['mag_obs'][peak_index] if len(obs_record['mag_obs']) > 0 else np.nan

        obs_record.update({
            'first_det_mjd': first_det_mjd,
            'last_det_mjd': last_det_mjd,
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

class KNeCharacterizeColorEvolveMetric(Base_Metric):
    """
    Characterization metric for kilonovae based on color evolution.

    Requires:
    - ≥2 distinct color measurements (filter pairs within 1 hr of each other)
    - Separated by ≥3 days
    - Same filter pair (or one overlapping filter and redder second)
    - All detections SNR ≥ 5
    """
    def __init__(self, **kwargs):
        use_extinction = kwargs.pop('use_extinction', True)
        super().__init__(**kwargs, use_extinction=use_extinction)
        self.metricName = kwargs.get('metricName', 'Characterize_ColorEvolve')
        self.obs_records = {}
        self.parent_instance = Base_Metric(use_extinction=use_extinction)

        self.max_pair_dt = 1.0 / 24  # 1 hour in days
        self.min_sep_days = 3.0

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)     
        detected = self.parent_instance.detect(filters, snr, times, obs_record)
        if not detected:
            return 0.0

        # Only SNR ≥ 5
        good = snr >= 5
        if np.sum(good) < 4:
            return 0.0

        filters = filters[good]
        times = times[good]
        mags = obs_record['mag_obs'][good]
        mjds = obs_record['mjd_obs'][good]

        # Sort by time
        sort_idx = np.argsort(mjds)
        filters = filters[sort_idx]
        times = times[sort_idx]
        mags = mags[sort_idx]
        mjds = mjds[sort_idx]

        # Search for color pairs ≤1 hr apart
        pairs = []
        for i in range(len(mjds)):
            for j in range(i + 1, len(mjds)):
                if abs(mjds[i] - mjds[j]) <= self.max_pair_dt and filters[i] != filters[j]:
                    filt_pair = tuple(sorted([filters[i], filters[j]]))
                    pairs.append((mjds[i], mjds[j], filt_pair))

                    #print(filt_pair)
                    #print(mags[i])
                    #print(mags[j])
                    #print(mjds[i])
                    #print(mjds[j])
            
                        

        if len(pairs) < 2: 
            return 0.0

        # Look for valid evolution pair (≥3 days apart, same or overlapping filters)
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                t1, tone, f1 = pairs[i]
                t2, tsecond, f2 = pairs[j]
                if abs(t2 - t1) >= self.min_sep_days:
                    overlap = set(f1).intersection(set(f2))
                    if len(overlap) >= 1:
                        print(pairs[i])
                        print(pairs[j], "\n")
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
        'Detect': Detect_Metric(lc_model=lc_model, use_extinction=use_extinction),
        'Characterize_ColorEvolve': KNeCharacterizeColorEvolveMetric(lc_model=lc_model, use_extinction=use_extinction)

    }

    if include is None:
        return list(all_metrics.values())
    else:
        return [all_metrics[name] for name in include if name in all_metrics]


