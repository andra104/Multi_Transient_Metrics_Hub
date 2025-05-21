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

__all__ = [
    'KN_lc', 'KNePopMetric', 'generateKNPopSlicer',
    'KNeDetectMetric', 'KNeZTFRestSimpleMetric', 'KNeZTFRestSimpleRedMetric',
    'KNeZTFRestSimpleBlueMetric', 'KNeMultiColorDetectMetric',
    'KNeRedColorDetectMetric', 'KNeBlueColorDetectMetric'
]

    
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


class KN_lc:
    def __init__(self, num_samples=None, num_lightcurves=None,
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
                data = pickle.load(f)
            self.data = data['lightcurves']
            self.filts = list(self.data[0].keys())
            print(f"Loaded KN light curve templates from {load_from}")
            return

        if file_list is None:
            datadir = get_data_dir()
            file_list = glob.glob(os.path.join(datadir, 'maf', 'bns', '*.dat'))

        self.filts = ["u", "g", "r", "i", "z", "y"]
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
            print(f"[WARNING] lc_indx {lc_indx} out of range; using last template.")
            lc_indx = len(self.data) - 1
        return np.interp(t,
                         self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)

# --------------------------------------------------
# Light Curve Template Generator (Separate from Population)
# --------------------------------------------------
def generateKNeTemplates(file_list=None, save_to="kne_templates_used.pkl"):
    """
    Generate and cache kilonova light curve templates from Bulla model files.

    Parameters
    ----------
    file_list : list of str or None
        List of Bulla `.dat` files. If None, loads all available models from the default path.
    save_to : str
        Output path for the pickle file storing the light curve templates.

    Notes
    -----
    Templates are stored as a list of dictionaries under the key 'lightcurves', 
    following the same format as GRBAfterglowLC.

    This is optional but recommended if you're injecting a fixed population and want 
    reproducibility and performance when running over multiple cadences.
    """
    if os.path.exists(save_to):
        print(f"Found existing KN light curve templates at {save_to}. Not regenerating.")
        return

    lc_model = KN_lc(file_list=file_list)
    with open(save_to, "wb") as f:
        pickle.dump({'lightcurves': lc_model.data}, f)
    print(f"Saved KN light curve templates to {save_to}")

# --------------------------------------------
# Base KNe Metric with extinction and SNR
# --------------------------------------------
class BaseKNeMetric(BaseMetric):
    def __init__(self, metric_name='BaseKNeMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=59853.5, outputLc=False, badval=-666,
                 filter_include=None,
                 load_from="kne_templates_used.pkl",
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
            self.lc_model = KN_lc(load_from=load_from)

        self.ax1 = DustValues().ax1
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.mjd0 = mjd0
        self.outputLc = outputLc
        self.filter_include = filter_include

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metric_name,
                         units='Detection Efficiency',
                         badval=badval, **kwargs)

    def evaluate_kne(self, dataSlice, slice_point, return_full_obs=True):
        """
        Compute SNR and interpolated light curve for kilonova at this sky location.

        Returns
        -------
        snr : np.ndarray
        filters : np.ndarray
        times : np.ndarray (days since peak)
        obs_record : dict or None
        """
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros(t.size, dtype=float)

        for f in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == f)
            mags[infilt] = self.lc_model.interp(t[infilt], f, lc_indx=slice_point['file_indx'])
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
            }
            return snr, filters, times, obs_record
        return snr, filters, times, None

    def _multi_detect(self, filters, snr, times, min_time_sep=0.0105):
        """
        Check for at least two detections (SNR ≥ 5) in any single filter,
        separated by at least 15 minutes (0.0105 days).
        """
        for f in np.unique(filters):
            mask = (filters == f) & (snr >= 5)
            if np.sum(mask) >= 2:
                t_detect = times[mask]
                dt = np.max(t_detect) - np.min(t_detect)
                if dt >= min_time_sep:
                    return 1
        return 0

        
# --------------------------------------------
# Non-color specific Detection metric (multi_detect)
# --------------------------------------------
class KNeDetectMetric(BaseKNeMetric):
    """
    This is multi-detect. 
    
    KNeDetectMetric:
    Basic kilonova detection criterion based on the 'multi_detect' metric from Andreoni et al. (2022).
    
    Detection requires:
    - At least 2 SNR ≥ 5 detections
    - In the same filter
    - Separated by more than 15 minutes (0.0105 days)
    
    This rejects moving objects and forms the most permissive kilonova detection condition.

    """
    def __init__(self, metric_name="KNe_Detect", **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        self.obs_records = {}


    def run(self, dataSlice, slice_point=None):
        try:
            snr, filters, times, obs_record = self.evaluate_kne(dataSlice, slice_point, return_full_obs=True)

            detected = self._multi_detect(filters, snr, times)
            if detected:
                return self._record_obs(slice_point, snr, filters, times, obs_record, detection_type="baseline")
            return 0.0

        except Exception:
            return self.badval

    def _record_obs(self, slice_point, snr, filters, times, obs_record, detection_type):
        detected_mask = snr >= 5
        first_det_mjd = obs_record['mjd_obs'][detected_mask].min()
        last_det_mjd = obs_record['mjd_obs'][detected_mask].max()
        fade_time = last_det_mjd - (self.mjd0 + slice_point['peak_time'])

        peak_index = np.argmin(obs_record['mag_obs'])
        peak_mjd = obs_record['mjd_obs'][peak_index]
        peak_mag = obs_record['mag_obs'][peak_index]

        obs_record.update({
            'first_det_mjd': first_det_mjd,
            'last_det_mjd': last_det_mjd,
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
            'detected': True,
            'detection_type': detection_type,
            'mjd_obs': obs_record.get('mjd_obs', np.array([])),
            'mag_obs': obs_record.get('mag_obs', np.array([])),
            'snr_obs': obs_record.get('snr_obs', np.array([])),
            'filter': obs_record.get('filter', np.array([]))
        })
        self.obs_records[slice_point['sid']] = obs_record
        self.latest_obs_record = obs_record
        return 1.0

# --------------------------------------------
# Red specific Detection metric (red color detect)
# --------------------------------------------
class KNeRedColorDetectMetric(BaseKNeMetric):
    """
    KNeRedColorDetectMetric:
    Detection metric for red kilonovae.
    
    Detection requires:
    - At least 2 SNR ≥ 5 detections
    - In any of the red filters (i, z, y)
    - Separated by more than 15 minutes
    
    Matches the 'multi_detect' red-filter criterion from Andreoni et al. (2022).
    """
    def __init__(self, metric_name="KNe_RedColorDetect", **kwargs):
        super().__init__(metric_name=metric_name, filter_include=['i', 'z', 'y'], **kwargs)
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        try:
            snr, filters, times, obs_record = self.evaluate_kne(dataSlice, slice_point, return_full_obs=True)

            mask = np.isin(filters, self.filter_include)
            detected = self._multi_detect(filters[mask], snr[mask], times[mask])
            if detected:
                return self._record_obs(slice_point, snr[mask], filters[mask], times[mask], obs_record, detection_type="red")
            return 0.0

        except Exception:
            return self.badval

    def _record_obs(self, slice_point, snr, filters, times, obs_record, detection_type):
        # Same as baseline, but reused
        detected_mask = snr >= 5
        first_det_mjd = obs_record['mjd_obs'][detected_mask].min()
        last_det_mjd = obs_record['mjd_obs'][detected_mask].max()
        fade_time = last_det_mjd - (self.mjd0 + slice_point['peak_time'])

        peak_index = np.argmin(obs_record['mag_obs'])
        peak_mjd = obs_record['mjd_obs'][peak_index]
        peak_mag = obs_record['mag_obs'][peak_index]

        obs_record.update({
            'first_det_mjd': first_det_mjd,
            'last_det_mjd': last_det_mjd,
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
            'detected': True,
            'detection_type': detection_type,
            'mjd_obs': obs_record.get('mjd_obs', np.array([])),
            'mag_obs': obs_record.get('mag_obs', np.array([])),
            'snr_obs': obs_record.get('snr_obs', np.array([])),
            'filter': obs_record.get('filter', np.array([]))
        })
        self.obs_records[slice_point['sid']] = obs_record
        self.latest_obs_record = obs_record
        return 1.0


# --------------------------------------------
# blue specific Detection metric (blue color detect)
# --------------------------------------------
class KNeBlueColorDetectMetric(BaseKNeMetric):
    """
    KNeBlueColorDetectMetric:
    Detection metric for blue kilonovae.
    
    Detection requires:
    - At least 2 SNR ≥ 5 detections
    - In any of the blue filters (u, g, r)
    - Separated by more than 15 minutes
    
    Matches the 'multi_detect' blue-filter criterion from Andreoni et al. (2022).
    """
    def __init__(self, metric_name="KNe_BlueColorDetect", **kwargs):
        super().__init__(metric_name=metric_name, filter_include=['u', 'g', 'r'], **kwargs)
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        try:
            snr, filters, times, obs_record = self.evaluate_kne(dataSlice, slice_point, return_full_obs=True)

            mask = np.isin(filters, self.filter_include)
            detected = self._multi_detect(filters[mask], snr[mask], times[mask])
            if detected:
                return self._record_obs(slice_point, snr[mask], filters[mask], times[mask], obs_record, detection_type="blue")
            return 0.0

        except Exception:
            return self.badval

    def _record_obs(self, slice_point, snr, filters, times, obs_record, detection_type):
        # Same as baseline, reused
        detected_mask = snr >= 5
        first_det_mjd = obs_record['mjd_obs'][detected_mask].min()
        last_det_mjd = obs_record['mjd_obs'][detected_mask].max()
        fade_time = last_det_mjd - (self.mjd0 + slice_point['peak_time'])

        peak_index = np.argmin(obs_record['mag_obs'])
        peak_mjd = obs_record['mjd_obs'][peak_index]
        peak_mag = obs_record['mag_obs'][peak_index]

        obs_record.update({
            'first_det_mjd': first_det_mjd,
            'last_det_mjd': last_det_mjd,
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
            'detected': True,
            'detection_type': detection_type,
            'mjd_obs': obs_record.get('mjd_obs', np.array([])),
            'mag_obs': obs_record.get('mag_obs', np.array([])),
            'snr_obs': obs_record.get('snr_obs', np.array([])),
            'filter': obs_record.get('filter', np.array([]))
        })
        self.obs_records[slice_point['sid']] = obs_record
        self.latest_obs_record = obs_record
        return 1.0

# --------------------------------------------
# Characterization: ztfrest_simple
# --------------------------------------------
class KNeZTFRestSimpleMetric(BaseKNeMetric):
    """
    KNeZTFRestSimpleMetric:
    Characterization metric based on the 'ztfrest_simple' logic from Andreoni et al. (2022).
    
    Detection requires:
    - At least 2 SNR ≥ 5 detections
    - In the same filter
    - Separated by more than 15 minutes
    - Rise rate ≥ 1 mag/day OR fade rate ≥ 0.3 mag/day
    
    This metric enables identification of kilonovae as fast-evolving transients.
    """
    def __init__(self, metric_name='KNe_ZTFReST_Simple', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        try:
            snr, filters, times, obs_record = self.evaluate_kne(dataSlice, slice_point, return_full_obs=True)
            passed, slope_val, filt_used = self._ztfrest_simple_logic(filters, snr, times, obs_record)

            if passed:
                self.obs_records[slice_point['sid']] = {
                    'sid': slice_point['sid'],
                    'ra': slice_point['ra'],
                    'dec': slice_point['dec'],
                    'distance_Mpc': slice_point['distance'],
                    'ebv': slice_point['ebv'],
                    'peak_time': slice_point['peak_time'],
                    'file_indx': slice_point['file_indx'],
                    'characterization_type': 'ztfrest',
                    'characterization_filter': filt_used,
                    'slope_mag_per_day': slope_val,
                }
            return float(passed)

        except Exception:
            return self.badval

    def _ztfrest_simple_logic(self, filters, snr, times, obs_record,
                              min_dt=0.0105, min_fade=0.3, max_rise=-1.0):
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
# Characterization: ztfrest_simple blue
# --------------------------------------------
class KNeZTFRestSimpleBlueMetric(BaseKNeMetric):
    """
    KNeZTFRestSimpleBlueMetric:
    Characterization metric restricted to blue filters (u, g, r), based on 'ztfrest_simple'.
    
    Detection requires:
    - At least 2 SNR ≥ 5 detections in any one of the blue filters
    - Separated by more than 15 minutes
    - Rise rate ≥ 1 mag/day OR fade rate ≥ 0.3 mag/day
    
    Useful for identifying blue kilonova scenarios (e.g., polar or early-time ejecta).
    """
    def __init__(self, metric_name='KNe_ZTFReST_Simple_Blue', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        try:
            snr, filters, times, obs_record = self.evaluate_kne(dataSlice, slice_point)
            passed, slope_val, filt_used = self._ztfrest_simple_logic(filters, snr, times, obs_record, allowed_filters=['u', 'g', 'r'])

            if passed:
                self.obs_records[slice_point['sid']] = {
                    'sid': slice_point['sid'],
                    'ra': slice_point['ra'],
                    'dec': slice_point['dec'],
                    'distance_Mpc': slice_point['distance'],
                    'ebv': slice_point['ebv'],
                    'peak_time': slice_point['peak_time'],
                    'file_indx': slice_point['file_indx'],
                    'characterization_type': 'ztfrest_blue',
                    'characterization_filter': filt_used,
                    'slope_mag_per_day': slope_val,
                }
            return float(passed)

        except Exception:
            return self.badval

    def _ztfrest_simple_logic(self, filters, snr, times, obs_record, allowed_filters,
                              min_dt=0.0105, min_fade=0.3, max_rise=-1.0):
        # identical logic
        for f in np.unique(filters):
            if f not in allowed_filters:
                continue
            mask = (filters == f) & (snr >= 5)
            if np.sum(mask) < 2:
                continue
            t_f = times[mask]
            if np.max(t_f) - np.min(t_f) < min_dt:
                continue
            order = np.argsort(t_f)
            t_f = t_f[order]
            mag_f = obs_record['mag_obs'][mask][order]
            snr_f = obs_record['snr_obs'][mask][order]
            mag_unc_f = 2.5 * np.log10(1. + 1. / snr_f)
            idx_max = np.argmax(mag_f)
            idx_min = np.argmin(mag_f)
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
# Characterization: ztfrest_simple red
# --------------------------------------------
class KNeZTFRestSimpleRedMetric(BaseKNeMetric):
    """
    KNeZTFRestSimpleRedMetric:
    Characterization metric restricted to red filters (i, z, y), based on 'ztfrest_simple'.
    
    Detection requires:
    - At least 2 SNR ≥ 5 detections in any one of the red filters
    - Separated by more than 15 minutes
    - Rise rate ≥ 1 mag/day OR fade rate ≥ 0.3 mag/day
    
    Useful for identifying red kilonova scenarios (e.g., lanthanide-rich ejecta).
    """
    def __init__(self, metric_name='KNe_ZTFReST_Simple_Red', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        try:
            snr, filters, times, obs_record = self.evaluate_kne(dataSlice, slice_point)
            passed, slope_val, filt_used = self._ztfrest_simple_logic(filters, snr, times, obs_record, allowed_filters=['i', 'z', 'y'])

            if passed:
                self.obs_records[slice_point['sid']] = {
                    'sid': slice_point['sid'],
                    'ra': slice_point['ra'],
                    'dec': slice_point['dec'],
                    'distance_Mpc': slice_point['distance'],
                    'ebv': slice_point['ebv'],
                    'peak_time': slice_point['peak_time'],
                    'file_indx': slice_point['file_indx'],
                    'characterization_type': 'ztfrest_red',
                    'characterization_filter': filt_used,
                    'slope_mag_per_day': slope_val,
                }
            return float(passed)

        except Exception:
            return self.badval

    def _ztfrest_simple_logic(self, filters, snr, times, obs_record, allowed_filters,
                              min_dt=0.0105, min_fade=0.3, max_rise=-1.0):
        for f in np.unique(filters):
            if f not in allowed_filters:
                continue
            mask = (filters == f) & (snr >= 5)
            if np.sum(mask) < 2:
                continue
            t_f = times[mask]
            if np.max(t_f) - np.min(t_f) < min_dt:
                continue
            order = np.argsort(t_f)
            t_f = t_f[order]
            mag_f = obs_record['mag_obs'][mask][order]
            snr_f = obs_record['snr_obs'][mask][order]
            mag_unc_f = 2.5 * np.log10(1. + 1. / snr_f)
            idx_max = np.argmax(mag_f)
            idx_min = np.argmin(mag_f)
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
# multi color specific Detection metric (multi color detect)
# --------------------------------------------
class KNeMultiColorDetectMetric(BaseKNeMetric):
    """
    KNeMultiColorDetectMetric:
    Broadband detection diagnostic.
    
    Detection requires:
    - At least 2 filters with SNR ≥ 5 detections
    
    This does not require time separation or filter co-temporality,
    but confirms color information is available to support classification.
    """
    def __init__(self, metric_name='KNe_MultiColorDetect', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        try:
            snr, filters, times, obs_record = self.evaluate_kne(dataSlice, slice_point, return_full_obs=True)
            filters_detected = filters[snr >= 5]
            passed = len(np.unique(filters_detected)) >= 2
            if passed:
                self.obs_records[slice_point['sid']] = {
                    'sid': slice_point['sid'],
                    'ra': slice_point['ra'],
                    'dec': slice_point['dec'],
                    'distance_Mpc': slice_point['distance'],
                    'ebv': slice_point['ebv'],
                    'peak_time': slice_point['peak_time'],
                    'file_indx': slice_point['file_indx'],
                    'characterization_type': 'multicolor',
                    'n_filters_detected': len(np.unique(filters_detected))
                }
            return float(passed)
        except Exception:
            return self.badval



# ---------------------------------------------------------------------------
# KNe volumetric rate model  5 − 950 * 10^−9 Mpc^−3*y^−1 (5–950 Gpc^−3*y^−1)
# ---------------------------------------------------------------------------
def sample_kne_rate_from_volume(t_start, t_end, d_min, d_max,
                                rate_density=300e-9):  # events/Mpc^3/yr
    """
    Compute expected number of kilonovae based on volumetric rate.

    Parameters
    ----------
    t_start, t_end : float
        Time window in days.
    d_min, d_max : float
        Distance bounds in Mpc.
    rate_density : float
        Rate in events/Mpc³/yr (default = 300e-9 = 300 Gpc⁻³ yr⁻¹).

    Returns
    -------
    int
        Number of expected kilonovae in the survey duration and volume.
    """
    years = (t_end - t_start) / 365.25
    z_min = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
    z_max = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)

    V = cosmo.comoving_volume(z_max).to(u.Mpc**3).value - cosmo.comoving_volume(z_min).to(u.Mpc**3).value
    return np.random.poisson(rate_density * V * years)
   
# --------------------------------------------
# KNe population generator
# --------------------------------------------
def generateKNePopSlicer(t_start=1, t_end=3652, rate_density=300e-9,  # Default to median estimate
                         seed=42, d_min=10, d_max=300, n_files=100,gal_lat_cut=None, nside=64,
                         load_from=None, save_to=None):
    """
    Generate a kilonova population using uniform HEALPix injection.

    Parameters
    ----------
    t_start, t_end : float
        Time range of peak brightness (days).
    n_events : int
        Total number of kilonovae to simulate.
    d_min, d_max : float
        Luminosity distance range (Mpc).
    n_files : int
        Number of Bulla light curve templates to randomly assign.
    gal_lat_cut : float or None
        If set, remove events near the Galactic plane (|b| < cut).
    nside : int
        HEALPix nside for uniform injection.
    load_from : str or None
        If provided, load a previously generated population.
    save_to : str or None
        If provided, save this population to .pkl file.

    Returns
    -------
    slicer : UserPointsSlicer
        Slicer object with injected population metadata.
    """
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        print(f"Loaded KNe population from {load_from}")
        return slicer

    rng = np.random.default_rng(seed)
    n_events = sample_kne_rate_from_volume(t_start, t_end, d_min, d_max, rate_density)

    # HEALPix injection
    ra, dec = inject_uniform_healpix(nside=nside, n_events=n_events, seed=seed)
    dec = np.clip(dec, -89.9999, 89.9999)

    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0) #returns radians 


    # Metadata
    peak_times = rng.uniform(t_start, t_end, n_events)
    distances = rng.uniform(d_min, d_max, n_events)
    file_indx = rng.integers(0, n_files, n_events)

    coords = SkyCoord(ra=np.degrees(slicer.slice_points['ra']) * u.deg, dec=np.degrees(slicer.slice_points['dec']) * u.deg, frame='icrs') #this line correctly converts them and labels them
    sfd = SFDQuery()
    ebv_vals = sfd(coords)

    # Apply Galactic latitude cut if requested
    if gal_lat_cut is not None:
        b = coords.galactic.b.deg
        mask = np.abs(b) > gal_lat_cut
        ra = ra[mask]
        dec = dec[mask]
        peak_times = peak_times[mask]
        distances = distances[mask]
        file_indx = file_indx[mask]
        ebv_vals = ebv_vals[mask]
        coords = coords[mask]

    # Slicer
    #slicer = UserPointsSlicer(ra=ra_rad, dec=dec_rad, badval=0)
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['distance'] = distances
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    slicer.slice_points['gall'] = coords.galactic.l.deg
    slicer.slice_points['galb'] = coords.galactic.b.deg

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)
        print(f"Saved KNe population to {save_to}")

    # --- GRB-style diagnostic plots and summary ---

    plt.hist(slicer.slice_points['ra'], bins=50)
    plt.xlabel("RA [rad]")
    plt.title("Injected KNe Population – RA Distribution")
    plt.grid(True)
    plt.show()

    plt.hist(slicer.slice_points['dec'], bins=50)
    plt.xlabel("Dec [rad]")
    plt.title("Injected KNe Population – Dec Distribution")
    plt.grid(True)
    plt.show()

    plt.hist(peak_times, bins=50)
    plt.xlabel("Peak Time [days]")
    plt.title("Injected KNe Population – Peak Time Distribution")
    plt.grid(True)
    plt.show()

    plt.hist(distances, bins=50)
    plt.xlabel("Distance [Mpc]")
    plt.title("Injected KNe Population – Distance Distribution")
    plt.grid(True)
    plt.show()

    print(f"[DEBUG] First 5 RA/Dec (rad): {slicer.slice_points['ra'][:5]}, {slicer.slice_points['dec'][:5]}")

    coords = SkyCoord(
        ra=np.degrees(slicer.slice_points['ra']) * u.deg,
        dec=np.degrees(slicer.slice_points['dec']) * u.deg,
        frame='icrs'
    )

    print(f"[DEBUG] SkyCoord sample Dec: {coords.dec[:5]}")
    print(f"[DEBUG] SkyCoord Dec units: {coords.dec.unit}")

    plt.hist(coords.ra.deg, bins=50)
    plt.xlabel("RA [deg]")
    plt.title("Injected KNe Population – SkyCoord RA [deg]")
    plt.grid(True)
    plt.show()

    plt.hist(coords.dec.deg, bins=50)
    plt.xlabel("Dec [deg]")
    plt.title("Injected KNe Population – SkyCoord Dec [deg]")
    plt.grid(True)
    plt.show()


    return slicer






