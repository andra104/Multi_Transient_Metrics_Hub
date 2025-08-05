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

# -----------------------------------------------------------------------------
# Light Curve Parameter Definitions for Shock-Cooling Emission Peak in SNe IIb
# -----------------------------------------------------------------------------
SCE_PARAMETERS = {
    'g': {
        'rise_rate_mu': 1.09, #mag/day -> use rise rate and 1st peak mag to define the initial mag w t_rise
        'rise_rate_sigma': 0.34, #mag/day 
        'fade_rate_mu': 0.23, #mag/day -> use the 1st peak mag and the decline rate to set the min mag between peaks t_fade
        'fade_rate_sigma': 0.087, #mag/day
        
        'peak_mag_range': (-18.65, -14.82), #mag 
        #'duration_at_peak': 2.35, #days ?????
        'min_mag_bw_peaks': (-14.2,-17.2),
        'second_peak_mag_range': (-17.5, -15.0), #mag
        'second_peak_rise_mu': 0.082, #mag/day -> use the second peak mag, the min mag between peak, the rise time to set the time of Ni peak
        'second_peak_rise_sigma': 0.059, #mag/day
        'final_fade': 0.01
    },
    'r': {
        'rise_rate_mu': 0.97,
        'rise_rate_sigma': 0.35,
        'fade_rate_mu': 0.18,
        'fade_rate_sigma': 0.095,
        'peak_mag_range': (-18.21, -14.82),
        #'duration_at_peak': 2.90,
        'min_mag_bw_peaks': (-14.4,-17.5),
        'second_peak_mag_range': (-17.9, -15.4),
        'second_peak_rise_mu': 0.091,
        'second_peak_rise_sigma': 0.053,
        'final_fade': 0.01

    },
    'i': { #SHAR BELOW HERE NOT PHYSICAL
        'rise_rate_mu': 0.97,
        'rise_rate_sigma': 0.35,
        'fade_rate_mu': 0.18,
        'fade_rate_sigma': 0.095,
        'peak_mag_range': (-18.21, -14.82),
        #'duration_at_peak': 2.90,
        'min_mag_bw_peaks': (-14.4,-17.5),
        'second_peak_mag_range': (-17.9, -15.4),
        'second_peak_rise_mu': 0.091,
        'second_peak_rise_sigma': 0.053,
        'final_fade': 0.01
    },
    'u': {
        'rise_rate_mu': 0.97,
        'rise_rate_sigma': 0.35,
        'fade_rate_mu': 0.18,
        'fade_rate_sigma': 0.095,
        'peak_mag_range': (-18.21, -14.82),
        #'duration_at_peak': 2.90,
        'min_mag_bw_peaks': (-14.4,-17.5),
        'second_peak_mag_range': (-17.9, -15.4),
        'second_peak_rise_mu': 0.091,
        'second_peak_rise_sigma': 0.053,
        'final_fade': 0.01
    },
    'z': {
        'rise_rate_mu': 0.97,
        'rise_rate_sigma': 0.35,
        'fade_rate_mu': 0.18,
        'fade_rate_sigma': 0.095,
        'peak_mag_range': (-18.21, -14.82),
        #'duration_at_peak': 2.90,
        'min_mag_bw_peaks': (-14.4,-17.5),
        'second_peak_mag_range': (-17.9, -15.4),
        'second_peak_rise_mu': 0.091,
        'second_peak_rise_sigma': 0.053,
        'final_fade': 0.01

    },
    'y': {
        'rise_rate_mu': 0.97,
        'rise_rate_sigma': 0.35,
        'fade_rate_mu': 0.18,
        'fade_rate_sigma': 0.095,
        'peak_mag_range': (-18.21, -14.82),
        #'duration_at_peak': 2.90,
        'min_mag_bw_peaks': (-14.4,-17.5),
        'second_peak_mag_range': (-17.9, -15.4),
        'second_peak_rise_mu': 0.091,
        'second_peak_rise_sigma': 0.053,
        'final_fade': 0.01

    }
}

# --------------------------------------------
# Light Curve Model
# --------------------------------------------

class LC:
    """
    Generate synthetic light curves.

    """
    def __init__(self, num_samples = 100, num_lightcurves=1000, load_from=None):
        self.filts = ['u', 'g', 'r', 'i', 'z', 'y']
        self.data = []
        self.t_grid = None  # 0.1–10 days

        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                obj = pickle.load(f)
            self.data = obj['lightcurves']
            self.t_grid = obj['t_grid']
            print(f"Loaded templates from {load_from}")
            return


        # --- Otherwise generate templates from scratch ---
        self.data = []
        self.durations = {}
        self.filts = list(SCE_PARAMETERS.keys())

        def sample_rate(mu, sigma):
            return np.random.normal(mu, sigma)

        t_rise = np.linspace(-1.5, 0, num_samples // 5)
        t_fade = np.linspace(0.01, 5, num_samples)
        t_rerise = np.linspace(7, 13, num_samples)

        
        for _ in range(num_lightcurves):
            lightcurve = {}
            for i,f in enumerate(self.filts):
                params = SCE_PARAMETERS[f]
    
                #bound magnitudes
                if i==0:
                    peak_mag_1 = np.random.uniform(*params['peak_mag_range'])
                    peak_mag_2 = np.random.uniform(*params['second_peak_mag_range'])
                    min_mag = max(np.random.uniform(*params['min_mag_bw_peaks']),
                                  max(peak_mag_1, peak_mag_2) + 0.5)
                    #at least 0.5 mag lower than either mags
                    rise1_rate = max(sample_rate(params['rise_rate_mu'], params['rise_rate_sigma']), 0.5)
                    fade1_rate = max(sample_rate(params['fade_rate_mu'], params['fade_rate_sigma']), 0.1)
                    rise2_rate = max(sample_rate(params['second_peak_rise_mu'], params['second_peak_rise_sigma']), 0.02)
                else:
                    peak_mag_1 = max(sample_rate(peak_mag_1, 0.2), peak_mag_1)
                    peak_mag_2 = min(sample_rate(peak_mag_2, 0.3), peak_mag_2)
                    min_mag = sample_rate(min_mag, 0.05)
                    rise1_rate = max(sample_rate(rise1_rate, 0.01), 0.5)
                    fade1_rate = max(sample_rate(fade1_rate, 0.01), 0.1)
                    rise2_rate = max(sample_rate(rise2_rate, 0.01), 0.02)
            
                #firse rise: use fist peak mag and rate and let evolve cor 2 days backward
                t_rise = np.linspace(-2, 0, 2)
                mag_rise = peak_mag_1 + rise1_rate * (t_rise[::-1] - t_rise[0]) / np.ptp(t_rise)
        
                if DEBUG:
                    print("initial mag:", mag_rise[0])
                    print("first peak", peak_mag_1)
                    print("min mag:", min_mag)                                   
                    print("second peak", peak_mag_2)
                    #first fade
            
        
                dmag = min_mag - peak_mag_1 
                #ensure t_fade is not too long
                t_fade = min(dmag / fade1_rate, 10)
                #recalc fade1_rate: if t_fade did not get replaced it will be the same otherwise its faster
                fade1_rate = dmag / t_fade
                t_fade = np.linspace(0, t_fade, 2)
                mag_fade = peak_mag_1 + fade1_rate * (t_fade) 
        
                #52Ni peak
                dmag = min_mag - peak_mag_2 
                t_rerise = min(dmag / rise2_rate, 20 - t_fade[-1])
                #recalc rise time : if the rise time was short enough to fit in 18 days its the same, otherwise its faster
                rise2_rate = dmag / t_rerise
                t_rerise = np.linspace(t_fade[-1], t_fade[-1] + t_rerise, 2)
                mag_rerise = min_mag - rise2_rate * (t_rerise - t_rerise[0]) 
        
                #final decline
                t_decline = np.linspace(t_rerise[1], 25, 2)
                mag_decline = peak_mag_2 + params["final_fade"] * (t_decline - t_decline[0]) 
        
                lightcurve[f] = {'ph': np.concatenate([t_rise, t_fade, t_rerise, t_decline]), 
                     'mag': np.concatenate([mag_rise, mag_fade, mag_rerise, mag_decline])
                     }
                timeline = np.linspace(lightcurve[f]['ph'][0], lightcurve[f]['ph'][-1], 100)
                # plt.plot(timeline, np.interp(timeline, lightcurve[f]['ph'],
                #                  lightcurve[f]['mag'],
                #                  left=99, right=99))
    
                # plt.gca().invert_yaxis()
                # plt.show()

                

                if f not in self.durations:
                    self.durations[f] = {'rise': [], 'fade': [], 'rerise': []}
                self.durations[f]['rise'].append(np.ptp(t_rise))
                self.durations[f]['fade'].append(np.ptp(t_fade))
                self.durations[f]['rerise'].append(np.ptp(t_rerise))

            self.data.append(lightcurve)

    def interp(self, t, filtername, lc_indx=0):
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)



# --------------------------------------------
# Base Metric
# --------------------------------------------
class Base_Metric(BaseMetric):
    """
    Base metric class for evaluating light curves against simulated observations.

    This class handles light curve interpolation, extinction correction, and signal-to-noise
    calculation, providing a standardized evaluation framework for derived metrics.
    """
    def __init__(self, metricName='Base_Metric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=60980.5, outputLc=False, badval=-666,
                 filter_include=None, load_from="templates.pkl", use_extinction=True, use_kcorrect=True,k_correct_type=None,k_correct_arg=None,
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
        self.use_kcorrect = use_kcorrect
        self.k_correct_type = k_correct_type
        self.k_correct_arg = k_correct_arg
        self.extinction_printed = False

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName, units='Detection Efficiency', badval=badval, **kwargs)


    def detect(self, filters, snr, times, obs_record):
        detected = False
    
        # Convert to arrays just in case
        filters = np.array(filters)
        snr = np.array(snr)
        times = np.array(times)
    

        detected_part_1=False
        #new detection criteria: two in different filters in the night, two in same filter in six nights
        snr_good = snr>5
        times_good = times[snr_good]
        filters_good = filters[snr_good]
        for i, time in enumerate(times_good):
            time_diff = (0<(times_good-time)) * ((times_good-time)<0.5) #must two obs in same filter between 0 and half a day
            # print(time_diff)
            if len(np.unique(filters_good[time_diff])) >=2:
                detected_part_1 = True
                break
        if not detected_part_1:
            return detected
            
        for f in np.unique(filters):
            mask = filters == f
            times_in_filter = times[mask]
            snr_in_filter = snr[mask]
            observed_detection_times = times_in_filter[snr_in_filter >= 5]
            if len(observed_detection_times)>=2: #require 2+ detections in the same filter
                for i, time in enumerate(observed_detection_times):
                    time_diff = (1<(observed_detection_times-time)) * ((observed_detection_times-time)<6*3) #shar added *3
                    # print(observed_detection_times-time)
                    if np.sum(time_diff)>0:
                        # print("found one")
                        detected = True
                        return detected
        
        return detected


    

# --------------------------------------------
# Detection Metric for LFBOTs
# --------------------------------------------
class Detect_Metric(Base_Metric):
    """
    Right now this is just the LFBOT metric, we may not even need it ultimately
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
            # 'theta_obs': slice_point['theta_obs'],
            'filter': obs_record.get('filter', np.array([])).tolist(),
            'distance_modulus': 5 * np.log10(slice_point['distance'] * 1e6) - 5
        })    

        self.obs_records[slice_point['sid']] = obs_record
        self.latest_obs_record = obs_record if detected else None
    
        return 1.0 if detected else 0.0

# --------------------------------------------
# Characterization Metric
# --------------------------------------------
class SCECharacterizeMetric(Base_Metric):
    """

    2 detections in the rise, 2 in fade, 2 in rerise
    """
    def __init__(self, **kwargs):
        use_extinction = kwargs.pop('use_extinction', True)
        use_kcorrect = kwargs.pop('use_kcorrect', True)
        super().__init__(**kwargs, use_extinction=use_extinction, use_kcorrect=use_kcorrect)
        self.metricName = kwargs.get('metricName', 'LFBOT_Characterize')
        self.obs_records = {}  # <-- NEW: to store all detected event records individually
        self.parent_instance = Base_Metric(use_extinction=use_extinction, use_kcorrect=use_kcorrect)

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)
        is_detected = self.parent_instance.detect(filters, snr, times, obs_record)
        mjd_obs = obs_record.get('mjd_obs', np.array([]))
        detected=False

        if is_detected:
            for f in np.unique(filters):
                #print("[DEBUG] filter", f)
                mask = (dataSlice[self.filterCol] == f) 
                t_filt = times[mask]  #time stamps in thet filter
                snr_filt = snr[mask] #associated SNR
                
                if np.sum(snr_filt >= 3) >= 4: #need at least 4 points cause 3 are for sure before peak to be characterized
                    dur_rise   = shared_lc_model.durations[f][
                        'rise'][slice_point['file_indx']]#[mask] 
                    dur_fade   = shared_lc_model.durations[f][
                        'fade'][slice_point['file_indx']]#[mask]
                    dur_rerise = shared_lc_model.durations[f][
                        'rerise'][slice_point['file_indx']]#[mask]
                
                    t_second_rise = dur_rise + dur_fade + dur_rerise
                    
                    #request 1 3sig observation after second peak
                    second_rise = np.sum((t_filt > t_second_rise) & (snr_filt >= 3.0))
                    if second_rise >= 1:
                        double_peak_detected = True
                        break
        
        return detected
    

# --------------------------------------------
# Multi_Metric Standardized Call
# --------------------------------------------
def get_multi_metrics(lc_model, include=None, use_extinction=True, use_kcorrect=False, k_correct_type=None, k_correct_arg=None):
    """
    Return a list of metrics. `include` can be a list of metric names to include.
    """
    all_metrics = {
        'detect': Detect_Metric(lc_model=lc_model, use_extinction=use_extinction, use_kcorrect=use_kcorrect, k_correct_type=k_correct_type, k_correct_arg=k_correct_arg),
        'characterize': SCECharacterizeMetric(lc_model=lc_model, use_extinction=use_extinction, use_kcorrect=use_kcorrect, k_correct_type=k_correct_type, k_correct_arg=k_correct_arg),
        # 'GRBcharacterize': GRBAfterglowCharacterizeMetric(lc_model=lc_model, use_extinction=use_extinction),
        # 'GRB_spec_trigger': GRBAfterglowSpecTriggerableMetric(lc_model=lc_model, use_extinction=use_extinction),
        # 'GRBDetect': GRBDetect_Metric(lc_model=lc_model, use_extinction=use_extinction)
    }

    if include is None:
        return list(all_metrics.values())
    else:
        return [all_metrics[name] for name in include if name in all_metrics]



