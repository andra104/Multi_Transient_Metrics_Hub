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


# local_IbnIcn_metric.py
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.phot_utils import DustValues
import numpy as np, os, pickle

# pull shared helpers (don’t reimplement)
from shared_utils import evaluate, compare_flux_diff_to_error

# --------------------------
# LC families (Icn / Ibn)
# --------------------------
class LC_Icn:
    def __init__(self, num_samples=200, num_lightcurves=1000, load_from=None):
        self.filts = ['u','g','r','i','z','y']
        self.data, self.t_grid = [], None
        self.ratios = {'u':0.47,'g':1.0,'r':0.44,'i':0.204,'z':0.097,'y':0.035}
        if load_from and os.path.exists(load_from):
            with open(load_from,'rb') as f: obj = pickle.load(f)
            self.data, self.t_grid = obj['lightcurves'], obj.get('t_grid')
            print(f"Loaded Icn templates from {load_from}")
            return
        rng = np.random.default_rng(123)
        for _ in range(num_lightcurves):
            m_peak_g   = rng.uniform(-20.0, -17.0)      # g ~ -17..-20
            alpha_rise = rng.uniform(0.6, 1.6)          # ≈ 0.2–0.3 mag/day near peak
            alpha_fade = rng.uniform(0.4, 1.0)          # ≈ 0.14 mag/day
            A = 10**(0.3/alpha_rise)-1.0; B = 10**(0.3/alpha_fade)-1.0
            t_eps = float(np.clip( np.random.uniform(6,10) / max(A+B,1e-3), 0.05, 5.0))
            t_pre, t_post = -max(5*t_eps,0.5), max(10*t_eps,10.0)
            self.t_grid = np.linspace(t_pre, t_post, num_samples)
            t = self.t_grid
            mag_g = np.empty_like(t); pre = t<0
            mag_g[pre]  = m_peak_g + 2.5*alpha_rise*np.log10((np.abs(t[pre])+t_eps)/t_eps)
            mag_g[~pre] = m_peak_g + 2.5*alpha_fade*np.log10((t[~pre]+t_eps)/t_eps)
            flux_g = 10**(-0.4*mag_g)
            lc = {}
            for f in self.filts:
                flux_f = flux_g * self.ratios[f]
                lc[f]  = {'ph': self.t_grid, 'mag': -2.5*np.log10(flux_f)}
            self.data.append(lc)
    def interp(self, t, f, i=0):
        ph  = np.asarray(self.data[i][f]['ph'], float)
        mag = np.asarray(self.data[i][f]['mag'], float)
        t   = np.asarray(t, float)
        out = np.interp(t, ph, mag, left=np.nan, right=np.nan)
        out[(t<ph.min())|(t>ph.max())] = np.nan
        return out

class LC_Ibn:
    def __init__(self, num_samples=200, num_lightcurves=1000, load_from=None):
        self.filts = ['u','g','r','i','z','y']
        self.data, self.t_grid = [], None
        self.ratios = {'u':0.47,'g':1.0,'r':0.44,'i':0.204,'z':0.097,'y':0.035}
        if load_from and os.path.exists(load_from):
            with open(load_from,'rb') as f: obj = pickle.load(f)
            self.data, self.t_grid = obj['lightcurves'], obj.get('t_grid')
            print(f"Loaded Ibn templates from {load_from}")
            return
        rng = np.random.default_rng(456)
        for _ in range(num_lightcurves):
            m_peak_g   = rng.uniform(-19.5, -17.0)
            alpha_rise = rng.uniform(0.3, 1.1)          # ≈ 0.1–0.15 mag/day
            alpha_fade = rng.uniform(0.4, 0.9)          # ≈ 0.12–0.13 mag/day
            A = 10**(0.3/alpha_rise)-1.0; B = 10**(0.3/alpha_fade)-1.0
            t_eps = float(np.clip( np.random.uniform(5,8) / max(A+B,1e-3), 0.05, 5.0))
            t_pre, t_post = -max(5*t_eps,0.5), max(10*t_eps,10.0)
            self.t_grid = np.linspace(t_pre, t_post, num_samples)
            t = self.t_grid
            mag_g = np.empty_like(t); pre = t<0
            mag_g[pre]  = m_peak_g + 2.5*alpha_rise*np.log10((np.abs(t[pre])+t_eps)/t_eps)
            mag_g[~pre] = m_peak_g + 2.5*alpha_fade*np.log10((t[~pre]+t_eps)/t_eps)
            flux_g = 10**(-0.4*mag_g)
            lc = {}
            for f in self.filts:
                flux_f = flux_g * self.ratios[f]
                lc[f]  = {'ph': self.t_grid, 'mag': -2.5*np.log10(flux_f)}
            self.data.append(lc)
    def interp(self, t, f, i=0):
        ph  = np.asarray(self.data[i][f]['ph'], float)
        mag = np.asarray(self.data[i][f]['mag'], float)
        t   = np.asarray(t, float)
        out = np.interp(t, ph, mag, left=np.nan, right=np.nan)
        out[(t<ph.min())|(t>ph.max())] = np.nan
        return out

# --------------------------
# Thin Base_Metric (reuses shared_utils.evaluate)
# --------------------------
class Base_Metric(BaseMetric):
    def __init__(self, metricName='BaseIbnIcn', 
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night', mjd0=60980.5,
                 lc_model=None, use_extinction=True, use_kcorrect=True,
                 k_correct_type=None, k_correct_arg=None,
                 filter_include=None, badval=-666, **kwargs):
        self.ax1 = DustValues().ax1
        self.mjdCol, self.m5Col = mjdCol, m5Col
        self.filterCol, self.nightCol = filterCol, nightCol
        self.mjd0 = mjd0
        self.lc_model = lc_model
        self.use_extinction = use_extinction
        self.use_kcorrect = use_kcorrect
        self.k_correct_type = k_correct_type
        self.k_correct_arg  = k_correct_arg
        self.filter_include = filter_include
        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName, units='Detection Efficiency', badval=badval, **kwargs)

# --------------------------
# Detection & Characterization
# --------------------------
class IbnIcnDetect_Metric(Base_Metric):
    """Gate A: pre-peak color (≥2 SNR≥10 within 2 hr in different filters)
       Gate B: peak resolution (≥2 SNR≥5 within |t| ≤ 4 d)
       Gate C: pre-peak variability (>3σ flux change in any single band)"""
    def __init__(self, family='icn', **kwargs):
        super().__init__(**kwargs)
        self.family = family
        self.metricName = kwargs.get('metricName', f'{family.upper()}_Detect')
        self.obs_records = {}
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)
        if obs_record is None:
            return self.badval
        if self.filter_include is not None:
            keep = np.isin(filters, self.filter_include)
            snr, filters, times = snr[keep], filters[keep], times[keep]
            for k in ['mjd_obs','mag_obs','snr_obs','filter']:
                if isinstance(obs_record.get(k), np.ndarray):
                    obs_record[k] = obs_record[k][keep]

        # Gate A (pre-peak color)
        pre  = times < 0.0
        good = pre & (snr >= 10)
        color_ok = False
        if np.sum(good) >= 2:
            tg, fg = times[good], np.asarray(filters)[good]
            color_ok = any((0 < abs(tg[j]-tg[i]) < (2/24)) and (fg[i] != fg[j])
                           for i in range(len(tg)) for j in range(i+1, len(tg)))
        if not color_ok:
            detected = False
        else:
            # Gate B (peak resolution)
            near_peak = (np.abs(times) <= 4.0) & (snr >= 5)
            detected = np.sum(near_peak) >= 2
            # Gate C (pre-peak variability)
            if detected:
                f_pre = np.asarray(filters)[pre]
                s_pre = snr[pre]
                m_pre = np.asarray(obs_record['mag_obs'], float)[pre]
                var_ok = False
                for f in np.unique(f_pre):
                    mask = (f_pre == f) & (s_pre >= 5)
                    if np.sum(mask) >= 2:
                        m = m_pre[mask]; s = s_pre[mask]
                        for i in range(m.size):
                            sig = compare_flux_diff_to_error(m, m[i], s, s[i], return_bool=False)
                            if np.size(sig) and np.nanmax(np.atleast_1d(sig)) > 3.0:
                                var_ok = True; break
                    if var_ok: break
                detected = detected and var_ok

        # pack some diagnostics (same pattern as your other metrics)
        detected_mask = snr >= 5
        first_det_mjd = np.min(np.asarray(obs_record['mjd_obs'])[detected_mask]) if np.any(detected_mask) else np.nan
        last_det_mjd  = np.max(np.asarray(obs_record['mjd_obs'])[detected_mask]) if np.any(detected_mask) else np.nan
        mags = np.asarray(obs_record['mag_obs'], float); mjds = np.asarray(obs_record['mjd_obs'], float)
        finite = np.isfinite(mags) & (mags < 90)
        if np.any(finite):
            i_rel = np.nanargmin(mags[finite]); i_abs = np.flatnonzero(finite)[i_rel]
            peak_mag, peak_mjd = float(mags[i_abs]), float(mjds[i_abs])
        else:
            peak_mag, peak_mjd = np.nan, np.nan

        obs_record.update({
            'first_det_mjd': first_det_mjd, 'last_det_mjd': last_det_mjd,
            'sid_duplicate': slice_point['sid'], 'file_indx': slice_point['file_indx'],
            'ra': slice_point['ra'], 'dec': slice_point['dec'],
            'distance_Mpc': slice_point['distance'], 'ebv': slice_point['ebv'],
            'peak_mjd_observed': peak_mjd, 'peak_mag_observed': peak_mag,
            'peak_time': slice_point['peak_time'], 'detected': bool(detected),
            'mag_obs': np.asarray(obs_record.get('mag_obs', [])).tolist(),
            'snr_obs': np.asarray(obs_record.get('snr_obs', [])).tolist(),
            'mjd_obs': np.asarray(obs_record.get('mjd_obs', [])).tolist(),
            'filter':  np.asarray(obs_record.get('filter',  [])).tolist(),
            'distance_modulus': slice_point.get('distance_modulus')
        })
        self.obs_records[slice_point['sid']] = obs_record
        self.latest_obs_record = obs_record if detected else None
        return 1.0 if detected else 0.0

class IbnIcnCharacterizeMetric(Base_Metric):
    """≥5 detections (SNR≥5) within ≤20 d:
       ≥1 on rise (t<0), ≥2 within |t|≤2 d, ≥2 on decline (0<t≤12 d),
       and ≤4 d cadence in at least one band."""
    def __init__(self, family='icn', **kwargs):
        super().__init__(**kwargs)
        self.family = family
        self.metricName = kwargs.get('metricName', f'{family.upper()}_Characterize')
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(self, dataSlice, slice_point, return_full_obs=True)
        good = snr >= 5
        if np.sum(good) < 5:
            return 0.0
        t = times[good]; f = np.asarray(filters)[good]
        if (np.nanmax(t) - np.nanmin(t)) > 20.0:
            return 0.0
        have_rise = np.any(t < 0.0)
        at_peak   = np.sum(np.abs(t) <= 2.0) >= 2
        have_decl = np.sum((t > 0.0) & (t <= 12.0)) >= 2
        cadence_ok = False
        for band in np.unique(f):
            tt = np.sort(t[f == band])
            if tt.size >= 2 and np.nanmax(np.diff(tt)) <= 4.0:
                cadence_ok = True; break
        return 1.0 if (have_rise and at_peak and have_decl and cadence_ok) else 0.0

# --------------------------
# Factory
# --------------------------
def get_multi_metrics(lc_model, include=None, use_extinction=True, use_kcorrect=True,
                      k_correct_type=None, k_correct_arg=None, family='icn'):
    all_metrics = {
        f'{family}_detect': IbnIcnDetect_Metric(family=family, lc_model=lc_model,
                            use_extinction=use_extinction, use_kcorrect=use_kcorrect,
                            k_correct_type=k_correct_type, k_correct_arg=k_correct_arg),
        f'{family}_char'  : IbnIcnCharacterizeMetric(family=family, lc_model=lc_model,
                            use_extinction=use_extinction, use_kcorrect=use_kcorrect,
                            k_correct_type=k_correct_type, k_correct_arg=k_correct_arg),
    }
    if include is None:
        return list(all_metrics.values())
    return [all_metrics[name] for name in include if name in all_metrics]


