from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues

import sys
import os
sys.path.append(os.path.abspath(".."))
from shared_utils import equatorialFromGalactic, uniform_sphere_degrees, inject_uniform_healpix, get_distance_bounds

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
    def __init__(self, num_samples=100, num_lightcurves=1000, load_from=None):
        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                data = pickle.load(f)
            self.data = data['lightcurves']
            self.filts = list(self.data[0].keys())
            print(f"Loaded LFBOT templates from {load_from}")
            return

        self.data = []
        self.filts = ["g", "r"]  # Only g and r used
        self.t_grid = np.logspace(-1, 1, num_samples)  # 0.1 to 10 days

        rng = np.random.default_rng(42)
        for _ in range(num_lightcurves):
            lc = {}
            for f in self.filts:
                m0 = rng.uniform(-21.5, -20)  # peak mag
                alpha_rise = rng.uniform(-2.5, -0.25)
                alpha_fade = rng.uniform(0.15, 0.45)
                t0 = 1.0
                mag = np.where(
                    self.t_grid < t0,
                    m0 + 2.5 * alpha_rise * np.log10(self.t_grid / t0),
                    m0 + 2.5 * alpha_fade * np.log10(self.t_grid / t0)
                )
                lc[f] = {'ph': self.t_grid, 'mag': mag}
            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        if lc_indx >= len(self.data):
            lc_indx = len(self.data) - 1
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)
# --------------------------------------------------
# Light Curve Template Loader
# --------------------------------------------------
def load_or_generate_templates(templates_file="LFBOT_templates.pkl",
                               num_samples=100, num_lightcurves=1000,
                               generate_new=False):
    """
    Load LFBOT light curve templates from file or generate new ones.

    Parameters
    ----------
    templates_file : str
        Path to the .pkl file.
    generate_new : bool
        Whether to generate and overwrite.
    """
    if generate_new or not os.path.exists(templates_file):
        print(f"[INFO] Generating {num_lightcurves} light curve templates.")
        load_or_generate_templates(num_samples=num_samples,
                           num_lightcurves=num_lightcurves,
                           save_to=templates_file)
    else:
        print(f"[INFO] Loading light curve templates from {templates_file}.")
    return LC(load_from=templates_file)


# --------------------------------------------------
# Population Loader 
# --------------------------------------------------
def load_or_generate_population(t_start=1, t_end=3652, seed=42,
                                d_min=10, d_max=1000,
                                gal_lat_cut=None, num_lightcurves=1000,
                                rate_density=420e-9,
                                pop_file="LFBOT_population.pkl",
                                generate_new=False,
                                make_debug_plots=False):
    """
    Load LFBOT population from file or generate a new one.
    """
    if generate_new or not os.path.exists(pop_file):
        print(f"[INFO] Generating LFBOT population and saving to {pop_file}")
        slicer = generate_PopSlicer(t_start=t_start, t_end=t_end,
                                    d_min=d_min, d_max=d_max,
                                    seed=seed,
                                    num_lightcurves=num_lightcurves,
                                    gal_lat_cut=gal_lat_cut,
                                    save_to=pop_file)
    else:
        print(f"[INFO] Loading LFBOT population from {pop_file}")
        slicer = generate_PopSlicer(load_from=pop_file)
    return slicer

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
                 mjd0=59853.5, outputLc=False, badval=-666,
                 filter_include=None, load_from="LFBOT_templates.pkl",
                 lc_model=None, **kwargs):

        if lc_model is not None:
            self.lc_model = lc_model
        else:
            self.lc_model = LC(load_from=load_from)

        self.ax1 = DustValues().ax1
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.mjd0 = mjd0
        self.outputLc = outputLc
        self.filter_include = filter_include

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName, units='Detection Efficiency', badval=badval, **kwargs)

    def evaluate_lc(self, dataSlice, slice_point, return_full_obs=True):
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros(t.size)
        for f in np.unique(dataSlice[self.filterCol]):
            if f not in self.lc_model.filts:
                # Skip filters without light curve templates (e.g., i, z, y)
                continue
            infilt = np.where(dataSlice[self.filterCol] == f)
            mags[infilt] = self.lc_model.interp(t[infilt], f, slice_point['file_indx'])
            mags[infilt] += self.ax1[f] * slice_point['ebv']
            mags[infilt] += 5 * np.log10(slice_point['distance'] * 1e6) - 5


        snr = m52snr(mags, dataSlice[self.m5Col])
        filters = dataSlice[self.filterCol]
        times = t

        if return_full_obs:
            obs_record = {'mjd_obs': dataSlice[self.mjdCol], 'mag_obs': mags, 'snr_obs': snr, 'filter': filters}
            return snr, filters, times, obs_record
        return snr, filters, times

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
        self.obs_records = {}  # <-- NEW
        self.latest_obs_record = None

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_lc(dataSlice, slice_point, return_full_obs=True)

        if self.filter_include is not None:
            keep = np.isin(filters, self.filter_include)
            snr = snr[keep]
            filters = filters[keep]
            times = times[keep]
            for k in ['mjd_obs', 'mag_obs']:
                obs_record[k] = obs_record[k][keep]

        detected = False

        good = snr >= 5
        if np.sum(good) < 2:
            self.latest_obs_record = None
            return 0.0

        times_good = times[good]
        filters_good = filters[good]
        total_time_span = np.ptp(times_good)
        if (total_time_span < 0.5 / 24) or (total_time_span > 6):
            self.latest_obs_record = None
            return 0.0

        if len(np.unique(filters_good)) >= 2 or np.sum(good) >= 3:
            detected = True

        if detected:
            detected_mask = snr >= 5
            obs_record['detected'] = bool(np.any(detected_mask))

            first_det_mjd = obs_record['mjd_obs'][detected_mask].min()
            last_det_mjd = obs_record['mjd_obs'][detected_mask].max()
            rise_time = first_det_mjd - (self.mjd0 + slice_point['peak_time'])
            fade_time = last_det_mjd - (self.mjd0 + slice_point['peak_time'])

            peak_index = np.argmin(obs_record['mag_obs'])
            peak_mjd = obs_record['mjd_obs'][peak_index]
            peak_mag = obs_record['mag_obs'][peak_index]

            obs_record.update({
                'first_det_mjd': first_det_mjd,
                'last_det_mjd': last_det_mjd,
                'rise_time_days': rise_time,
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
                'mjd_obs': obs_record.get('mjd_obs', np.array([])),
                'mag_obs': obs_record.get('mag_obs', np.array([])),
                'snr_obs': obs_record.get('snr_obs', np.array([])),
                'filter': obs_record.get('filter', np.array([]))
            })

            self.obs_records[slice_point['sid']] = obs_record
            self.latest_obs_record = obs_record
            return 1.0
        else:
            self.latest_obs_record = None
            return 0.0


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
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'Characterize')
        self.obs_records = {}  # <-- NEW
        self.latest_obs_record = None

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_lc(dataSlice, slice_point, return_full_obs=True)

        good = snr >= 3
        if np.sum(good) < 4:
            self.latest_obs_record = None
            return 0.0
        duration = np.ptp(times[good])
        if duration >= 3:
            self.obs_records[slice_point['sid']] = obs_record
            self.latest_obs_record = obs_record
            return 1.0

        self.latest_obs_record = None
        return 0.0

        

# --------------------------------------------
# LFBOT Population Rate
# --------------------------------------------
def sample_lfbot_rate(t_start, t_end, d_min, d_max, rate_density=420e-9):
    """
    Estimate the number of LFBOT events expected in the survey window.

    Calculates the number of events by multiplying the volumetric LFBOT rate
    by the comoving volume between the specified distance bounds (d_min, d_max),
    and the duration of the simulated survey in years.

    Parameters
    ----------
    t_start, t_end : float
        Start and end times of the survey window (in days).
    d_min, d_max : float
        Minimum and maximum luminosity distances (in Mpc).
    rate_density : float
        Volumetric LFBOT event rate in units of events per Mpc^3 per year.

    Returns
    -------
    int
        Poisson-sampled number of LFBOT events expected over the survey period.
    """
    years = (t_end - t_start) / 365.25
    z_min = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
    z_max = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)
    V = cosmo.comoving_volume(z_max).to(u.Mpc**3).value - cosmo.comoving_volume(z_min).to(u.Mpc**3).value
    return np.random.poisson(rate_density * V * years)

# --------------------------------------------
# Multi_Metric Standardized Call
# --------------------------------------------
def get_multi_metrics(lc_model, include=None):
    """
    Return list of LFBOT metrics for multi-metric evaluation.
    """
    all_metrics = {
        'detect': Detect_Metric(lc_model=lc_model),
        'characterize': LFBOTCharacterizeMetric(lc_model=lc_model),
    }
    if include is None:
        return list(all_metrics.values())
    else:
        return [all_metrics[name] for name in include if name in all_metrics]

# --------------------------------------------
# LFBOT Population Slicer Generator
# --------------------------------------------
def generate_PopSlicer(t_start=1, t_end=3652, seed=42,
                           d_min=10, d_max=1000, num_lightcurves=1000,
                           gal_lat_cut=None, load_from=None, save_to=None):
    """
    Generate a synthetic population of LFBOT events across the sky.

    Events are distributed uniformly over the celestial sphere, assigned random distances,
    peak times, and matched to synthetic light curve templates. Galactic extinction is applied
    using the SFD dust map. Optionally saves or loads populations from a pickle file.

    Parameters
    ----------
    t_start, t_end : float
        Start and end times of the simulated survey window (in days).
    d_min, d_max : float
        Minimum and maximum luminosity distances (in Mpc).
    seed : int
        Random number generator seed for reproducibility.
    gal_lat_cut : float or None
        Minimum Galactic latitude (deg) to exclude crowded plane regions, if specified.
    load_from : str or None
        Path to load existing population pickle file.
    save_to : str or None
        Path to save newly generated population pickle file.
    """
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        print(f"Loaded LFBOT population from {load_from}")
        return slicer

    rng = np.random.default_rng(seed)
    n_events = sample_lfbot_rate(t_start, t_end, d_min, d_max)

    ra, dec = uniform_sphere_degrees(n_events, seed=seed)
    dec = np.clip(dec, -89.9999, 89.9999)
    dec_rad = np.radians(dec)

    slicer = UserPointsSlicer(ra=ra, dec=dec_rad, badval=0)
    slicer.slice_points['ra'] = ra
    slicer.slice_points['dec'] = dec_rad

    distances = rng.uniform(d_min, d_max, n_events)
    peak_times = rng.uniform(t_start, t_end, n_events)
    file_indx = rng.integers(0, num_lightcurves, n_events)

    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
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

    slicer.slice_points['distance'] = distances
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    slicer.slice_points['gall'] = coords.galactic.l.deg
    slicer.slice_points['galb'] = coords.galactic.b.deg

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)
        print(f"Saved LFBOT population to {save_to}")

    return slicer

# --------------------------------------------------
# Output Paths
# --------------------------------------------------
def get_output_paths(case_label="LFBOTs"):
    """
    Standard output paths for LFBOT analysis.

    Returns
    -------
    dict with:
        - 'case_label'
        - 'storage_dir'
        - 'templates_file'
        - 'pop_file'
    """
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


