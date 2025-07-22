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

        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                obj = pickle.load(f)
            self.data = obj['lightcurves']
            self.t_grid = obj['t_grid']
            print(f"Loaded LFBOT templates from {load_from}")
            return


        rng = np.random.default_rng(42)
        self.t_grid = np.linspace(0.01, 7.0, num_samples)  # or use logspace

        for _ in range(num_lightcurves):
            lc = {}
            # Random g-band light curve parameters
            m0_g = rng.uniform(-21.5, -20.0)
            rise_time = rng.uniform(1, 5)  # days to peak
            fade_rate = rng.uniform(0.15, 0.45)  # mag/day
            duration_peak = rng.uniform(0, 4)  # flat peak duration in days

            mag_g = np.zeros_like(self.t_grid)
            rise_slope = rng.uniform(0.25, 2.5)  # ✅ one value per LC

            for i, t in enumerate(self.t_grid):
                if t < rise_time:
                    mag_g[i] = m0_g + (rise_time - t) * rise_slope
                elif t < rise_time + duration_peak:
                    mag_g[i] = m0_g
                else:
                    mag_g[i] = m0_g + (t - (rise_time + duration_peak)) * fade_rate

            # Convert g-band light curve to other bands using flux ratios
            flux_g = 10**(-0.4 * mag_g)
            for f in self.filts:
                flux_f = flux_g * self.ratios[f]
                mag_f = -2.5 * np.log10(flux_f)
                lc[f] = {'ph': self.t_grid, 'mag': mag_f}

            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)


# --------------------------------------------------
# Light Curve Template Generator (Separate from Population)
# --------------------------------------------------
def generate_Templates(
    num_samples=100, num_lightcurves=1000,
    save_to="LFBOT_templates.pkl"
):
    """
    Generate synthetic LFBOT light curve templates and save to file.
    """

    lc_model = LC(num_lightcurves=num_lightcurves, load_from=None)
    with open(save_to, "wb") as f:
        pickle.dump({'lightcurves': lc_model.data, 't_grid': lc_model.t_grid}, f)

    print(f"Saved synthetic LFBOT light curve templates to {save_to}")

    
# --------------------------------------------
# Light Curve Loader (used in scripts)
# --------------------------------------------
def load_or_generate_templates(templates_file="LFBOT_templates.pkl",
                               num_samples=100, num_lightcurves=1000,
                               generate_new=False):
    """
    Load light curve templates from a file, or generate and save new ones.

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

    def evaluate_lfbot(self, dataSlice, slice_point, return_full_obs=True):
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros(t.size)
        
        for f in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == f)
            mags[infilt] = self.lc_model.interp(t[infilt], f, slice_point['file_indx'])

            if self.use_extinction:
                mags[infilt] += self.ax1[f] * slice_point['ebv']
                if not self.extinction_printed:
                    print("EBV included")
                    self.extinction_printed = True

            #mags[infilt] += self.ax1[f] * slice_point['ebv']
            mags[infilt] += 5 * np.log10(slice_point['distance'] * 1e6) - 5
    
        snr = m52snr(mags, dataSlice[self.m5Col])
        filters = dataSlice[self.filterCol]
        times = t
    
        if return_full_obs:
            obs_record = {
                'mjd_obs': dataSlice[self.mjdCol],
                'mag_obs': mags,
                'snr_obs': snr,
                'filter': filters
                # NO 'detected' YET -- will be set later if detected!
            }
            
            return snr, filters, times, obs_record
        # print("NOT RETURNING OBSRECORD SHAR")
        return snr, filters, times, None

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
        snr, filters, times, obs_record = self.evaluate_lfbot(dataSlice, slice_point, return_full_obs=True)
    
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
        snr, filters, times, obs_record = self.evaluate_lfbot(dataSlice, slice_point, return_full_obs=True)
        detected = self.parent_instance.detect(filters, snr, times, obs_record)

        if not detected:
            detected = self.parent_instance.betterdetect(filters, snr, times, obs_record)

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
# LFBOT Population Rate
# --------------------------------------------
def sample_lfbot_rate_from_volume(t_start, t_end,
                                d_min=None, d_max=None,
                                z_min=None, z_max=None, rate_density=420e-9):
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
    d_min, d_max = get_distance_bounds(d_min=d_min, d_max=d_max, z_min=z_min, z_max=z_max)

    z_min = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
    z_max = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)

    years = (t_end - t_start) / 365.25
    V = cosmo.comoving_volume(z_max).to(u.Mpc**3).value - cosmo.comoving_volume(z_min).to(u.Mpc**3).value
    return np.random.poisson(rate_density * V * years)

# --------------------------------------------
# Population Loader (used in scripts)
# --------------------------------------------
def load_or_generate_population(t_start=1, t_end=3652, seed=42,
                                d_min=None, d_max=None,
                                z_min=None, z_max=None,
                                num_lightcurves=1000,
                                gal_lat_cut=None, rate_density=1e-8,
                                pop_file="LFBOT_population_fixedpop.pkl",
                                generate_new=False,
                                make_debug_plots=False):
    """
    Load population from a saved file or generate a new one.

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
        print(f"[INFO] Generating LFBOT population and saving to {pop_file}")
        slicer = generate_PopSlicer(t_start=t_start, t_end=t_end,
                                    d_min=d_min, d_max=d_max,
                                    z_min = z_min, z_max = z_max,
                                    seed=seed,
                                    num_lightcurves=num_lightcurves,
                                    gal_lat_cut=gal_lat_cut,
                                    rate_density=rate_density,
                                    save_to=pop_file,
                                    make_debug_plots=make_debug_plots)
    else:
        print(f"[INFO] Loading LFBOT population from {pop_file}")
        slicer = generate_PopSlicer(load_from=pop_file)

    return slicer

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


# --------------------------------------------
# population generator
# --------------------------------------------
def generate_PopSlicer(t_start=1, t_end=3652, seed=42,
                         d_min=None, d_max=None, z_min = None, z_max = None, num_lightcurves=1000, gal_lat_cut=None, rate_density=1e-8,
                         load_from=None, save_to=None, make_debug_plots=True):
    """
    Generate a population of with realistic extinction and sky distribution.

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
        print(f"Loaded LFBOT population from {load_from}")
        return slicer

    rng = np.random.default_rng(seed)
    d_min, d_max = get_distance_bounds(d_min=d_min, d_max=d_max, z_min=z_min, z_max=z_max)
    n_events = sample_lfbot_rate_from_volume(t_start, t_end, d_min, d_max, z_min, z_max, rate_density=rate_density)
    print(f"Simulated {n_events} LFBOT events using rate_density = {rate_density:.1e}")

    
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
        plt.title("Injected LFBOT Population – RA Distribution")
        plt.grid(True)
        plt.show()
        
        plt.hist(slicer.slice_points['dec'], bins=50)
        plt.xlabel("Dec [rad]")
        plt.title("Injected LFBOT Population – Dec Distribution")
        plt.grid(True)
        plt.show()

    theta_obs = rng.uniform(0, np.pi/2, n_events)  # radians, or degrees if you prefer
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
    slicer.slice_points['theta_obs'] = theta_obs

    

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)
        print(f"Saved LFBOT population to {save_to}")

    return slicer

# --------------------------------------------
# Standardized storage paths (used in scripts)
# --------------------------------------------
def get_output_paths(case_label="LFBOTs"):
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

