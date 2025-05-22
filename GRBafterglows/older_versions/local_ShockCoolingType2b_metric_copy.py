from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues
from rubin_sim.maf.metric_bundles import MetricBundle
from rubin_sim.maf.utils import m52snr
import matplotlib.pyplot as plt
from rubin_sim.maf.db import ResultsDb
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import Galactic, ICRS as ICRSFrame
from rubin_sim.maf.slicers import HealpixSlicer
import rubin_sim.maf.metric_bundles as metric_bundles
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from rubin_sim.maf.metrics import CountMetric
from rubin_sim.maf.maps import StellarDensityMap
#from rubin_sim.phot_utils import SFDMap
import astropy.units as u
import healpy as hp
import numpy as np
import glob
import os
import sys
sys.path.append("/Users/andradenebula/Documents/Research/Transient_Metrics/Multi_Transient_Metrics_Hub")
from shared_utils import inject_uniform_healpix
import pickle 

# -----------------------------------------------------------------------------
# Light Curve Parameter Definitions for Shock-Cooling Emission Peak in SNe IIb
# -----------------------------------------------------------------------------
SCE_PARAMETERS = {
    'g': {
        'rise_rate_mu': 1.09,
        'rise_rate_sigma': 0.34,
        'fade_rate_mu': 0.23,
        'fade_rate_sigma': 0.087,
        'peak_mag_min': -18.65,
        'peak_mag_max': -14.82,
        'duration_at_peak': 2.35,
        'second_peak_mag_range': (-17.5, -15.0),
        'second_peak_rise_mu': 0.082,
        'second_peak_rise_sigma': 0.059
    },
    'r': {
        'rise_rate_mu': 0.97,
        'rise_rate_sigma': 0.35,
        'fade_rate_mu': 0.18,
        'fade_rate_sigma': 0.095,
        'peak_mag_min': -18.21,
        'peak_mag_max': -14.82,
        'duration_at_peak': 2.90,
        'second_peak_mag_range': (-17.9, -15.4),
        'second_peak_rise_mu': 0.091,
        'second_peak_rise_sigma': 0.053
    }
}

# ============================================================
# Light Curve Generator for Shock Cooling Events
# ============================================================
class ShockCoolingLC:
    def __init__(self, num_samples=100, load_from=None):
        """
        Generate or load synthetic light curves for Shock Cooling Emission in SN IIb.

        Parameters
        ----------
        num_samples : int
            Number of time samples per light curve
        load_from : str or None
            If provided, loads templates from a pickle file
        """
        # --- Load pre-generated templates if requested ---
        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                data = pickle.load(f)
            self.data = data['lightcurves']
            self.durations = data['durations']
            self.filts = list(self.data[0].keys())
            print(f"Loaded {len(self.data)} shock cooling light curves from {load_from}")
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

        for _ in range(100):
            lightcurve = {}
            for f in self.filts:
                params = SCE_PARAMETERS[f]

                peak_mag_1 = np.random.uniform(params['peak_mag_min'], params['peak_mag_max'])
                rise1 = sample_rate(params['rise_rate_mu'], params['rise_rate_sigma'])
                fade1 = sample_rate(params['fade_rate_mu'], params['fade_rate_sigma'])
                mag_rise = peak_mag_1 - rise1 * (t_rise - np.min(t_rise)) / np.ptp(t_rise)
                mag_peak1 = np.full((1,), peak_mag_1)
                mag_fade = peak_mag_1 + fade1 * t_fade

                peak_mag_2 = np.random.uniform(*params['second_peak_mag_range'])
                rise2 = sample_rate(params['second_peak_rise_mu'], params['second_peak_rise_sigma'])
                mag_rerise = peak_mag_2 - rise2 * (13 - t_rerise) / 6

                t_full = np.concatenate([t_rise, [0], t_fade, t_rerise])
                mag_full = np.concatenate([mag_rise, mag_peak1, mag_fade, mag_rerise])
                lightcurve[f] = {'ph': t_full, 'mag': mag_full}

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

# ============================================================
# Base Shock Cooling Metric 
# ============================================================

class BaseShockCoolingType2bMetric(BaseMetric):
    """
    Base class for evaluating Shock-Cooling Emission Peaks in Type IIb SNe.
    Simulates light curves and applies detection and classification criteria.
    """
    def __init__(self, metricName='ShockCoolingType2bMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=59853.5, outputLc=False, badval=-666,
                 include_second_peak=True,
                 load_from="ShockCooling_templates.pkl",
                 lc_model=None, 
                 **kwargs):


        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.mjd0 = mjd0
        self.outputLc = outputLc
        self.include_second_peak = include_second_peak

        self._lc_model = lc_model if lc_model is not None else ShockCoolingLC(load_from=load_from)
        self.ax1 = DustValues().ax1

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName, units='N/A',
                         maps=['DustMap'], badval=badval, **kwargs)

    def evaluate_event(self, dataSlice, slice_point):
        """
        Evaluate whether a Shock-Cooling light curve is detected, characterized, and shows a second peak.
    
        Parameters
        ----------
        dataSlice : numpy.recarray
            The subset of the cadence data relevant for this slice point.
        slice_point : dict
            Metadata for the specific injected SN event (distance, time, etc.).
    
        Returns
        -------
        result : dict
            Contains:
            - 'detection' (int): 1 if detected, 0 otherwise
            - 'characterization' (str): 'classical', 'ambiguous', or 'uncharacterized'
            - 'double_peak' (bool): True if second peak is resolved (optional)

        Changes
        --------
        Replaces static 7–13 day window with:

        t_start_rerise = dur_fade
        
        t_end_rerise = dur_fade + dur_rerise
        
        Uses actual per-template durations from ShockCoolingLC.durations[filter][file_indx]
        
        Keeps SNR threshold of 0.5 and 2 observations
        
        Continues looping through filters until one satisfies the condition
        """
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros_like(t)

        for f in np.unique(dataSlice[self.filterCol]):
            if f not in self._lc_model.filts:
                continue
            infilt = np.where(dataSlice[self.filterCol] == f)
            mags[infilt] = self._lc_model.interp(t[infilt], f, slice_point['file_indx'])
            mags[infilt] += self.ax1[f] * slice_point['ebv']

        snr = m52snr(mags, dataSlice[self.m5Col])

        # Detection logic
        detected = 0
        for f in self._lc_model.filts:
            filt_mask = (dataSlice[self.filterCol] == f)
            t_filt = t[filt_mask]
            snr_filt = snr[filt_mask]
            if np.sum(snr_filt >= 5) >= 2:
                if np.any(np.diff(np.sort(t_filt[snr_filt >= 5])) >= 0.5):
                    detected = 1
                    break

        # Default values for not-detected events
        characterization = 'uncharacterized'
        double_peak_detected = False

        if detected:
            for f in self._lc_model.filts:
                filt_mask = (dataSlice[self.filterCol] == f)
                if np.sum(filt_mask) < 2:
                    continue
                t_filt = t[filt_mask]
                snr_filt = snr[filt_mask]
                char = self.characterize_event(snr_filt, t_filt, f, slice_point['file_indx'])
                if char != 'uncharacterized':
                    characterization = char
                    break

        if self.include_second_peak and detected:
            for f in self._lc_model.filts:
                if f not in self._lc_model.durations:
                    continue
                mask = dataSlice[self.filterCol] == f
                t_filt = t[mask]
                snr_filt = snr[mask]

                dur_fade = self._lc_model.durations[f]['fade'][slice_point['file_indx']]
                dur_rerise = self._lc_model.durations[f]['rerise'][slice_point['file_indx']]

                t_start_rerise = dur_fade
                t_end_rerise = dur_fade + dur_rerise

                second_rise_times = t_filt[(t_filt > t_start_rerise) & (t_filt <= t_end_rerise) & (snr_filt >= 0.5)]
                if len(second_rise_times) >= 1:
                    double_peak_detected = True
                    break

        return {
            'detection': detected,
            'characterization': characterization,
            'double_peak': double_peak_detected
        }

    def characterize_event(self, snr, times, filtername, file_indx):
        """
        Characterize a light curve as 'classical', 'ambiguous', or 'uncharacterized'.

        Updated logic reflects the physical criteria for SCE light curves:
          - At least 2 detections (SNR ≥ 0.5) in each segment:
            • Initial rise (–5 to 0 days)
            • Decline (0 to +7 days)
            • Re-rise (7 to +17 days)
          - Time between detections in each segment must be:
            • ≤ 3 days for rise
            • ≤ 4 days for fade
            • ≤ 6 days for re-rise
          - Total: 6+ detections across segments within 13 days

        Previously, the re-rise segment only required 1 detection and ignored spacing.

        Parameters
        ----------
        snr : array
            Signal-to-noise ratio values for each observation.
        times : array
            Times of each observation relative to peak.
        filtername : str
            Filter used (e.g. 'g' or 'r').
        file_indx : int
            Index for light curve template.

        Returns
        -------
        str
            'classical', 'ambiguous', or 'uncharacterized'
        """
        snr_mask = snr >= 0.5
        if np.sum(snr_mask) < 6:
            return 'uncharacterized'

        t_rise = times[(times >= -5) & (times <= 0) & snr_mask]
        t_fade = times[(times > 0) & (times <= 7) & snr_mask]
        t_rerise = times[(times > 7) & (times <= 17) & snr_mask]

        def spaced_pair(t_arr, max_dt):
            if len(t_arr) < 2:
                return False
            return np.any(np.diff(np.sort(t_arr)) < max_dt)

        if len(t_rise) >= 2 and spaced_pair(t_rise, 3) and \
           len(t_fade) >= 2 and spaced_pair(t_fade, 4) and \
           len(t_rerise) >= 1:
            return 'classical'

        elif len(t_rise) + len(t_fade) + len(t_rerise) >= 6:
            return 'ambiguous'
        return 'uncharacterized'



class ShockCoolingDetectMetric(BaseShockCoolingType2bMetric):
    """
    Metric for determining if a Shock-Cooling SN IIb is detected.

    Detection = 1 if ≥2 detections (SNR ≥ 5) in a single filter, 
    separated by at least 0.5 days.
    """
    def __init__(self, metricName='ShockCooling_Detect', **kwargs):
        super().__init__(metricName=metricName, **kwargs)

    def run(self, dataSlice, slice_point):
        result = self.evaluate_event(dataSlice, slice_point)
        return result['detection']


class ShockCoolingCharacterizeMetric(BaseShockCoolingType2bMetric):
    """
    Metric for classifying the quality of a detected Shock-Cooling SN IIb.

    Only evaluates events with detection=1.

    Characterization = 
        'classical' if well-resolved three-phase behavior is seen,
        'ambiguous' if marginal behavior is seen,
        'uncharacterized' if insufficient SNR/timing.
    """
    def __init__(self, metricName='ShockCooling_Characterize', **kwargs):
        super().__init__(metricName=metricName, **kwargs)

    def run(self, dataSlice, slice_point):
        result = self.evaluate_event(dataSlice, slice_point)
        if result['detection']:
            label = result['characterization']
            if label == 'classical':
                return 2
            elif label == 'ambiguous':
                return 1
        return 0  # not detected or uncharacterized

class ShockCoolingDoublePeakMetric(BaseShockCoolingType2bMetric):
    """
    Metric to determine if the second (re-rise) peak is detected 
    in a Shock-Cooling Type IIb supernova.

    Requires:
      - Event is detected
      - Second peak phase has at least 1 detection (SNR ≥ 0.5) 
        with ≥0.5 day spacing
    """
    def __init__(self, metricName='ShockCooling_DoublePeak', **kwargs):
        super().__init__(metricName=metricName, **kwargs)

    def run(self, dataSlice, slice_point):
        result = self.evaluate_event(dataSlice, slice_point)
        return int(result['double_peak'])








# ============================================================
# Population Generator for Shock-Cooling Type IIb Events
# ============================================================

def generateShockCoolingPopSlicer(t_start=1, t_end=3652,
                                  rate_per_year=65000,
                                  seed=42, d_min=10, d_max=300,
                                  gal_lat_cut=None,
                                  nside=64, save_to=None,
                                  templates_file="ShockCooling_templates.pkl"):
    """
    Generate a uniformly distributed Shock Cooling SN IIb population using HEALPix.
    This version does NOT use LSST footprint weighting.
    
    Parameters
    ----------
    t_start : float
        Start time in MJD (default: 1).
    t_end : float
        End time in MJD (default: 3652 for 10 years).
    rate_per_year : int
        Expected number of observable SNe IIb with shock cooling per year (default: 65,000).
    seed : int
        Random seed.
    d_min : float
        Minimum distance in Mpc.
    d_max : float
        Maximum distance in Mpc.
    gal_lat_cut : float or None
        Minimum Galactic latitude to avoid crowded plane, e.g. 15 deg.
    nside : int
        HEALPix nside resolution.
    prob_map : ndarray or None
        Normalized probability map over the sky (e.g. from StellarDensityMap).
    save_to : str or None
        If provided, saves the population to this pickle file.
    templates_file : str
        File to save light curve templates.

    Returns
    -------
    UserPointsSlicer
        Slicer with assigned shock-cooling SN IIb population.
    """

    n_years = (t_end - t_start) / 365.25
    n_events = int(rate_per_year * n_years)
    print(f"Simulating {n_events} SN IIb events over {n_years:.2f} years (rate = {rate_per_year}/yr)")

    rng = np.random.default_rng(seed)

    # Uniform sky injection like GRBafterglows
    ra_deg, dec_deg = inject_uniform_healpix(nside=nside, n_events=n_events, seed=seed)
    dec_deg = np.clip(dec_deg, -89.9999, 89.9999)

    # Convert to SkyCoord
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    if gal_lat_cut is not None:
        b = coords.galactic.b.deg
        mask = np.abs(b) > gal_lat_cut
        coords = coords[mask]

        
    # Re-draw associated event metadata
    distances = rng.uniform(d_min, d_max, len(coords))
    peak_times = rng.uniform(t_start, t_end, len(coords))
    file_indx = rng.integers(0, 100, len(coords))

    # Query dust extinction
    sfd = SFDQuery()
    ebv_vals = sfd(coords)

    # Final RA/Dec
    ra = coords.ra.deg
    dec = coords.dec.deg

    # Build slicer
    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)
    slicer.slice_points['ra'] = ra
    slicer.slice_points['dec'] = dec
    slicer.slice_points['distance'] = distances
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    slicer.slice_points['sid'] = hp.ang2pix(nside, np.radians(90. - dec), np.radians(ra), nest=False)
    slicer.slice_points['nside'] = nside
    slicer.slice_points['dec_rad'] = np.radians(dec)

    # Save slicer population
    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)
        print(f"Saved Shock Cooling population to {save_to}")

    # Save light curve templates
    from local_ShockCoolingType2b_metric import ShockCoolingLC
    lc_model = ShockCoolingLC()
    with open(templates_file, "wb") as f:
        pickle.dump({'lightcurves': lc_model.data, 'durations': lc_model.durations}, f)
    print(f"Saved synthetic shock-cooling templates to {templates_file}")

    return slicer


def plot_shockcooling_population_summary(slicer, title_prefix="Shock Cooling SN IIb Population"):
    import matplotlib.pyplot as plt
    import numpy as np

    ra = slicer.slice_points['ra']
    dec = slicer.slice_points['dec']
    peak_times = slicer.slice_points['peak_time']
    distances = slicer.slice_points['distance']

    # RA
    plt.figure(figsize=(8, 4))
    plt.hist(ra, bins=50)
    plt.xlabel("RA [deg]")
    plt.ylabel("Count")
    plt.title(f"{title_prefix}: RA Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Dec
    plt.figure(figsize=(8, 4))
    plt.hist(dec, bins=50)
    plt.xlabel("Dec [deg]")
    plt.ylabel("Count")
    plt.title(f"{title_prefix}: Dec Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Peak time
    plt.figure(figsize=(8, 4))
    plt.hist(peak_times, bins=50)
    plt.xlabel("Peak Time [days since MJD 59853.5]")
    plt.ylabel("Count")
    plt.title(f"{title_prefix}: Peak Time Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Distance
    plt.figure(figsize=(8, 4))
    plt.hist(distances, bins=50)
    plt.xlabel("Distance [Mpc]")
    plt.ylabel("Count")
    plt.title(f"{title_prefix}: Distance Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

