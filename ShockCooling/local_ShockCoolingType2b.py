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

import pickle 
DEBUG  = False
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

    }
}

def generateSCETemplates(
    num_samples=100, num_lightcurves=1000,
    save_to="ShockCooling_template.pkl"
):
    """
    Generate synthetic SCE afterglow light curve templates and save to file.
    """
    if os.path.exists(save_to):
        print(f"Found existing SCE templates at {save_to}. Not regenerating.")
        return

    lc_model = ShockCoolingLC(num_samples=num_samples, #num_lightcurves=num_lightcurves,
                              load_from=None)
     
    with open(save_to, "wb") as f:
        pickle.dump({'lightcurves': lc_model.data, 'durations': lc_model.durations}, f)
    print(f"Saved synthetic GRB light curve templates to {save_to}")

    
# ============================================================
# Light Curve Generator for Shock Cooling Events
# ============================================================
class ShockCoolingLC:
    def __init__(self, num_samples=100, load_from=None, show=None):
        """
        Generate or load synthetic light curves for Shock Cooling Emission in SN IIb.

        Parameters
        ----------
        num_samples : int
            Number of time samples per light curve
        load_from : str or None
            If provided, loads templates from a pickle file
        """
        np.random.seed(302)
        print(load_from)
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
                plt.plot(timeline, np.interp(timeline, lightcurve[f]['ph'],
                                 lightcurve[f]['mag'],
                                 left=99, right=99))
    
                plt.gca().invert_yaxis()
                plt.show()

                

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
    def returnlc(self, lc_index):
        return self.data[lc_index]

# ============================================================
# Shared Evaluation Logic and Metric Subclasses
# ============================================================
class BaseShockCoolingType2bMetric(BaseMetric):
    """
    Base class for evaluating Shock-Cooling Emission Peaks in Type IIb SNe.
    Simulates light curves and applies detection and classification criteria.
    """
    def __init__(self, metricName='ShockCoolingType2bMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night', ptsNeeded=1,
                 mjd0=59853.5, outputLc=False, badval=-666,
                 include_second_peak=True, load_from="ShockCooling_templates.pkl", **kwargs):

        self._lc_model = ShockCoolingLC(load_from=load_from)
        self.ax1 = DustValues().ax1
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.outputLc = outputLc
        self.mjd0 = mjd0
        self.include_second_peak = include_second_peak  # Enables second peak detection logic


        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName,
                         units='Detected, 0 or 1', maps=['DustMap'],
                         badval=badval, **kwargs)

    def characterize_sce(self, snr, times, filtername, file_indx):

        idx = np.where(snr >= 3.0)[0]
        if len(idx) < 6:
            return 'uncharacterized'
        
        dur_rise   = self._lc_model.durations[filtername]['rise'][file_indx]
        dur_fade   = self._lc_model.durations[filtername]['fade'][file_indx]
        dur_rerise = self._lc_model.durations[filtername]['rerise'][file_indx]

        rise  = np.sum((times >= -dur_rise) & (times <= 0) & (snr >= 3))
        fade  = np.sum((times > 0) & (times <= dur_fade) & (snr >= 3))
        rerise = np.sum((times > dur_fade) & (times <= dur_fade + dur_rerise) & (snr >= 3))

        # 2 datapoits in rise at >=3sig + 2 datapoints in fade at >=3sig + 1 datapoint in rerise >=3sig
        if rise >= 2 and fade >= 2 and rerise >= 1: #changed from 2 
            return 'classical'
        elif rise + fade + rerise >= 6:
            return 'ambiguous'
        return 'uncharacterized'

    def evaluate_sce(self, dataSlice, slice_point):
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
        """

        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros(t.size, dtype=float)

        #create the lightcurve by inrepolate the model at slide_point['file_indx']
        for f in np.unique(dataSlice[self.filterCol]):
            if f not in self._lc_model.filts:
                continue
            infilt = (dataSlice[self.filterCol] == f)
            mags[infilt] = self._lc_model.interp(t[infilt], f, slice_point['file_indx'])
            mags[infilt] += self.ax1[f] * slice_point['ebv']  # Apply extinction
            
        snr = m52snr(mags, dataSlice[self.m5Col])

        # Detection logic: 2 obs 5sigma in >30 min
        detected = 0
        for f in self._lc_model.filts:
            infilt = (dataSlice[self.filterCol] == f)
            t_filt = t[infilt]
            snr_filt = snr[infilt]
            if np.sum(snr_filt >= 5) >= 2:
                highsnrepochs = t_filt[snr_filt >= 5]
                #if you have 2 observations within 15 days of second peak
                if np.any(np.diff(np.sort(highsnrepochs[highsnrepochs < 15])) >= 0.5 / 24):
                  # and they are within 30 min or more
                    detected = 1
                    break
                
        # Default values for not-detected events
        characterization = 'uncharacterized'
        double_peak_detected = False
        
        if detected:
            #print("[DEBUG] start detecting")
            characterization = self.characterize_sce(snr, t, f, slice_point['file_indx'])
            if characterization == 'classical' and self.include_second_peak:
                #can you see a second peak in either filter
                for f in self._lc_model.filts:
                    #print("[DEBUG] filter", f)
                    mask = (dataSlice[self.filterCol] == f) 
                    t_filt = t[mask]  #time stamps in thet filter
                    snr_filt = snr[mask] #associated SNR
                    
                    if np.sum(snr_filt >= 3) >= 4: #need at least 4 points cause 3 are for sure before peak to be characterized
                        dur_rise   = self._lc_model.durations[f][
                            'rise'][slice_point['file_indx']]#[mask] 
                        dur_fade   = self._lc_model.durations[f][
                            'fade'][slice_point['file_indx']]#[mask]
                        dur_rerise = self._lc_model.durations[f][
                            'rerise'][slice_point['file_indx']]#[mask]
                    
                        t_second_rise = dur_rise + dur_fade + dur_rerise
                        
                        #request 1 3sig observation after second peak
                        second_rise = np.sum((t_filt > t_second_rise) & (snr_filt >= 3.0))
                        if second_rise >= 1:
                            double_peak_detected = True
                            break

        return {
            'detection': detected,
            'characterization': characterization,
            'double_peak': double_peak_detected
        }



# Metric Subclasses
class ShockCoolingType2bDetectMetric(BaseShockCoolingType2bMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'SC_Detect')
    def run(self, dataSlice, slice_point=None):
        return self.evaluate_sce(dataSlice, slice_point)['detection']

class ShockCoolingType2bCharacterizeMetric(BaseShockCoolingType2bMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'SC_characterize')
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_sce(dataSlice, slice_point)
        if result['detection'] == 0:
            return self.badval
        return 1 if result['characterization'] == 'classical' else 0

class ShockCoolingType2bClassicalMetric(BaseShockCoolingType2bMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'SC_classical')
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_sce(dataSlice, slice_point)
        if result['detection'] == 0:
            return self.badval
        return 1 if result['characterization'] == 'classical' else 0

class ShockCoolingType2bAmbiguousMetric(BaseShockCoolingType2bMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'SC_ambiguous')
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_sce(dataSlice, slice_point)
        if result['detection'] == 0:
            return self.badval
        return 1 if result['characterization'] == 'ambiguous' else 0


class ShockCoolingType2bUncharacterizedMetric(BaseShockCoolingType2bMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'SC_uncharacterized')
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_sce(dataSlice, slice_point)
        if result['detection'] == 0:
            return self.badval
        return 1 if result['characterization'] == 'uncharacterized' else 0
    
class ShockCoolingType2bDoublePeakMetric(BaseShockCoolingType2bMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'SC_doublepeaked')
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_sce(dataSlice, slice_point)
        if result['detection'] == 0:
            return self.badval
        return 1 if result.get('double_peak', False) else 0

# Updated utility for realistic uniform sky with WFD mask
def uniform_wfd_sky(n_points, mask_map, nside=64, seed=None):
    rng = np.random.default_rng(seed)
    ipix_all = np.arange(len(mask_map))
    ipix_wfd = ipix_all[mask_map > 0.5]  # Use mask threshold to define footprint
    selected_ipix = rng.choice(ipix_wfd, size=n_points, replace=True)
    theta, phi = hp.pix2ang(64, selected_ipix, nest=False)
    dec = 90 - np.degrees(theta)
    ra = np.degrees(phi)
    return ra, dec





# ============================================================
# Population Generator for Shock-Cooling Type IIb Events
# ============================================================
def generateShockCoolingType2bSlicer(t_start=1, t_end=3652,
                                     seed=42, rate_per_year=65_000, num_lightcurves=100,
                                     d_min=10, d_max=300,
                                     gal_lat_cut=None, save_to=None, load_from=None):
    """
    Generate slicer populated with simulated Shock Cooling SNe IIb events.
    Ensures declination values are within [-90, +90] and no NaNs are passed to healpy.
    """

    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
            slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
            slicer.slice_points.update(slice_data)
            print(f"Loaded SC population from {load_from}")
        return slicer
    
    rng = np.random.default_rng(seed)

    n_years = (t_end - t_start) / 365.25
    n_events = int(rate_per_year * n_years)
    print(f"Generating {n_events} SN IIb events from rate: {rate_per_year}/yr × {n_years:.2f} yr")

    
    #ra, dec = uniform_sphere_degrees(n_events, seed=seed) #returns degrees
    nside = 64  # Or 128 if you want higher resolution
    ra, dec = inject_uniform_healpix(nside=nside, n_events=n_events, seed=seed)

    #print(f"[CHECK] Dec range: {dec.min():.2f} to {dec.max():.2f} (expected ~[-90, 90])")

    dec = np.clip(dec, -89.9999, 89.9999)
    #dec_rad = np.radians(dec)
    
    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0) #returns radians 
    #print(f"Print 10 = {ra[:10],dec[:10]}")
    #print(f" Value = {slicer.slice_points}")

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
    print(f"[DEBUG]Print 5 sample before SkyCoord - ra,dec: {slicer.slice_points}")


    #coords = SkyCoord(ra=slicer.slice_points['ra'] * u.deg, dec=slicer.slice_points['dec'] * u.deg, frame='icrs') - this code just labels them as deg. u.deg doesn't convert them. 

    coords = SkyCoord(ra=np.degrees(slicer.slice_points['ra']) * u.deg, dec=np.degrees(slicer.slice_points['dec']) * u.deg, frame='icrs') #this line correctly converts them and labels them
    
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
            print(f"Saved SC population to {save_to}")

    lc_model = ShockCoolingLC()
    with open("ShockCooling_templates.pkl", "wb") as f:
        pickle.dump({'lightcurves': lc_model.data, 'durations': lc_model.durations}, f)
    print("Saved synthetic SCE light curve templates to ShockCooling_templates.pkl")


    return slicer

