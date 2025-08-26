from astropy.coordinates import Galactic, ICRS as ICRSFrame
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pickle
import os
import rubin_sim.maf.db as db
from collections import OrderedDict
import rubin_sim.maf.metric_bundles as metric_bundles
from rubin_sim.maf.metric_bundles import MetricBundle, MetricBundleGroup
import pandas as pd
from rubin_sim.phot_utils import DustValues
import rubin_sim.maf.metrics as metrics
from rubin_sim import maf
import shutil
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.cosmology import z_at_value
from rubin_sim.maf.slicers import UserPointsSlicer
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from pathlib import Path
from rubin_sim.maf.utils import m52snr
import tempfile




dust_model = DustValues()

# --------------------------------------------------
# Utility: Convert Galactic to Equatorial coordinates
# --------------------------------------------------
def equatorialFromGalactic(lon, lat):
    gal = Galactic(l=lon * u.deg, b=lat * u.deg)
    equ = gal.transform_to(ICRSFrame())
    return equ.ra.deg, equ.dec.deg

# -----------------------------------------------------------------------------
# Local Utility: Uniform sky injection
# -----------------------------------------------------------------------------
def uniform_sphere_degrees(n_points, seed=None):

    """
    Generate RA, Dec uniformly over the celestial sphere.

    Parameters
    ----------
    n_points : int
        Number of sky positions.
    seed : int or None
        Random seed.

    Returns
    -------
    ra : ndarray
        Right Ascension in degrees.
    dec : ndarray
        Declination in degrees.
    """
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n_points)
    z = rng.uniform(-1, 1, n_points)  # uniform in cos(theta)
    dec = np.degrees(np.arcsin(z))   # arcsin(z) gives uniform in solid angle
    
    """
    plt.figure(figsize=(8, 4))
    plt.scatter(ra, dec, s=1, alpha=0.3, label="Injected", color="black")
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title("Event Sky UniformSphere Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
    print("YAY! UNIFORM SPHERE!")
    return ra, dec

# --------------------------------------------
# Uniform Sphere Healpix
# --------------------------------------------
def inject_uniform_healpix(nside, n_events, seed=42):
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    pix = rng.choice(npix, size=n_events)
    theta, phi = hp.pix2ang(nside, pix)
    ra = np.degrees(phi)
    dec = np.degrees(0.5 * np.pi - theta)
    return ra, dec
# --------------------------------------------
# Updated utility for realistic uniform sky with WFD mask
# --------------------------------------------

def uniform_wfd_sky(n_points, mask_map, nside=64, seed=None):
    rng = np.random.default_rng(seed)
    ipix_all = np.arange(len(mask_map))
    ipix_wfd = ipix_all[mask_map > 0.5]  # Use mask threshold to define footprint
    selected_ipix = rng.choice(ipix_wfd, size=n_points, replace=True)
    theta, phi = hp.pix2ang(64, selected_ipix, nest=False)
    dec = 90 - np.degrees(theta)
    ra = np.degrees(phi)
    return ra, dec

# --------------------------------------------
# Plotting light curves from pkl file
# --------------------------------------------

def plot_some_lcs_from_pkl(templates_file, num=3, use_log_time=True, plot_overlap=False, ylim=None):
    '''
    Parameters
    ----------
    templates_file : str
        Path to the pickle file containing LC templates.
    num : int
        Number of light curves to plot (from beginning of file).
    use_log_time : bool
        Whether to use log(time) on the x-axis.
    plot_overlap : bool
        If True, plot all LCs on one figure; otherwise, one plot per LC.
    ylim : tuple or None
        y-axis limits as (ymin, ymax). If None, don't set.
    '''
    lcdict = pickle.load(open(templates_file, "rb"))
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    colors = {'u': 'k', 'g': 'b', 'r': 'g', 'i': 'r', 'z': 'magenta', 'y': 'yellow'}

    for i in range(num):
        if not plot_overlap:
            plt.figure()

        for f in filters:
            time_vals = lcdict["lightcurves"][i][f]['ph']
            if use_log_time:
                time_vals = np.log10(time_vals + 1e-5)

            plt.plot(
                time_vals,
                lcdict["lightcurves"][i][f]['mag'],
                color=colors[f],
                label=f if i == 0 else None,
                alpha=0.5
            )

        plt.title(f"Light Curve Template #{i}",size=20)
        plt.xlabel("Log Time (days)" if use_log_time else "Time (days)",size=15)
        plt.ylabel("Absolute Mag",size=15)
        plt.gca().invert_yaxis()
        if ylim is not None:
            plt.ylim(ylim)
        plt.legend(prop={'size': 15})

        if not plot_overlap:
            plt.show()

    if plot_overlap:
        plt.show()

# --------------------------------------------
# Helper function to apply either redshift or distance (directly)
# --------------------------------------------

def get_distance_bounds(d_min=None, d_max=None, z_min=None, z_max=None):
    """
    Return distance bounds in Mpc from either distance or redshift input.

    Parameters
    ----------
    d_min, d_max : float or None
        Distance bounds in Mpc.
    z_min, z_max : float or None
        Redshift bounds.

    Returns
    -------
    (d_min_Mpc, d_max_Mpc) : tuple of floats
    """

    # If distances are provided, return them directly
    if d_min is not None and d_max is not None:
        return d_min, d_max

    # Else convert redshifts to distances
    if z_min is not None and z_max is not None:
        d_min = cosmo.comoving_distance(z_min).to_value(u.Mpc)
        d_max = cosmo.comoving_distance(z_max).to_value(u.Mpc)
        print(f"[INFO] z_min = {z_min:.5f} → d_min = {d_min:.15f} Mpc")
        print(f"[INFO] z_max = {z_max:.5f} → d_max = {d_max:.15f} Mpc")
        return d_min, d_max

    raise ValueError("You must provide either (d_min, d_max) or (z_min, z_max)")

# --------------------------------------------
# Run detect metric
# --------------------------------------------

def run_detect(metric, slicer, cadences, shared_lc_model, db_dir, storage_dir, df_file, use_extinction, use_kcorrect, k_correct_type=None, k_correct_arg=None,is_grb=False, ignore_triples=False, debug=True, plot=True, clean_temp=False):
    '''
    Runs the detect metric on given cadences and light curves
    
    parameters:
    metric: python file with the metric information in it
    slicer: slicer
    cadences: list of cadences you want to use
    shared_lc_model: light curve templates
    db_dir: where your cadences are located
    storage_dir: where some output files from here will go
    ignore_triples: if you want to ignore triples in the cadence
    debug: if you want it to print some stuff
    plot: if you want it to plot stuff as it goes

    returns: a dataframe - the last one that is created (last cadence)
    for troubleshooting etc

    saves: three files
    df_obs.to_csv(f"output/ObsRecords_{cadence}.csv
    output.txt although we should change that
    outfile = os.path.join(storage_dir, f"local_efficiency_{cadence}.csv")
        with open(outfile, "w") as out:
            out.write("sid,n_filters_detected\n")
            for i in range(n_events):
                out.write(f"{i},{n_filters_detected_per_event[i]}\n")
    '''
    # print(f'changes made 1')
    n_events = len(slicer.slice_points['distance'])
    note = "scheduler_note not like 'long%'"
    is_grb = hasattr(metric, '__name__') and 'GRBafterglow' in metric.__name__
    for cadence in cadences:
        print(f"\n--- Running {cadence} ---")
        opsdb = os.path.join(db_dir, f"{cadence}.db")
        outDir = os.path.join(storage_dir, f"Metric_temp_{cadence}")
        os.makedirs(outDir, exist_ok=True)
        resultsDb = db.ResultsDb(out_dir=outDir)


        per_filter_metrics = OrderedDict()
        filters = ['all']
        for filt in filters:
            detect = metric.Detect_Metric(metricName=f"Detect_{filt}",
                                          lc_model=shared_lc_model,
                                          use_extinction=use_extinction, use_kcorrect=use_kcorrect, k_correct_type=k_correct_type, k_correct_arg=k_correct_arg)
            if ignore_triples:
                per_filter_metrics[f"Detect_{filt}"] = metric_bundles.MetricBundle(
                    detect, slicer, note)
            else:
                per_filter_metrics[f"Detect_{filt}"] = metric_bundles.MetricBundle(
                    detect, slicer, '')

        pf_group = metric_bundles.MetricBundleGroup(per_filter_metrics, opsdb,
                                                    out_dir=outDir, results_db=resultsDb)
        pf_group.run_all()

        bundle = per_filter_metrics["Detect_all"]
        detect_metric = bundle.metric
        obs_records = list(detect_metric.obs_records.values())

        if debug:
            print("\nInspecting one obs_record before saving to CSV:")
            sample_record = obs_records[0]
            for key, val in sample_record.items():
                print(f"{key:15} | type: {type(val)}", end='')
                try:
                    print(f" | length: {len(val)}")
                except TypeError:
                    print(f" | value: {val}")

        df_obs = pd.DataFrame.from_dict(detect_metric.obs_records).T.reset_index().rename(columns={"index": "sid"})
        max_index = len(shared_lc_model.data) - 1
        df_obs = df_obs[df_obs['file_indx'] <= max_index]

        #8/19
        # Store the injected peak magnitude (from slicer slice_points)
        for f in ['u', 'g', 'r', 'i', 'z', 'y']:
            keys = {
                'abs': f'peak_mag_abs_{f}',
                'noebv': f'peak_app_mag_noebv_{f}',
                'ebv': f'peak_app_mag_ebv_{f}'
            }
            for k, slicer_key in keys.items():
                if slicer_key in slicer.slice_points:
                    df_obs[f'injected_peak_{k}_mag_{f}'] = df_obs['sid'].map(lambda sid: slicer.slice_points[slicer_key][sid])
                else:
                    print(f"[WARNING] {slicer_key} not found in slice_points — skipping injected_peak_{k}_mag_{f}.")
        
        df_obs["year"] = (df_obs["peak_time"] / 365.25).astype(int) + 1
        df_detected_per_year = df_obs[df_obs['detected'] == True].groupby("year").size().reset_index(name="n_detected")

        for col in ['filter', 'mjd_obs', 'mag_obs', 'snr_obs']:
            df_obs[col] = df_obs[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            # Extract evaluated peak mag from obs_records
            df_obs['eval_peak_mag'] = df_obs['sid'].map(lambda sid: np.min(detect_metric.obs_records[sid]['mag_obs']) if 'mag_obs' in detect_metric.obs_records[sid] else np.nan)


        n_observations_detected = []
        n_filters_detected_per_event = []
        n_filters_detected_per_detected_event = []
        n_detected = 0
        peak_abs_mag_g = []
        alpha_fade_g = []
        t_jetbreak_g = []

        for i, row in df_obs.iterrows():
            file_indx = min(row['file_indx'], len(shared_lc_model.data) - 1)
            filt_arr = np.array(row["filter"])
            snr_arr = np.array(row["snr_obs"])
            good = snr_arr >= 5
            n_filters_detected_per_event.append(len(np.unique(filt_arr[good])))
            n_observations_detected.append(np.sum(good))

            if is_grb:
                peak_abs_mag_g.append(shared_lc_model.data[file_indx]['g']['mag'][0])
                alpha_fade_g.append(shared_lc_model.data[file_indx]['g']['mag'][1])
                t_jetbreak_g.append(shared_lc_model.data[file_indx]['g']['mag'][2])

            if row['detected']:
                n_filters_detected_per_detected_event.append(len(np.unique(filt_arr[good])))

        mean_filters = np.mean(n_filters_detected_per_detected_event)
        std_filters = np.std(n_filters_detected_per_detected_event)
        n_detected = np.sum(df_obs['detected'])

        print(f"Out of {n_events} simulated events, with {len(obs_records)} in visible positions, Rubin detected {n_detected} under the {cadence} cadence.")
        print(f"Of those, each event was observed in an average of {mean_filters:.1f} ± {std_filters:.1f} filters.")

        df_obs['n_observations_detected'] = n_observations_detected
        df_obs['n_filters_detected'] = n_filters_detected_per_event

        if is_grb:
            df_obs['peak_abs_mag_g'] = peak_abs_mag_g
            df_obs['alpha_fade_g'] = alpha_fade_g
            df_obs['t_jetbreak_g'] = t_jetbreak_g
            df_obs['peak_apparent_mag_g_noebv'] = df_obs['peak_abs_mag_g'] + df_obs['distance_modulus']

        df_obs.to_csv(df_file+f"ObsRecords_{cadence}.csv", index=False)
        print("Obs_Record dataframe saved to", df_file+f"ObsRecords_{cadence}.csv")

        if plot:
            filtername = 'r'
            ax1 = DustValues().ax1
            ras, decs, peak_mags, detected_flags = [], [], [], []

            for i in range(n_events):
                ra = slicer.slice_points['ra'][i]
                dec = slicer.slice_points['dec'][i]
                d = slicer.slice_points['distance'][i]
                ebv = slicer.slice_points['ebv'][i]
                file_indx = min(slicer.slice_points['file_indx'][i], len(shared_lc_model.data) - 1)
                try:
                    m_peak = np.min(shared_lc_model.data[file_indx][filtername]['mag'])
                except:
                    m_peak = 99.0
                A = ax1[filtername] * ebv
                dm = 5 * np.log10(d * 1e6) - 5
                m_app = m_peak + dm + A

                ras.append(ra)
                decs.append(dec)
                peak_mags.append(m_app)
                detected = any(
                    per_filter_metrics[f"Detect_{f}"].metric_values[i] == 1
                    and not per_filter_metrics[f"Detect_{f}"].metric_values.mask[i]
                    for f in filters
                )
                detected_flags.append(detected)

            plt.figure(figsize=(8, 4))
            plt.scatter(ras, peak_mags, c='black', s=10, label='Injected', alpha=0.6)
            plt.scatter(np.array(ras)[detected_flags], np.array(peak_mags)[detected_flags],
                        c='red', s=20, label='Detected', alpha=0.9, edgecolors='black')
            plt.xlabel("RA [rad]")
            plt.ylabel(f"Apparent Peak Magnitude ({filtername}-band)")
            plt.title(f"{cadence} – Apparent Mag vs RA")
            plt.gca().invert_yaxis()
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 4))
            plt.scatter(decs, peak_mags, c='black', s=10, label='Injected', alpha=0.6)
            plt.scatter(np.array(decs)[detected_flags], np.array(peak_mags)[detected_flags],
                        c='red', s=20, label='Detected', alpha=0.9, edgecolors='black')
            plt.xlabel("Dec [rad]")
            plt.ylabel(f"Apparent Peak Magnitude ({filtername}-band)")
            plt.title(f"{cadence} – Apparent Mag vs Dec")
            plt.gca().invert_yaxis()
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 4))
            plt.hist(df_obs["year"], bins=np.arange(0.5, 11.5, 1), edgecolor='black')
            plt.xticks(ticks=np.arange(1, 11), labels=[f"Year {i}" for i in range(1, 11)])
            plt.xlabel("Survey Year")
            plt.ylabel("Number of Events")
            plt.title("Distribution of Peak Times")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 4))
            plt.bar(df_detected_per_year["year"], df_detected_per_year["n_detected"],
                    width=0.7, align='center', edgecolor='black')
            plt.xticks(ticks=np.arange(1, 11), labels=[f"Year {i}" for i in range(1, 11)])
            plt.xlabel("Survey Year")
            plt.ylabel("Number of Detections")
            plt.title("Distribution of DETECTED Peak Times")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 4))
            plt.hist(np.degrees(slicer.slice_points['dec']), bins=50, alpha=0.5, label='Injected')
            plt.hist(np.degrees(np.array(decs)[detected_flags]), bins=50, alpha=0.8, label='Detected', color='red')
            plt.xlabel("Declination [deg]")
            plt.ylabel("Number of Events")
            plt.title(f"{cadence} – Declination Distribution")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        outfile = os.path.join(storage_dir, f"local_efficiency_{cadence}.csv")
        with open(outfile, "w") as out:
            out.write("sid,n_filters_detected\n")
            for i in range(len(df_obs)):
                out.write(f"{i},{n_filters_detected_per_event[i]}\n")

        if clean_temp:
            print(f"[CLEANUP] Removing temp directory: {outDir}")
            shutil.rmtree(outDir, ignore_errors=True)

    return df_obs


# --------------------------------------------
# Run several metrics
# --------------------------------------------



def run_multi_metrics(multi_metrics, slicer, cadences, shared_lc_model, db_dir, storage_dir, summary_filename, ignore_triples=False, plot=True, clean_temp=False, use_extinction=True, use_kcorrect=True):
    '''
    Runs the detect metric on given cadences and light curves
    
    parameters:
    multi_metrics: list of metrics to run
    slicer: slicer
    cadences: list of cadences you want to use
    shared_lc_model: light curve templates
    db_dir: where your cadences are located
    storage_dir: where some output files from here will go
    ignore_triples: if you want to ignore triples in the cadence
    plot: if you want it to plot stuff as it goes

    returns: a dataframe with results

    saves: that dataframe
    '''
    first = 1
    n_events = len(slicer.slice_points['distance'])
    note = "scheduler_note not like 'long%'" #for if we want to avoid triples
    for cadence in cadences:
        runName = cadence
        opsdb = os.path.join(db_dir, f"{cadence}.db")
        outDir = os.path.join(storage_dir, f"Metric_temp_{cadence}")
        os.makedirs(outDir, exist_ok=True)
        resultsDb = db.ResultsDb(out_dir=outDir)
        
        print(f"\n--- Running {cadence} ---")
        print(np.shape(multi_metrics))
        print(multi_metrics)

        for one_metric in multi_metrics:
            # print("We are in one_metric")
            mb_key = f"{runName}_{one_metric.__class__.__name__}"
            if ignore_triples == True:
                bundle = metric_bundles.MetricBundle(one_metric, slicer, '' + note, file_root=mb_key, plot_funcs=[], summary_metrics=[metrics.SumMetric()])
            else:
                bundle = metric_bundles.MetricBundle(one_metric, slicer, '', file_root=mb_key, plot_funcs=[], summary_metrics=[metrics.SumMetric()])
            
            bd = maf.metricBundles.make_bundles_dict_from_list([bundle])
            bgroup = metric_bundles.MetricBundleGroup({mb_key: bundle}, opsdb, out_dir=outDir, results_db=resultsDb)
            bgroup.run_all()
            # print("We just ran all")

            if first:
                # print("We out here")
                df = pd.DataFrame([bd[k].summary_values for k in bd], index=list(bd.keys()))
                df["run"] = runName
                df["n_events_full_sky"] =  n_events  
                first = 0
                print(df)
            else:
                # print("Now in else")
                _ = pd.DataFrame([bd[k].summary_values for k in bd], index=list(bd.keys()))
                _["run"] = runName
                _["n_events_full_sky"] =  n_events              
                df = pd.concat([df, _])
        # Healpix plotting

            if plot == True:
                # Plot: Apparent magnitude vs RA and Dec for one filter (e.g. 'r')
                filtername = 'r'
                ax1 = DustValues().ax1
                 
                ras, decs, peak_mags, detected_flags = [], [], [], []
             
                for i in range(n_events):
                    ra = slicer.slice_points['ra'][i]
                    dec = slicer.slice_points['dec'][i]  # this is in radians already
                    d = slicer.slice_points['distance'][i]
                    ebv = slicer.slice_points['ebv'][i]
                    file_indx = min(slicer.slice_points['file_indx'][i], len(shared_lc_model.data) - 1)
                    
                    m_peak = np.min(shared_lc_model.data[file_indx][filtername]['mag'])
                    A = ax1[filtername] * ebv
                    dm = 5 * np.log10(d * 1e6) - 5
                    m_app = m_peak + dm + A
                 
                    ras.append(ra)
                    decs.append(dec)
                    peak_mags.append(m_app)
                 

                
                if plot == True:
                    nside = slicer.nside if hasattr(slicer, 'nside') else 64
                    npix = hp.nside2npix(nside)
                    injected_map = np.zeros(npix)
                    detected_map = np.zeros(npix)
            
                    ra_rad = slicer.slice_points['ra']
                    dec_rad = slicer.slice_points['dec']
                    theta = 0.5 * np.pi - dec_rad
                    phi = ra_rad
                    pix_inds = hp.ang2pix(nside, theta, phi)
            
                    #print(f"[DEBUG] RA range [rad]: {ra_rad.min():.2f} – {ra_rad.max():.2f}")
                    #print(f"[DEBUG] Dec range [rad]: {dec_rad.min():.2f} – {dec_rad.max():.2f}")
                    #print(f"[DEBUG] Dec range [deg]: {np.degrees(dec_rad).min():.2f} – {np.degrees(dec_rad).max():.2f}")
                    
                    for i, pix in enumerate(pix_inds):
                        injected_map[pix] += 1
                        #if detected_flags[i] :
                        if bundle.metric_values[i] == 1:
                            if np.random.rand() < 0.001:
                                print(f"[DEBUG] Detected RA, Dec: {np.degrees(ra_rad[i]):.2f}, {np.degrees(dec_rad[i]):.2f}")
                            detected_map[pix] += 1
            
                    eff_map = np.zeros(npix)
                    mask = injected_map > 0
                    eff_map[mask] = detected_map[mask] / injected_map[mask]
                    eff_map[~mask] = hp.UNSEEN
            
                    hp.mollview(eff_map, title=f"{runName} – {one_metric.metricName} Efficiency", unit='Efficiency', cmap='viridis')
                    hp.graticule()
                    plt.show()

        if clean_temp:
            for cadence in cadences:
                outDir = os.path.join(storage_dir, f"Metric_temp_{cadence}")
                print(f"[CLEANUP] Removing temp directory: {outDir}")
                shutil.rmtree(outDir, ignore_errors=True)
    df.to_csv(summary_filename)
    print("saved summary to ",summary_filename)
    return df




# --------------------------------------------
# Volumetric rate model (for GRBs, on-axis ≈ 10⁻⁹ Mpc⁻³ yr⁻¹)
# --------------------------------------------
def sample_rate_from_volume(rate_density, t_start, t_end, 
                                d_min=None, d_max=None,
                                z_min=None, z_max=None): #1e-8 for GRBs to account for dirty fireball and off axis, 1e-9 without
    """
    Estimate the number of event from comoving volume and volumetric rate.

    Parameters
    ----------
    t_start : float
        Start of the time window (days).
    t_end : float
        End of the time window (days).
    d_min : float
        Minimum luminosity distance in Mpc.
    d_max : float
        Maximum luminosity distance in Mpc.
    rate_density : float
        Volumetric event rate in events/Mpc^3/yr.

    Returns
    -------
    int
        Expected number of events in the survey.
    """

    d_min, d_max = get_distance_bounds(d_min=d_min, d_max=d_max, z_min=z_min, z_max=z_max)
    years = (t_end - t_start) / 365.25
    if d_max > 1:
        z_min = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
        z_max = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)    
        V = cosmo.comoving_volume(z_max).to(u.Mpc**3).value - cosmo.comoving_volume(z_min).to(u.Mpc**3).value
    else: #for things that are close, regular volume
        V = ((4/3)*np.pi*(d_max**3 - d_min**3))
    return np.random.poisson(rate_density * V * years)


# --------------------------------------------
# Population Loader (used in scripts)
# --------------------------------------------
#8/26 updated for atomic saving 

def load_or_generate_population(use_extinction, pop_file, lc_model=None, t_start=1, t_end=3652, seed=42,
                                d_min=None, d_max=None,
                                z_min=None, z_max=None,
                                num_lightcurves=1000,
                                gal_lat_cut=None, rate_density=1e-8,
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
        print(f"[INFO] Generating population and saving to {pop_file}")
        slicer = generate_PopSlicer(use_extinction=use_extinction,
                                    lc_model=lc_model,
                                    t_start=t_start, t_end=t_end,
                                    d_min=d_min, d_max=d_max, z_min=z_min, z_max=z_max,
                                    seed=seed, num_lightcurves=num_lightcurves,
                                    gal_lat_cut=gal_lat_cut, rate_density=rate_density,
                                    save_to=pop_file,               # ensure atomic save on generate
                                    make_debug_plots=make_debug_plots)
    else:
        print(f"[INFO] Loading population from {pop_file}")
        slicer = generate_PopSlicer(use_extinction=use_extinction,
                                    lc_model=lc_model,               # <-- CRITICAL
                                    load_from=pop_file,
                                    save_to=pop_file,                # re-save atomically if backfilled
                                    make_debug_plots=False)
    return slicer

    
# --------------------------------------------
# Population generator
# --------------------------------------------

#8/26 cleaned up this function from old material, added atomic saving unit

def generate_PopSlicer(use_extinction, lc_model=None, t_start=1, t_end=3652, seed=42,
                         d_min=None, d_max=None, z_min = None, z_max = None, num_lightcurves=1000, 
                       gal_lat_cut=None, rate_density=None, 
                         load_from=None, save_to=None, make_debug_plots=True):
    """
    Generate or load a population of events. When loading:
      - if peak columns are missing and lc_model is provided, they are backfilled.
      - if 'save_to' is set and any backfill occurred, the upgraded file is saved atomically.
    When generating new:
      - population is written with an atomic pickle if 'save_to' is provided.
      
    Parameters
    ----------
    use_extinction: whether to use extinction (Bool)
    gal_lat_cut : float or None
        Optional Galactic latitude cut (e.g., 15 deg).
    load_from : str or None
        If set, load slice_points from this pickle file.
    save_to : str or None
        If set, save the slice_points to this pickle file.
    make_debug_plots : True or anything else
        if true, will plot various distributions and print some stuff
    """
    # -------- LOAD PATH --------

    if load_from and os.path.exists(load_from):
        # [NEW] robust load + auto-backfill + optional atomic re-save
        try:
            with open(load_from, 'rb') as f:
                slice_data = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {load_from} ({e}). Regenerating population instead.")
        else:
            slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
            slicer.slice_points.update(slice_data)
            print(f"Loaded population from {load_from}")

            backfilled = False
            try:
                before = set(slicer.slice_points.keys())
                # auto-backfill only if lc_model is provided
                if lc_model is not None:
                    slicer = backfill_injected_peaks_if_missing(
                        slicer, lc_model=lc_model, use_extinction=use_extinction
                    )
                after = set(slicer.slice_points.keys())
                backfilled = (after != before)
            except Exception as bf_e:
                print(f"[WARN] Backfill skipped/failed ({bf_e}).")

            # Persist upgraded file if requested
            if backfilled and save_to:
                atomic_save_pickle(dict(slicer.slice_points), save_to)

            return slicer
    # -------- END LOAD PATH --------

    rng = np.random.default_rng(seed)
    d_min, d_max = get_distance_bounds(d_min=d_min, d_max=d_max, z_min=z_min, z_max=z_max)
    n_events = sample_rate_from_volume(t_start=t_start, t_end=t_end, d_min=d_min, d_max=d_max, 
                                       z_min=z_min, z_max=z_max, rate_density=rate_density)
    print(f"Simulated {n_events} events using rate_density = {rate_density:.1e}")

    nside = 64  # Or 128 if you want higher resolution
    ra, dec = inject_uniform_healpix(nside=nside, n_events=n_events, seed=seed)

    dec = np.clip(dec, -89.9999, 89.9999)
    coords = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')

    print("Len ra before masking: ", len(ra))
    if gal_lat_cut is not None:
        b = coords.galactic.b.deg
        print("b: ", b)
        mask = np.abs(b) < gal_lat_cut  # shar switched this to less
        print("len mask, num true in mask: ", len(mask), np.sum(mask))
        ra, dec = ra[mask], dec[mask]
        coords = coords[mask]
        print("len ra after masking: ", len(ra))
    
    n_events = len(ra)
    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)  # returns radians 
    if make_debug_plots==True:
        plt.hist(slicer.slice_points['ra'], bins=50)
        plt.xlabel("RA [rad]")
        plt.title("Injected Population – RA Distribution")
        plt.grid(True)
        plt.show()
        
        plt.hist(slicer.slice_points['dec'], bins=50)
        plt.xlabel("Dec [rad]")
        plt.title("Injected Population – Dec Distribution")
        plt.grid(True)
        plt.show()

    distances = rng.uniform(d_min, d_max, n_events)
    peak_times = rng.uniform(t_start, t_end, n_events)
    print("initial peak times: ", peak_times)
    file_indx = rng.integers(0, num_lightcurves, len(ra))

    sfd = SFDQuery()
    if use_extinction:
        ebv_vals = sfd(coords)
    else:
        ebv_vals = np.zeros(len(distances))

    slicer.slice_points['distance'] = distances
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    slicer.slice_points['gall'] = coords.galactic.l.deg
    slicer.slice_points['galb'] = coords.galactic.b.deg

    if lc_model is not None:
        filters = ['u', 'g', 'r', 'i', 'z', 'y']
        ax1 = dust_model.ax1  # extinction coefficients per filter
    
        # Initialize per-filter storage
        peak_mag_abs = {}
        peak_app_mag_noebv = {}
        peak_app_mag_ebv = {}
    
        for f in filters:
            peak_mag_abs[f] = []
            peak_app_mag_noebv[f] = []
            peak_app_mag_ebv[f] = []
    
        distance_modulus_list = []
    
        for idx, dist, ebv in zip(file_indx, distances, ebv_vals):
            dm = 5 * np.log10(dist * 1e6) - 5
            distance_modulus_list.append(dm)
    
            for f in filters:
                m_abs = np.min(lc_model.data[idx][f]['mag'])  # absolute peak
                A = ax1[f] * ebv
    
                m_noebv = m_abs + dm
                m_with_ebv = m_noebv + A
    
                peak_mag_abs[f].append(m_abs)
                peak_app_mag_noebv[f].append(m_noebv)
                peak_app_mag_ebv[f].append(m_with_ebv)
    
        # Save to slice_points
        for f in filters:
            slicer.slice_points[f'peak_mag_abs_{f}'] = np.array(peak_mag_abs[f])
            slicer.slice_points[f'peak_app_mag_noebv_{f}'] = np.array(peak_app_mag_noebv[f])
            slicer.slice_points[f'peak_app_mag_ebv_{f}'] = np.array(peak_app_mag_ebv[f])
    
        slicer.slice_points['distance_modulus'] = np.array(distance_modulus_list)
    else:
        print("[WARNING] lc_model not provided - skipping injected peak magnitude storage.")
    
    print("gal_lat_cut is none")

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
    
    if save_to:
        # [NEW] atomic save instead of raw pickle.dump
        atomic_save_pickle(dict(slicer.slice_points), save_to)

    print(f"DEBUG: type(peak_times) = {type(peak_times)}")
    print(f"DEBUG: shape(peak_times) = {peak_times.shape}")

    return slicer

# --------------------------------------------
# Filename builder for saving outputs
# --------------------------------------------

def build_filenames(rate_density, 
                        z_min, 
                        z_max,
                        d_min, 
                        d_max,
                        science_case, #"GRBafterglows" for instance
                        testname=None,
                        testname_metric_only=None,
                        ignore_triples=None,                   
                        use_extinction=None,
                        use_kcorrect=None,
                        base_dir=None):
    """
    Construct filenames for saving templates, filename, output dataframe, storage_dir, summary_filename

    Parameters
    ----------


    Returns
    -------
    four strs : path for templates, filename, output dataframe, storage_dir, summary_filename
    """

    if ignore_triples==True:
        testname_metric_only = str(testname_metric_only)+"_it_"+str(ignore_triples)

    #shar
    if base_dir==None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
        
    label = science_case + f"_den_{rate_density}_d_{d_min}-{d_max}_Mpc_z_{z_min}-{z_max}_Mpc_ext_{use_extinction}_kcor_{use_kcorrect}_{testname}"
    print(label)

    storage_dir = os.path.join(base_dir, science_case)
    templates_file = os.path.join(storage_dir, label+"_templates.pkl")
    pop_file = os.path.join(storage_dir, label+"_population.pkl")
    df_file = os.path.join(storage_dir, label+f"_{testname_metric_only}_obs_record")
    summary_filename = os.path.join(storage_dir, label+f"_{testname_metric_only}_multi_summary.csv")
    
    return templates_file, pop_file, df_file, storage_dir, summary_filename



# --------------------------------------------------
# Light Curve Template Generator (Separate from Population)
# --------------------------------------------------
def generate_Templates(LC, save_to,
    num_samples=100, num_lightcurves=1000
):
    """
    Generate synthetic light curve templates and save to file.
    """
    # Create the directory if it doesn't exist
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    
    lc_model = LC(num_lightcurves=num_lightcurves, load_from=None)
    with open(save_to, "wb") as f:
        pickle.dump({'lightcurves': lc_model.data, 't_grid': lc_model.t_grid}, f)

    print(f"Saved synthetic light curve templates to {save_to}")

# --------------------------------------------
# Light Curve Loader (used in scripts)
# --------------------------------------------
def load_or_generate_templates(LC, templates_file,
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
        generate_Templates(LC, 
                           save_to=templates_file, 
                           num_samples=num_samples,
                           num_lightcurves=num_lightcurves
                           )
    else:
        print(f"[INFO] Loading light curve templates from {templates_file}.")
    return LC(load_from=templates_file)

# -------------------------------------------
# Applying correlation between filters 
# -------------------------------------------

def apply_spectral_index(mag_ref, filtername, ref_filter="Rc", beta=-0.75):
    """
    This function adjusts a magnitude from R_c to any Rubin filter assuming a power-law SED where F_ν ∝ ν^β.

    We're working in AB magnitudes, so we don't need to convert to flux first — the zero-points cancel out.
    The difference between magnitudes in two bands corresponds directly to the spectral slope:

    Δm = 2.5 * β * log10(ν_target / ν_ref)

    This is based on Cenko and other GRB afterglow spectral energy distributions that assume a constant β. We assume the reference
    band is R_c, and the correction is applied to get estimated peak magnitudes in ugrizy.

    Rc_freq = c / wavelength_Rc = 2.998e17 / 658

    Parameters
    ----------
    mag_ref : float or array_like
        Magnitude in the reference filter (e.g., r-band).
    filtername : str
        Target LSST filter (e.g., 'g', 'i').
    ref_filter : str or None
        Reference filter used to simulate the light curve. Default is from config.
    beta : float or None
        Spectral index. If None, uses value from GRB_CONFIG.

    Returns
    -------
    mag_target : float or array_like
        Adjusted magnitude in the target filter.
    """
    FILTER_CENTRAL_FREQS = {
        'Rc': 2.99792458e17 / 658.0,  # Hz, R_c band
        'u': 8.088e14,
        'g': 6.293e14,
        'r': 4.844e14,
        'i': 3.979e14,
        'z': 3.461e14,
        'y': 3.080e14,
    }

    if filtername == ref_filter:
        return mag_ref

    nu_ref = FILTER_CENTRAL_FREQS[ref_filter]
    nu_target = FILTER_CENTRAL_FREQS[filtername]
    delta_mag = 2.5 * beta * np.log10(nu_target / nu_ref)
    return mag_ref + delta_mag
    
# -------------------------------------------
# Evaluate light curve 
# -------------------------------------------

#organized 8/25
def evaluate(self, dataSlice, slice_point, return_full_obs=True):
    """
    Evaluate the model light curve for an event at observed times and filters. 
    Interpolates the template LC, applies distance modulus, extinction, and optional 
    k-corrections to yield apparent magnitudes and SNRs. If diagnostic sampling is 
    enabled (via diag_*), a small subsample of per-visit mags/SNRs/MJDs/filters is 
    stored for QA without large memory overhead.

    Parameters
    ----------
    dataSlice : np.ndarray
        Subset of the simulated survey database containing observation times, filters, 
        and depth information for this sky location.
    slice_point : dict
        Metadata for the injected transient (e.g., distance, extinction, template index).
    return_full_obs : bool, default=True
        If True, return a dictionary of per-visit quantities (magnitudes, SNRs, filters, MJDs).
        If False, return only arrays of SNR, filters, and relative times.
    
    Returns
    -------
    snr : np.ndarray
        Signal-to-noise ratios for each observation.
    filters : np.ndarray
        Filters corresponding to each observation.
    times : np.ndarray
        Relative times (days since transient peak) for each observation.
    obs_record : dict or None
        Per-visit quantities for this event, including:
            - 'mjd_obs': observation MJDs
            - 'mag_obs': apparent magnitudes (after DM, EBV, k-correction if enabled)
            - 'snr_obs': per-visit signal-to-noise ratios
            - 'filter' : filter codes for each observation
          If diagnostic sampling is active, may also include:
            - 'diag_sample_mjd', 'diag_sample_mag', 'diag_sample_snr', 'diag_sample_filter'
              (subsampled visits retained for QA)
        Returns None if return_full_obs=False.
    """
    # Phase relative to peak (days)
    t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']

    # Allocate magnitude array (apparent mags)
    mags = np.zeros(t.size)

    # Cache
    filters = dataSlice[self.filterCol]
    m5 = dataSlice[self.m5Col]
    dm = 5 * np.log10(slice_point['distance'] * 1e6) - 5  # distance modulus

    # Precompute redshift once per event if using k-corrections
    if self.use_kcorrect:
        z = z_at_value(cosmo.comoving_distance, slice_point['distance'] * u.Mpc)

    # Per-filter evaluation at observed epochs
    for f in np.unique(filters):
        infilt = np.where(filters == f)[0]

        # template → observed times
        mags[infilt] = self.lc_model.interp(t[infilt], f, slice_point['file_indx'])

        # K-correction (if enabled)
        if self.use_kcorrect:
            k_correction = apply_kcorrection(z, f, self.k_correct_type, self.k_correct_arg)
            mags[infilt] += k_correction

        # Galactic extinction (if enabled)
        if self.use_extinction:
            mags[infilt] += self.ax1[f] * slice_point['ebv']
            if not self.extinction_printed:
                print("EBV included")
                self.extinction_printed = True

        # Distance modulus
        mags[infilt] += dm

    # LSST single-visit SNR at each observation
    snr = m52snr(mags, m5)
    times = t

    if not return_full_obs:
        return snr, filters, times, None

    # Build obs_record
    obs_record = {
        'mjd_obs': dataSlice[self.mjdCol],
        'mag_obs': mags,
        'snr_obs': snr,
        'filter': filters
    }

    # --- Lightweight diagnostic sampling of per-visit mags (optional) ---
    if getattr(self, 'diag_store', False):
        # Ensure an RNG exists
        if not hasattr(self, '_rng') or self._rng is None:
            self._rng = np.random.default_rng(12345)

        keep_mask = np.ones_like(snr, dtype=bool)

        # Gating (optional)
        if getattr(self, 'diag_min_snr', None) is not None:
            keep_mask &= (snr >= self.diag_min_snr)
        if getattr(self, 'diag_max_mag', None) is not None:
            keep_mask &= (mags <= self.diag_max_mag)

        # Bernoulli thinning
        p = float(np.clip(getattr(self, 'diag_sample_rate', 0.0), 0.0, 1.0))
        if p > 0.0:
            rnd = self._rng.random(size=keep_mask.size)
            keep_mask &= (rnd < p)

        idx = np.where(keep_mask)[0]

        # Per-event hard cap
        cap = int(getattr(self, 'diag_per_event_cap', 0))
        if cap > 0 and idx.size > cap:
            idx = self._rng.choice(idx, size=cap, replace=False)

        # Attach samples (empty lists if none)
        obs_record['diag_sample_mjd']    = dataSlice[self.mjdCol][idx].tolist() if idx.size else []
        obs_record['diag_sample_mag']    = mags[idx].tolist() if idx.size else []
        obs_record['diag_sample_snr']    = snr[idx].tolist() if idx.size else []
        obs_record['diag_sample_filter'] = filters[idx].tolist() if idx.size else []
    # --------------------------------------------------------------------

    return snr, filters, times, obs_record
    

    if return_full_obs:
        obs_record = {
            'mjd_obs': dataSlice[self.mjdCol],
            'mag_obs': mags,
            'snr_obs': snr,
            'filter': filters
        }
        
        return snr, filters, times, obs_record
    print("DID YOU NOT RETURN THE OBS RECORD ON PURPOSE??")
    return snr, filters, times, None


FILTER_CENTRAL_FREQS = {
    'u': 8.088e14,
    'g': 6.293e14,
    'r': 4.844e14,
    'i': 3.979e14,
    'z': 3.461e14,
    'y': 3.080e14,
}

def get_kcorrection_powerlaw(z, obs_filter, spectral_index):
    """
    Calculate k-correction for power-law spectrum: f_ν ∝ ν^α
    
    For a pure power law, k-correction is the same for all filters.
    This is because the power law is scale-invariant.
    
    Parameters
    ----------
    z : float
        Redshift
    obs_filter : str
        Observed filter ('u', 'g', 'r', 'i', 'z', 'y')
    spectral_index : float
        Power-law spectral index α where f_ν ∝ ν^α
        For GRB afterglows: α ≈ -0.75 (flux increases with frequency)
        
    Returns
    -------
    k_correction : float
        K-correction in magnitudes (positive = dimmer)
    """
    # For pure power law f_ν ∝ ν^α:
    # K = 2.5 * (1 - α) * log10(1 + z)
    # This accounts for both spectral slope and cosmological dimming
    
    k_corr = 2.5 * (1 - spectral_index) * np.log10(1 + z)
    
    return k_corr


def get_kcorrection_blackbody(z, obs_filter, temperature):
    """
    Calculate k-correction for blackbody spectrum.
    
    Parameters
    ----------
    z : float or array
        Redshift
    obs_filter : str
        Observed filter
    temperature : float
        Blackbody temperature in Kelvin
        
    Returns
    -------
    k_correction : float
        K-correction in magnitudes
    """
    # print(z)
    # Physical constants
    h = 6.626e-34  # J⋅s
    k_b = 1.381e-23  # J/K
    c = 2.998e8  # m/s
    
    # Get observed frequency
    nu_obs = FILTER_CENTRAL_FREQS[obs_filter]
    
    # Rest-frame frequency being observed
    nu_rest = nu_obs * (1 + z)
    
    # Planck function ratio: B_ν(rest) / B_ν(obs)
    x_rest = h * nu_rest / (k_b * temperature)
    # print(x_rest)
    x_obs = h * nu_obs / (k_b * temperature)
    
    if type(z)==list or type(z)==np.ndarray:
        flux_ratio = np.zeros(len(x_rest))
        # Overflow protection: if x > ~700, exp(x) overflows
        for i in range(len(x_rest)):
            if x_rest[i]> 700 or x_obs > 700:
                # print("using Wien approximation")
                # Use Wien approximation: B_ν ∝ ν³ exp(-x)
                flux_ratio[i] = (nu_rest[i]/nu_obs)**3 * np.exp(x_obs - x_rest[i])
            else:
                # print("using planck function")
                planck_rest = nu_rest[i]**3 / (np.exp(x_rest[i]) - 1)
                planck_obs = nu_obs**3 / (np.exp(x_obs) - 1)
                flux_ratio[i] = planck_rest / planck_obs
    else:
        flux_ratio = 0
        # Overflow protection: if x > ~700, exp(x) overflows
        if x_rest> 700 or x_obs > 700:
            # print("using Wien approximation")
            # Use Wien approximation: B_ν ∝ ν³ exp(-x)
            flux_ratio = (nu_rest/nu_obs)**3 * np.exp(x_obs - x_rest)
        else:
            # print("using planck function")
            planck_rest = nu_rest**3 / (np.exp(x_rest) - 1)
            planck_obs = nu_obs**3 / (np.exp(x_obs) - 1)
            flux_ratio = planck_rest / planck_obs       

        

    
    # Convert to magnitude difference
    k_corr = -2.5 * np.log10(flux_ratio)
    
    # Add cosmological (1+z) dimming factor
    k_corr += 2.5 * np.log10(1 + z)
    # print(k_corr)
    return k_corr


def apply_kcorrection(z, obs_filter, spectrum_type, k_correct_arg):
    """
    Unified function to apply k-correction based on spectrum type.
    
    Parameters
    ----------
    z : float
        Redshift
    obs_filter : str
        Observed filter
    spectrum_type : str
        'powerlaw' or 'blackbody'
    **kwargs : 
        For powerlaw: spectral_index (float)
        For blackbody: temperature (float, Kelvin)
        
    Returns
    -------
    k_correction : float
        K-correction in magnitudes
    """
    if spectrum_type == 'powerlaw':
        # if 'spectral_index' not in kwargs:
        #     raise ValueError("Must provide 'spectral_index' for powerlaw spectrum")
        return get_kcorrection_powerlaw(z, obs_filter, k_correct_arg)
    
    elif spectrum_type == 'blackbody':
        # if 'temperature' not in kwargs:
        #     raise ValueError("Must provide 'temperature' for blackbody spectrum")
        return get_kcorrection_blackbody(z, obs_filter, k_correct_arg)
    
    else:
        raise ValueError(f"Unknown spectrum_type: {spectrum_type}")

#8/25
def backfill_injected_peaks_if_missing(slicer, lc_model, use_extinction=True):
    """
    Ensure slicer.slice_points contains:
      - peak_mag_abs_{f}
      - peak_app_mag_noebv_{f}
      - peak_app_mag_ebv_{f}
      - distance_modulus
    for f in ugrizy. Computes and inserts them if absent.
    """
    filters = ['u','g','r','i','z','y']
    need_backfill = any(
        f'peak_mag_abs_{f}' not in slicer.slice_points or
        f'peak_app_mag_noebv_{f}' not in slicer.slice_points or
        f'peak_app_mag_ebv_{f}' not in slicer.slice_points
        for f in filters
    )

    if not need_backfill and 'distance_modulus' in slicer.slice_points:
        return slicer  # nothing to do

    if lc_model is None:
        raise ValueError("lc_model is required to backfill injected peaks.")

    # Gather event-level quantities
    file_indx = np.asarray(slicer.slice_points['file_indx'])
    dist = np.asarray(slicer.slice_points['distance'])  # Mpc
    ebv = np.asarray(slicer.slice_points.get('ebv', np.zeros_like(dist)))

    dm = 5 * np.log10(dist * 1e6) - 5
    slicer.slice_points['distance_modulus'] = dm

    ax1 = dust_model.ax1  # extinction coefficients

    # Build arrays per filter
    for f in filters:
        # absolute template peak in this filter from lc_model
        m_abs = np.array([np.min(lc_model.data[idx][f]['mag']) for idx in file_indx])

        slicer.slice_points[f'peak_mag_abs_{f}'] = m_abs
        m_noebv = m_abs + dm
        slicer.slice_points[f'peak_app_mag_noebv_{f}'] = m_noebv

        if use_extinction:
            A = ax1[f] * ebv
        else:
            A = 0.0
        slicer.slice_points[f'peak_app_mag_ebv_{f}'] = m_noebv + A

    return slicer
    
#8/25
def plot_population_lcs(pop_file,
                        lc_model=None,
                        templates_file=None,
                        sids=None,
                        num=3,
                        days_before=0.01,
                        days_after=10.0,
                        n_time=200,
                        filters=('u','g','r','i','z','y'),
                        use_extinction=True,
                        use_kcorrect=True,
                        k_correct_type=None,
                        k_correct_arg=None,
                        use_log_time=True,
                        overlap=False,
                        ylim=None,
                        fast_peaks=False):
    """
    Plot apparent (not absolute) light curves for a few events from a saved population.

    If fast_peaks=True and the population slice_points contain 'peak_app_mag_ebv_{f}',
    plot per-filter PEAK points at t=0 using those stored values (and add k-correction
    if enabled), avoiding template interpolation for a quick diagnostic view.

    Otherwise, reconstruct apparent light curves via the template library with DM, EBV,
    and optional k-correction.

    Parameters are as before; see earlier docstring. New flag:
    fast_peaks : bool
        If True and peak columns exist, plot only peak magnitudes at t=0 for speed.
    """
    import numpy as _np  # local, shadow-proof

    # coerce pop_file to a plain string if it came in as a 0-d array / scalar box
    if not isinstance(pop_file, (str, os.PathLike)):
        try:
            pop_file = _np.asarray(pop_file).item()
        except Exception:
            raise TypeError(f"pop_file must be a path-like string; got {type(pop_file)}: {pop_file!r}")

    if not os.path.exists(pop_file):
        raise FileNotFoundError(f"pop_file not found: {pop_file}")

    with open(pop_file, 'rb') as f:
        slice_points = pickle.load(f)

    # choose SIDs
    n_events = len(slice_points['file_indx'])
    all_sids = _np.arange(n_events)
    if sids is None:
        sids = all_sids[:min(num, n_events)]
    else:
        sids = _np.array(sids, dtype=int)

    colors = {'u': 'k', 'g': 'b', 'r': 'g', 'i': 'r', 'z': 'magenta', 'y': 'yellow'}
    ax1 = dust_model.ax1

    have_peaks = all(f'peak_app_mag_ebv_{f}' in slice_points for f in filters)

    # -------- FAST PEAKS-ONLY --------
    if fast_peaks and have_peaks:
        for sid in sids:
            dist_mpc  = _np.asarray(slice_points['distance']).ravel()[sid].item()
            ebv_arr   = _np.asarray(slice_points.get('ebv', 0.0))
            ebv       = (ebv_arr if _np.isscalar(ebv_arr) else ebv_arr.ravel()[sid].item()) if not _np.isscalar(ebv_arr) else float(ebv_arr)
            file_indx = int(_np.asarray(slice_points['file_indx']).ravel()[sid].item())

            z = z_at_value(cosmo.comoving_distance, dist_mpc * u.Mpc) if use_kcorrect else None

            if not overlap:
                plt.figure()

            x0 = _np.log10(1e-5) if use_log_time else 0.0
            for f in filters:
                m_peak = _np.asarray(slice_points[f'peak_app_mag_ebv_{f}']).ravel()[sid].item()
                if use_kcorrect:
                    m_peak += apply_kcorrection(z, f, k_correct_type, k_correct_arg)
                plt.scatter([x0], [m_peak], label=f if sid == sids[0] else None,
                            c=colors.get(f, 'gray'), s=40, alpha=0.8)

            title = (f"Population PEAKS (apparent) — SID {sid} | idx={file_indx} | "
                     f"d={dist_mpc:.0f} Mpc | E(B−V)={ebv:.03f}")
            plt.title(title, fontsize=12)
            plt.xlabel("log10(days since peak)" if use_log_time else "Days since peak", fontsize=11)
            plt.ylabel("Apparent mag", fontsize=11)
            plt.gca().invert_yaxis()
            if ylim is not None:
                plt.ylim(ylim)
            if not overlap:
                plt.legend(title="filter", ncol=6, fontsize=9)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

        if overlap:
            plt.legend(title="filter", ncol=6, fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        return

    # -------- FULL LC RECONSTRUCTION --------
    if lc_model is None:
        if templates_file is None:
            raise ValueError("Provide lc_model or templates_file.")
        from local_GRBafterglows_metric import LC
        lc_model = LC(load_from=templates_file)

    # time grid (avoid log of negatives by clipping to ≥1e-5 for plotting)
    t_rel  = _np.linspace(-days_before, days_after, n_time)
    t_plot = _np.log10(_np.maximum(t_rel, 1e-5)) if use_log_time else t_rel

    # Optional: avoid evaluating before template start (prevents 99-mag padding)
    t0 = getattr(lc_model, "t_grid", _np.array([0.0]))[0]
    t_eval = _np.maximum(t_rel, t0)

    for sid in sids:
        file_indx = int(_np.asarray(slice_points['file_indx']).ravel()[sid].item())
        dist_mpc  = float(_np.asarray(slice_points['distance']).ravel()[sid].item())
        ebv_arr   = _np.asarray(slice_points.get('ebv', 0.0))
        ebv       = (ebv_arr if _np.isscalar(ebv_arr) else ebv_arr.ravel()[sid].item()) if not _np.isscalar(ebv_arr) else float(ebv_arr)

        dm = 5 * _np.log10(dist_mpc * 1e6) - 5
        z  = z_at_value(cosmo.comoving_distance, dist_mpc * u.Mpc) if use_kcorrect else None

        if not overlap:
            plt.figure()

        for f in filters:
            m_abs = lc_model.interp(t_eval, f, file_indx)
            # pad out pre-peak times with NaN
            m_abs = _np.where(t_rel < t0, _np.nan, m_abs)
            kcorr = apply_kcorrection(z, f, k_correct_type, k_correct_arg) if use_kcorrect else 0.0
            A_f   = ax1[f] * ebv if use_extinction else 0.0
            m_app = m_abs + kcorr + A_f + dm

            plt.plot(t_plot, m_app, color=colors.get(f, 'gray'),
                     label=f if sid == sids[0] else None, alpha=0.6)

        title = (f"Population LC (apparent): SID {sid} | idx={file_indx} | "
                 f"d={dist_mpc:.0f} Mpc | E(B−V)={ebv:.03f}")
        plt.title(title, fontsize=12)
        plt.xlabel("log10(days since peak)" if use_log_time else "Days since peak", fontsize=11)
        plt.ylabel("Apparent mag", fontsize=11)
        plt.gca().invert_yaxis()
        if ylim is not None:
            plt.ylim(ylim)
        if not overlap:
            plt.legend(title="filter", ncol=6, fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    if overlap:
        plt.legend(title="filter", ncol=6, fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

#8/26

def atomic_save_pickle(obj, path):
    """Atomically save a pickle to avoid partial writes or EOFErrors."""
    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dir_)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)  # atomic on POSIX
        print(f"[INFO] Saved atomically to {path}")
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

