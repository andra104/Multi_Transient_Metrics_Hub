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

def plot_some_lcs_from_pkl(templates_file, num=3):
    '''
    templates_file is the path to the pkl with the lc templates
    num is how many we plot, starting at the beginning of the file
    '''
    lcdict = pickle.load(open(templates_file, "rb"))
    for i in range(num):
        plt.plot(np.log10(lcdict["lightcurves"][i]['u']['ph']+1e-5),
                         lcdict["lightcurves"][i]['u']['mag'], color='k')
        plt.plot(np.log10(lcdict["lightcurves"][i]['u']['ph']+1e-5),
                          lcdict["lightcurves"][i]['g']['mag'], color='b')
        plt.plot(np.log10(lcdict["lightcurves"][i]['u']['ph']+1e-5),
                          lcdict["lightcurves"][i]['r']['mag'], color='g')
        plt.plot(np.log10(lcdict["lightcurves"][i]['u']['ph']+1e-5),
                          lcdict["lightcurves"][i]['i']['mag'], color='r')
        plt.plot(np.log10(lcdict["lightcurves"][i]['u']['ph']+1e-5),
                        lcdict["lightcurves"][i]['z']['mag'], color='brown')
        ylims = plt.ylim()
        plt.title("Light curves from pkl file: #"+str(i))
        plt.xlabel("log time (days)")
        plt.ylabel("abs mag")
        plt.ylim(*ylims[::-1])
        plt.show()


# --------------------------------------------
# Run detect metric
# --------------------------------------------

def run_detect(metric, slicer, cadences, shared_lc_model, db_dir, storage_dir, ignore_triples=False, debug=True, plot=True):
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
    n_events = len(slicer.slice_points['distance'])
    note = "scheduler_note not like 'long%'" #if we want to avoid triples
    
    for cadence in cadences:
        runName = cadence
        opsdb = os.path.join(db_dir, f"{cadence}.db")
        outDir = os.path.join(storage_dir, f"Metric_temp_{cadence}")
        os.makedirs(outDir, exist_ok=True)
        resultsDb = db.ResultsDb(out_dir=outDir)
        
    
        print(f"\n--- Running {cadence} ---")

        #### per filter metric here
        per_filter_metrics = OrderedDict()
        filters = ['all']
        for filt in filters:
            detect = metric.Detect_Metric(metricName=f"Detect_{filt}", #filter_include=[filt], 
                                             lc_model=shared_lc_model)
                        #GRBAfterglowSpecTriggerableMetric(metricName=f"GRB_Detect_{filt}", filter_include=[filt], 
                        #                      lc_model=shared_lc_model)
            if ignore_triples == True:
                per_filter_metrics[f"Detect_{filt}"] = metric_bundles.MetricBundle(detect, slicer, '' + note)
            else:
                per_filter_metrics[f"Detect_{filt}"] = metric_bundles.MetricBundle(detect, slicer, '')

        pf_group = metric_bundles.MetricBundleGroup(per_filter_metrics, opsdb, out_dir=outDir, results_db=resultsDb)
        pf_group.run_all()

                # save obs_data
        bundle = per_filter_metrics["Detect_all"]
        
        # Pull the actual metric instance used inside the bundle
        detect_metric = bundle.metric

        #get results
        obs_records = list(detect_metric.obs_records.values())
    
        #Error checking 1
        if debug==True:
            print("\nInspecting one obs_record before saving to CSV:")
            sample_record = obs_records[0]
            for key, val in sample_record.items():
                print(f"{key:15} | type: {type(val)}", end='')
                try:
                    print(f" | length: {len(val)}")
                except TypeError:
                    print(f" | value: {val}")
        # Now get the results
    
        df_obs = pd.DataFrame.from_dict(detect_metric.obs_records).T.reset_index().rename(columns={"index": "sid"})

        #if you want to keep the local version without turning arrays to lists
        df_obs_arr = pd.DataFrame.from_dict(detect_metric.obs_records).T.reset_index().rename(columns={"index": "sid"})
        
        # =======================================================================
        # Add calendar year assuming MJD0 = 59853.5 (LSST start)
        # Convert peak MJD to years since LSST start (365.25 days/year)
        df_obs["year"] = (df_obs["peak_time"] / 365.25).astype(int) + 1
        df_detected_per_year = df_obs[df_obs['detected'] == True].groupby("year").size().reset_index(name="n_detected")
        # =======================================================================
    
    
        # Convert problematic ndarray columns to lists before saving
        #shar note - just use df_obs_arr if you want arrays
        for col in ['filter', 'mjd_obs', 'mag_obs', 'snr_obs']:
            df_obs[col] = df_obs[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        # Now save
        df_obs.to_csv(f"{outDir}/ObsRecords_{cadence}.csv", index=False)
    
    
        n_filters_detected_per_event = np.array([
            sum(per_filter_metrics[f"Detect_{f}"].metric_values[i] == 1 
                and not per_filter_metrics[f"Detect_{f}"].metric_values.mask[i]
                for f in filters)
            for i in range(n_events)
        ])
        
        detected_mask = n_filters_detected_per_event >= 1
        n_detected = np.sum(detected_mask)
        mean_filters = np.mean(n_filters_detected_per_event[detected_mask])
        std_filters = np.std(n_filters_detected_per_event[detected_mask])
    
        print(f"Out of {n_events} simulated events, Rubin detected {n_detected} under the {cadence} cadence.")
        print(f"Of those, each event was observed in an average of {mean_filters:.1f} ± {std_filters:.1f} filters.")
        #shar i want this in a file but we should change it later
        with open("output.txt", "a") as f:
            print(f"Out of {n_events} simulated events, Rubin detected {n_detected} under the {cadence} cadence.", file=f)
            print(f"Of those, each event was observed in an average of {mean_filters:.1f} ± {std_filters:.1f} filters.", file=f)
        
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
                file_indx = slicer.slice_points['file_indx'][i]
                
                m_peak = np.min(shared_lc_model.data[file_indx][filtername]['mag'])
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
        
            # Plot: Apparent magnitude vs RA
        
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
            
            # Plot: Apparent magnitude vs Dec
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
        
    
        # save summaries
        outfile = os.path.join(storage_dir, f"local_efficiency_{cadence}.csv")
        with open(outfile, "w") as out:
            out.write("sid,n_filters_detected\n")
            for i in range(n_events):
                out.write(f"{i},{n_filters_detected_per_event[i]}\n")
        
        if plot == True:
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
            plt.hist(np.degrees(slicer.slice_points['dec']), bins=50, alpha=0.5, label='Injected')
            plt.hist(np.degrees(np.array(decs)[detected_flags]), bins=50, alpha=0.8, label='Detected', color='red')
            plt.xlabel("Declination [deg]")
            plt.ylabel("Number of Events")
            plt.title(f"{cadence} – Declination Distribution")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    return df_obs_arr #for the last cadence that ran

# --------------------------------------------
# Run several metrics
# --------------------------------------------



def run_multi_metrics(multi_metrics, slicer, cadences, shared_lc_model, db_dir, storage_dir, ignore_triples=False, plot=True):
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

    saves: nothing (maybe it should though idk)
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




        for one_metric in multi_metrics:
            mb_key = f"{runName}_{one_metric.__class__.__name__}"
            if ignore_triples == True:
                bundle = metric_bundles.MetricBundle(one_metric, slicer, '' + note, file_root=mb_key, plot_funcs=[], summary_metrics=[metrics.SumMetric()])
            else:
                bundle = metric_bundles.MetricBundle(one_metric, slicer, '', file_root=mb_key, plot_funcs=[], summary_metrics=[metrics.SumMetric()])
            
            bd = maf.metricBundles.make_bundles_dict_from_list([bundle])
            bgroup = metric_bundles.MetricBundleGroup({mb_key: bundle}, opsdb, out_dir=outDir, results_db=resultsDb)
            bgroup.run_all()
            if first:
                df = pd.DataFrame([bd[k].summary_values for k in bd], index=list(bd.keys()))
                df["run"] = runName
                first = 0
            else:
                _ = pd.DataFrame([bd[k].summary_values for k in bd], index=list(bd.keys()))
                _["run"] = runName            
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
                    file_indx = slicer.slice_points['file_indx'][i]
                    
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
        

    
    return df