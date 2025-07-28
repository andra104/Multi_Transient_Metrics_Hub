def run_detect(metric, slicer, cadences, shared_lc_model, db_dir, storage_dir, df_file, ignore_triples=False, debug=True, plot=True, clean_temp=False, use_extinction=True):
    '''
    Runs the detect metric on given cadences and light curves
    '''
    n_events = len(slicer.slice_points['distance'])
    note = "scheduler_note not like 'long%'"  # if we want to avoid triples

    for cadence in cadences:
        runName = cadence
        opsdb = os.path.join(db_dir, f"{cadence}.db")
        outDir = os.path.join(storage_dir, f"Metric_temp_{cadence}")
        os.makedirs(outDir, exist_ok=True)
        resultsDb = db.ResultsDb(out_dir=outDir)

        print(f"\n--- Running {cadence} ---")

        # Initialize per-filter metrics
        per_filter_metrics = OrderedDict()
        filters = ['all']
        for filt in filters:
            detect = metric.Detect_Metric(metricName=f"Detect_{filt}", lc_model=shared_lc_model, use_extinction=use_extinction)
            if ignore_triples:
                per_filter_metrics[f"Detect_{filt}"] = metric_bundles.MetricBundle(detect, slicer, '' + note)
            else:
                per_filter_metrics[f"Detect_{filt}"] = metric_bundles.MetricBundle(detect, slicer, '')

        pf_group = metric_bundles.MetricBundleGroup(per_filter_metrics, opsdb, out_dir=outDir, results_db=resultsDb)
        pf_group.run_all()

        # Pull results
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

        df_obs["year"] = (df_obs["peak_time"] / 365.25).astype(int) + 1
        df_detected_per_year = df_obs[df_obs['detected'] == True].groupby("year").size().reset_index(name="n_detected")

        for col in ['filter', 'mjd_obs', 'mag_obs', 'snr_obs']:
            df_obs[col] = df_obs[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        # GRB-specific flag
        is_grb = hasattr(metric, '__name__') and 'GRBafterglow' in metric.__name__

        n_observations_detected = []
        n_filters_detected_per_event = []
        n_filters_detected_per_detected_event = []
        n_detected = 0
        peak_abs_mag_g, alpha_fade_g, t_jetbreak_g = [], [], []

        for i, row in df_obs.iterrows():
            file_indx = row['file_indx']
            filt_arr = np.array(row["filter"])
            snr_arr = np.array(row["snr_obs"])
            good = snr_arr >= 5

            n_filters_detected_per_event.append(len(np.unique(filt_arr[good])))
            n_observations_detected.append(np.sum(good))

            if is_grb:
                try:
                    peak_abs_mag_g.append(shared_lc_model.data[file_indx]['g']['mag'][0])
                    alpha_fade_g.append(shared_lc_model.data[file_indx]['g']['mag'][1])
                    t_jetbreak_g.append(shared_lc_model.data[file_indx]['g']['mag'][2])
                except IndexError:
                    peak_abs_mag_g.append(np.nan)
                    alpha_fade_g.append(np.nan)
                    t_jetbreak_g.append(np.nan)

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

        df_obs.to_csv(df_file + f"ObsRecords_{cadence}.csv", index=False)
        print("Obs_Record dataframe saved to ", df_file + f"ObsRecords_{cadence}.csv")

        outfile = os.path.join(storage_dir, f"local_efficiency_{cadence}.csv")
        with open(outfile, "w") as out:
            out.write("sid,n_filters_detected\n")
            for i in range(len(df_obs)):
                out.write(f"{i},{n_filters_detected_per_event[i]}\n")

        if plot:
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
            plt.bar(df_detected_per_year["year"], df_detected_per_year["n_detected"], width=0.7, align='center', edgecolor='black')
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
    
            pass

        if clean_temp:
            print(f"[CLEANUP] Removing temp directory: {outDir}")
            shutil.rmtree(outDir, ignore_errors=True)

    return df_obs
