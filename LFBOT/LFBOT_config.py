
import sys
import os

#use this to add text to the filename if you're testing template or population variables 
#other than the variables in build_filenames
testname = None  

#use this to add text to the filename for the observation record dataframe only 
#(does not change pop/template filenames)
#so for anything you're changing that's only about the detection criteria
testname_metric_only = None 

#control whether we generate new files - 
#new files will still generate if these are false and it can't find a file with the correct parameters
#so recommend leaving it on false
generate_new_templates = False 
generate_new_pop = False
make_debug_plots = True #toggle whether or not the pop generation makes plots




'''population variables'''
rate_density = 420e-9

use_kcorrect = True
k_correct_type='blackbody'
k_correct_arg=25000 


# apply None for non-use case 
z_min, z_max = 0.00226, 3 #0.3
d_min, d_max = None, None

gal_lat_cut = None #latitude cut, for Galactic phenomena
use_extinction = True

'''metric variables'''

t_start = 1 #start time in days
t_end = 3652

# Whether to remove Metric_temp_* folders after running
clean_temp = True  

#cadence variables
# cadences = ['four_roll_v4.3.1_10yrs'] #'baseline_v4.3.1_10yrs','noroll_v3.6_10yrs']
cadences = ['four_roll_v5.0.0_10yrs', 'baseline_v5.0.0_10yrs','noroll_v3.6_10yrs','baseline_v3.6_10yrs']

ignore_triples = False #turn this to true to ignore triples

num_lightcurves = 1000
'''utility functions'''


def get_user_config(user):
    '''
    configures paths and databases for Shar or Cristina
    valid options: Cristina, Shar
    
    '''

    if user=="Cristina" or user=="cristina":
        print("[CONFIG] Using Cristina's local MacBook setup")
        sys.path.insert(0, "/Users/andradenebula/Documents/Research/Transient_Metrics/Multi_Transient_Metrics_Hub")
        os.environ["RUBIN_SIM_DATA_DIR"] = "/Users/andradenebula/rubin_sim_data"
        db_dir = "/Users/andradenebula/Documents/Research/Transient_Metrics/Multi_Transient_Metrics_Hub"
        base_dir = '/Users/andradenebula/Documents/Research/Transient_Metrics/Multi_Transient_Metrics_Hub/output'
    elif user=="Shar" or user=="shar":
        print("[CONFIG] Using Shar's Darwin setup")
        sys.path.insert(0, "/lustre/lrspec/metrics")
        sys.path.insert(0, "/home/3155/metrics/Multi_Transient_Metrics_Hub")
        os.environ["RUBIN_SIM_DATA_DIR"] = "/lustre/lrspec/metrics/rubin_sim_data"
        db_dir = "/lustre/lrspec/metrics"
        base_dir = '/lustre/lrspec/metrics/results'
    else:
        print("error: user is not shar or cristina")
    sys.path.append(os.path.abspath(".."))
    return db_dir, base_dir


def get_metric_and_su(metric_filename="LFBOT_metric"):
    '''
    load and reload metric and shared_utils
    '''
    s_u = "shared_utils"
    # --- Reload metric module ---
    if metric_filename in sys.modules:
        del sys.modules[metric_filename]
    metric = __import__(metric_filename)
    
    # --- Reload shared_utils module ---
    if s_u in sys.modules:
        del sys.modules[s_u]
    shared_utils = __import__(s_u)
    
    print(f"[INFO] Loaded metric module: {metric_filename}")
    print(f"[INFO] Loaded shared_utils module")
    return metric, shared_utils