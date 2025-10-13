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
z_min, z_max = None, None 
d_min, d_max = 0.0000013, .03
# #proxima cen is an m-dwarf apparently at 1.3 pc
#the milky way is 30 kpc across and the sun is 8kpc from the center
#let's just use 30 kpc because we don't have a way to weight it right now

gal_lat_cut = 30 #latitude cut, for Galactic phenomena
use_extinction = True

rate_density = 3880912272582138 #should be recalculated

use_kcorrect=False
k_correct_type = None
k_correct_arg = None



'''metric variables'''
t_start = 1 #start time in days
t_end = 3652

# Whether to remove Metric_temp_* folders after running
clean_temp = True  # <- NEW toggle

cadences = ['four_roll_v5.0.0_10yrs', 'baseline_v5.0.0_10yrs','noroll_v3.6_10yrs','baseline_v3.6_10yrs']

ignore_triples = False #turn this to true to ignore triples





num_lightcurves = 1000

# #this is how i got the rate density, should be checked - reversing from numbers
# rate_per_year = 10**11
# d_min, d_max = d_min, d_max = 0.0000013, .03
# V = ((4/3)*np.pi*(d_max**3 - d_min**3))
# #except only within 30 degrees
# #how much of a sphere is plus or minus 30 degrees i wonder
# #i think it's actually just a third
# new_rate_density = 3*(rate_per_year / V)*(65130/44516)

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


def get_metric_and_su(metric_filename="MDwarfFlares_metric"):
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