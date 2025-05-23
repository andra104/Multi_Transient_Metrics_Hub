from astropy.coordinates import Galactic, ICRS as ICRSFrame
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

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
    plt.title("GRB Sky UniformSphere Distribution")
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