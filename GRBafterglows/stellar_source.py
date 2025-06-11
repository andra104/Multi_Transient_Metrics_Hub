#Crossmatch each candidate’s RA/Dec with Gaia DR3 or DR2, using a small radius (like 1 arcsec).
#If there's a match and the event has stellar morphology (low astrometric excess noise, parallax and proper motion consistent with a star), reject it.

#PROBLEM: 
#If we do this during a full metric execution it could cause a jam with the API unless we cache the result
#could predownload gaia stars? cross match locally? or use a local star catalog or even mock data? 


from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

def is_stellar_source(ra_deg, dec_deg, radius_arcsec=1.0):
    coord = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree)
    Vizier.ROW_LIMIT = 5
    result = Vizier.query_region(coord, radius=radius_arcsec*u.arcsec, catalog="I/355/gaiadr3")

    if len(result) > 0:
        # You could refine using parallax, phot_bp_rp_excess_factor, etc.
        return True
    return False

# We can apply it something like: 
if is_stellar_source(slice_point['ra'] * 180/np.pi, slice_point['dec'] * 180/np.pi):
    return 0.0  # Not a viable transient
