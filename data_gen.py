from astroquery.sdss import SDSS
from astroquery.simbad import Simbad
from astropy import coordinates as coords
from astropy.wcs import WCS
import astropy.units as u

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# -------------- ENV ---------------- #
MAX_XID = 10
MAX_ARCMIN = 500
TESTING = True
TESTING_STAR_THR = 2
# ----------------------------------- #

# going through every star in the star dataset and saving their data
star_df = pd.read_csv('./star_data/hygdata_v41.csv')
#print(star_df['proper'].unique())

# looking for the pictures containing the stars for each star

def calculate_dynamic_radius(mag):
    """
    Returns a radius in arcminutes based on star brightness.
    Brighter stars (lower magnitude) get a wider field of view.
    """
    if mag < 1.0:       # Super bright (e.g., Betelgeuse, Rigel)
        return 60.0     # 1 Degree (Wide view)
    elif mag < 3.0:     # Very bright (e.g., Meissa)
        return 30.0     # 30 arcmin
    elif mag < 6.0:     # Naked eye visible
        return 15.0
    else:               # Telescope objects
        return 5.0      # Zoom in tight

# loop over each star
for star in star_df.itertuples():
    if star.id == 0: continue  # ommiting the Sun
    if TESTING and star.id == TESTING_STAR_THR: break  # temporary early loop end

    # defining the search position
    ra_deg = star.ra * 15
    dec_deg = star.dec
    radius_arcmin = calculate_dynamic_radius(star.mag)

    xid = []

    # looking for the specific star amount from queried database
    while len(xid) > MAX_XID or len(xid) < 1:
        while len(xid) < 1:
            query = f"""
            SELECT DISTINCT p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.objID,
                   p.run, p.rerun, p.camcol, p.field
            FROM PhotoObj AS p
            JOIN dbo.fGetNearbyObjEq({ra_deg}, {dec_deg}, {radius_arcmin}) AS r ON p.objID = r.objID
            """

            xid = SDSS.query_sql(query)
            radius_arcmin *= 1.1
            if xid is None:
                xid = []
            #print(star.id, radius_arcmin, len(xid))
            if radius_arcmin > MAX_ARCMIN:
                break
        if radius_arcmin > MAX_ARCMIN:
            break

        radius_arcmin /= 1.05

        while len(xid) > MAX_XID:
            # querying the SDSS picture database
            query = f"""
            SELECT DISTINCT p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.objID,
                   p.run, p.rerun, p.camcol, p.field
            FROM PhotoObj AS p
            JOIN dbo.fGetNearbyObjEq({ra_deg}, {dec_deg}, {radius_arcmin}) AS r ON p.objID = r.objID
            """

            xid = SDSS.query_sql(query)

            radius_arcmin *= 0.95

    print(star.id, len(xid), radius_arcmin)

    images = SDSS.get_images(matches=xid, band='r')
    if images:
         print(f"Downloaded {len(images)} image(s).")
    else:
        print("No images found.")
        continue

    # CLEANUP - reducing the picture duplicates
    images_reduced = {}
    for image in images:
        run = image[0].header["RUN"]
        camcol = image[0].header["CAMCOL"]
        frame = image[0].header["FRAME"]

        if (run, camcol, frame) not in images_reduced.keys():
            images_reduced[(run, camcol, frame)] = image
            print("img added")
        else:
            print("img already added")

    # searching for the brightest star in the found ones
    # looking for the catalog name
    # hyg first, Hipparcos Catalog then
    # adding it to the specified picture

    # test - displaying the pictures
    if TESTING:
        for image in images_reduced.values():
            hdu = image[0]
            wcs = WCS(hdu.header)
            data = hdu.data

            plt.imshow(data, cmap='gray', origin='lower', norm=LogNorm())
            plt.show()
