# # from astroquery.sdss import SDSS
# # from astropy import coordinates as coords
# # import astropy.units as u
# # from astropy.wcs import WCS
# #
# # # 1. Define Position (Meissa / Lambda Orionis)
# # ra_deg = 83.7845
# # dec_deg = 9.9342
# # radius_arcmin = 22
# #
# # # 2. SQL Query: We MUST include run, rerun, camcol, and field
# # # distinct=True is good practice to avoid duplicate star entries
# # query = f"""
# # SELECT DISTINCT p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.objID,
# #        p.run, p.rerun, p.camcol, p.field
# # FROM PhotoObj AS p
# # JOIN dbo.fGetNearbyObjEq({ra_deg}, {dec_deg}, {radius_arcmin}) AS r ON p.objID = r.objID
# # """
# #
# # print("Querying SDSS database...")
# # xid = SDSS.query_sql(query)
# #
# # if xid is None:
# #     print("No objects found.")
# # else:
# #     print(f"Found {len(xid)} stars/objects.")
# #
# #     # 3. Download Images using 'matches'
# #     # Because xid now has 'run', 'camcol', etc., this will work.
# #     print("Downloading images...")
# #     images = SDSS.get_images(matches=xid, band='r')
# #
# #     if images:
# #         print(f"Downloaded {len(images)} image(s).")
# #
# #         # 4. Map the first star to pixels on the first image
# #         # Note: SDSS might return multiple overlapping images for a large region.
# #         # We check the first one.
# #         wcs = WCS(images[0][0].header)
# #
# #         # Let's map the first star in our list
# #         star_ra = xid[0]['ra']
# #         star_dec = xid[0]['dec']
# #
# #         star_coords = coords.SkyCoord(ra=star_ra * u.degree, dec=star_dec * u.degree)
# #         pixel_x, pixel_y = wcs.world_to_pixel(star_coords)
# #
# #         print(f"Star RA/Dec: {star_ra}, {star_dec}")
# #         print(f"Pixel Coordinates: X={pixel_x:.2f}, Y={pixel_y:.2f}")
# #     else:
# #         print("No images returned.")
# #
# # import matplotlib.pyplot as plt
# # from matplotlib.colors import LogNorm
# #
# # # Ensure we actually downloaded images in the previous step
# # if images:
# #     # 1. Extract data from the first downloaded FITS file
# #     hdu = images[0][0]  # Access the primary Header Data Unit
# #     wcs = WCS(hdu.header)  # Get the localization matrix
# #     data = hdu.data  # The raw pixel array
# #
# #     # 2. Setup the Plot
# #     plt.figure(figsize=(10, 10))
# #
# #     # We use 'LogNorm' because stars are very bright compared to the background.
# #     # 'origin=lower' is standard for FITS files (0,0 is bottom-left).
# #     plt.imshow(data, cmap='gray', origin='lower', norm=LogNorm())
# #
# #     # 3. Overlay the stars (Localization Check)
# #     # Convert ALL stars in your catalog to pixel coordinates for this specific image
# #     star_coords = coords.SkyCoord(ra=xid['ra'] * u.degree, dec=xid['dec'] * u.degree)
# #
# #     # Convert World (RA/Dec) -> Pixel (X/Y)
# #     pixel_coords = wcs.world_to_pixel(star_coords)
# #     pixel_x, pixel_y = pixel_coords
# #
# #     # Plot red circles around the stars
# #     # alpha=0.6 makes them slightly transparent
# #     plt.scatter(pixel_x, pixel_y, s=50, edgecolor='red', facecolor='none', alpha=0.6, label='Catalog Stars')
# #     plt.scatter(pixel_x[0], pixel_y[0], s=50, edgecolor='green', facecolor='none', alpha=0.6, label='Catalog Stars')
# #
# #     # 4. Final Polish
# #     plt.title(f"SDSS Image (Band: r) with {len(xid)} Catalog Stars")
# #     plt.xlabel("Pixels X")
# #     plt.ylabel("Pixels Y")
# #     plt.legend(loc='upper right')
# #     plt.grid(color='white', linestyle='--', alpha=0.3)
# #
# #     # Adjust limits to match image size (removes stars that fall outside this specific image patch)
# #     plt.xlim(0, data.shape[1])
# #     plt.ylim(0, data.shape[0])
# #
# #     plt.show()
# # else:
# #     print("No images to show.")
#
#
# # from astroquery.skyview import SkyView
# # from astropy.wcs import WCS
# # import astropy.units as u
# # import matplotlib.pyplot as plt
# # from matplotlib.colors import Normalize
# #
# # # 1. Define Position & Scale
# # target = 'Betelgeuse'
# # survey = 'DSS2 Red'
# #
# # # 2. KEY CHANGE: Use Degrees instead of Arcminutes
# # # Radius = 5 degrees means a total width of 10 degrees (huge!)
# # # pixels=1000 forces the server to return a 1000x1000 image,
# # # preventing the download of a massive 5GB file.
# # print(f"Querying wide-field view around {target}...")
# # images = SkyView.get_images(position=target,
# #                             survey=[survey],
# #                             radius=7.5 * u.degree,
# #                             pixels=1000)
# #
# # if images:
# #     # 3. Extract Data
# #     fits_file = images[0][0]
# #     wcs = WCS(fits_file.header)
# #     data = fits_file.data
# #
# #     # 4. Plotting
# #     plt.figure(figsize=(12, 12))
# #
# #     # Use Normalize. DSS values vary, usually 2000-15000 is a good range for visibility.
# #     # If it looks black, lower vmin/vmax. If white, raise them.
# #     plt.imshow(data, origin='lower', cmap='inferno', norm=Normalize(vmin=2000, vmax=12000))
# #
# #     plt.title(f"Wide Field View: {target} & Surroundings (10 Degrees Width)")
# #     plt.xlabel("Pixels")
# #     plt.ylabel("Pixels")
# #
# #     # Remove grid for a cleaner "photo" look
# #     plt.axis('off')
# #
# #     plt.show()
# #
# #     # 5. Verify it's not just one star
# #     # Let's check the pixel scale (degrees per pixel)
# #     scale = wcs.proj_plane_pixel_scales()[0].to(u.degree)
# #     print(f"Pixel Scale: Each pixel represents {scale:.4f} degrees.")
# #     print(f"Total Image Width: {scale * data.shape[1]:.2f} degrees.")
# #
# # else:
# #     print("No images found.")
#
#
# from astroquery.sdss import SDSS
# from astroquery.simbad import Simbad
# from astropy import coordinates as coords
# from astropy.wcs import WCS
# import astropy.units as u
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
#
# # ==========================================
# # 1. SETUP & SEARCH
# # ==========================================
#
# # Position: Meissa (Lambda Orionis)
# ra_deg = 83.7845
# dec_deg = 9.9342
# radius_arcmin = 26
#
# print("1. Querying SDSS Database...")
#
# # SQL Query to find objects and their image addresses
# query = f"""
# SELECT DISTINCT p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.objID,
#        p.run, p.rerun, p.camcol, p.field
# FROM PhotoObj AS p
# JOIN dbo.fGetNearbyObjEq({ra_deg}, {dec_deg}, {radius_arcmin}) AS r ON p.objID = r.objID
# """
# xid = SDSS.query_sql(query)
#
# if xid is None:
#     print("No objects found.")
#     exit()
#
# print(f"   Found {len(xid)} objects.")
#
# # ==========================================
# # 2. DOWNLOAD IMAGE
# # ==========================================
# print("2. Downloading Image...")
# images = SDSS.get_images(matches=xid, band='r')
#
# if not images:
#     print("No images returned.")
#     exit()
#
# # Use the first image found
# hdu = images[0][0]
# wcs = WCS(hdu.header)
# data = hdu.data
#
# # ==========================================
# # 3. NAME LOOKUP (The New Part)
# # ==========================================
# print("3. Looking up names for bright stars (Magnitude < 12)...")
#
# # --- FIX: Reset Simbad to defaults to ensure 'MAIN_ID' exists ---
# Simbad.reset_votable_fields()
#
# # Sort data so brightest stars (lowest 'r') are first
# xid.sort('r')
#
# # We will create a list to store names so we can plot them later
# named_stars_labels = []
#
# # SIMBAD Setup: We want the main ID
# Simbad.add_votable_fields('main_id')
#
# # Loop through the stars found by SDSS
# for row in xid:
#     # PERFORMANCE HACK: Only check stars brighter than magnitude 12
#     # Checking all 1000+ stars would take 20 minutes.
#     if row['r'] < 12:
#         try:
#             # Create a coordinate object for this specific star
#             star_pos = coords.SkyCoord(ra=row['ra'] * u.degree, dec=row['dec'] * u.degree)
#
#             # Ask SIMBAD: "Who is here?" (radius=5 arcsec tolerance)
#             result = Simbad.query_region(star_pos, radius=5 * u.arcsec)
#
#             if result:
#                 # --- THE FIX ---
#                 # Instead of asking for 'MAIN_ID', we find out what the first column is called
#                 # and use that. This works regardless of version/formatting.
#                 id_column_name = result.colnames[0]
#                 name = str(result[0][id_column_name])
#                 # ---------------
#
#                 # Convert this star's position to pixels for plotting
#                 px, py = wcs.world_to_pixel(star_pos)
#
#                 # Save the pixel location and name for the plot
#                 named_stars_labels.append((px, py, name))
#                 print(f"   -> Match: SDSS Obj {row['objID']} = {name}")
#
#         except Exception as e:
#             print(f"   -> Error querying star: {e}")
#
# # ==========================================
# # 4. VISUALIZATION
# # ==========================================
# print("4. Plotting...")
# plt.figure(figsize=(12, 12))
#
# # Plot Image (Black & White)
# plt.imshow(data, cmap='gray', origin='lower', norm=LogNorm())
#
# # Overlay ALL stars from SDSS (Red Circles)
# all_star_coords = coords.SkyCoord(ra=xid['ra'] * u.degree, dec=xid['dec'] * u.degree)
# pixel_x, pixel_y = wcs.world_to_pixel(all_star_coords)
# plt.scatter(pixel_x, pixel_y, s=30, edgecolor='red', facecolor='none', alpha=0.5, label='SDSS Catalog Objects')
# plt.scatter(pixel_x[0], pixel_y[0], s=30, edgecolor='green', facecolor='none', alpha=0.5, label='SDSS Catalog Objects')
#
# # Overlay NAMES (Yellow Text)
# for px, py, name in named_stars_labels:
#     # Verify the text is inside the image bounds before drawing
#     if 0 <= px < data.shape[1] and 0 <= py < data.shape[0]:
#         plt.text(px + 10, py + 10, name, color='yellow', fontsize=12, weight='bold')
#         plt.scatter(px, py, s=100, edgecolor='yellow', facecolor='none')  # Highlight named stars
#
# # Final formatting
# plt.title(f"SDSS Image of Meissa Region\nRed = All Objects | Yellow = Named Bright Stars")
# plt.xlabel("Pixels X")
# plt.ylabel("Pixels Y")
# plt.xlim(0, data.shape[1])
# plt.ylim(0, data.shape[0])
# plt.legend(loc='upper right')
# plt.grid(color='white', linestyle='--', alpha=0.2)
#
# plt.show()

from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

# 1. Setup Vizier
# We increase the row limit because the default is only 50
v = Vizier(columns=['HIP', 'RAhms', 'DEdms', 'Vmag', 'Plx', 'pmRA', 'pmDE'],
           row_limit=5000)

# 2. Define a region (e.g., around Betelgeuse)
coord = SkyCoord(ra=88.7929, dec=7.4070, unit=(u.deg, u.deg), frame='icrs')

# 3. Query the Hipparcos Catalog (I/239)
result = v.query_region(coord, radius=2*u.deg, catalog="I/239/hip_main")

# 4. Access the data
if result:
    df = result[0].to_pandas()
    print(df.head())
else:
    print("No stars found.")