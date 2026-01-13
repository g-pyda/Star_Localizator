import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS

# ---------------- CONFIG ---------------- #
CATALOG_PATH = "./star_data/hygdata_v41.csv"

OUT_DIR = Path("./dataset")
IMAGES_DIR = OUT_DIR / "images"
LABELS_DIR = OUT_DIR / "labels"
SPLITS_DIR = OUT_DIR / "splits"

BAND = "r"

# Patch settings (recommended for CNN)
MAKE_PATCHES = True
PATCH_SIZE = 128  # 128x128
BORDER = PATCH_SIZE // 2

# Control how many stars to process (debugging)
MAX_STARS = 200      # set None for full run
SKIP_SUN = True

# SDSS query controls
MAX_XID = 10
MAX_ARCMIN = 120     # keep smaller than 500 initially
MIN_ARCMIN = 0.2
MAX_ITERS = 60
MAX_ATTEMPTS = 3000
# ---------------------------------------- #


def calculate_dynamic_radius(mag: float) -> float:
    if mag < 1.0:
        return 60.0
    elif mag < 3.0:
        return 30.0
    elif mag < 6.0:
        return 15.0
    else:
        return 5.0


def ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def query_sdss_near(ra_deg: float, dec_deg: float, radius_arcmin: float):
    # Query PhotoObj around coordinates
    query = f"""
    SELECT DISTINCT p.ra, p.dec, p.objID, p.run, p.rerun, p.camcol, p.field, p.u, p.g, p.r, p.i, p.z
    FROM PhotoObj AS p
    JOIN dbo.fGetNearbyObjEq({ra_deg}, {dec_deg}, {radius_arcmin}) AS r ON p.objID = r.objID
    """
    return SDSS.query_sql(query)


def find_reasonable_matches(ra_deg: float, dec_deg: float, mag: float):
    radius = calculate_dynamic_radius(mag)
    xid = None

    # First grow until we have at least 1 match, but not beyond MAX_ARCMIN
    while True:
        xid = query_sdss_near(ra_deg, dec_deg, radius)
        n = 0 if xid is None else len(xid)
        if n >= 1 or radius >= MAX_ARCMIN:
            break
        radius *= 1.2

    if xid is None or len(xid) < 1:
        return None, radius

    # Then shrink if too many results, with safety bounds
    iters = 0
    while len(xid) > MAX_XID and iters < MAX_ITERS and radius > MIN_ARCMIN:
        radius *= 0.9
        xid = query_sdss_near(ra_deg, dec_deg, radius)
        if xid is None:
            xid = []
        iters += 1

    if isinstance(xid, list) and len(xid) == 0:
        return None, radius
    if xid is None or len(xid) < 1 or len(xid) > MAX_XID:
        return None, radius

    return xid, radius


def dedupe_images(images):
    reduced = {}
    for hdul in images:
        hdu0 = hdul[0]
        run = hdu0.header.get("RUN")
        camcol = hdu0.header.get("CAMCOL")
        frame = hdu0.header.get("FRAME")
        key = (run, camcol, frame)
        if key not in reduced:
            reduced[key] = hdul
    return reduced


def world_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float):
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(coord)
    return float(x), float(y)


def crop_patch(data: np.ndarray, x: float, y: float, patch_size: int):
    # data indexed as [y, x]
    h, w = data.shape
    cx, cy = int(round(x)), int(round(y))

    y0 = cy - patch_size // 2
    y1 = y0 + patch_size
    x0 = cx - patch_size // 2
    x1 = x0 + patch_size

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None, None

    patch = data[y0:y1, x0:x1].copy()
    # new coordinates inside patch
    px = x - x0
    py = y - y0
    return patch, (float(px), float(py))


def save_sample(image_id: str, array: np.ndarray, label: dict):
    # Save as .npy for now (simple + lossless); CNN can load npy easily
    img_path = IMAGES_DIR / f"{image_id}.npy"
    np.save(img_path, array)

    lbl_path = LABELS_DIR / f"{image_id}.json"
    with open(lbl_path, "w", encoding="utf-8") as f:
        json.dump(label, f, indent=2)

    return str(img_path), str(lbl_path)


def write_splits(image_ids):
    # deterministic split by star_id hash-like rule (here by trailing digits)
    train_ids = []
    val_ids = []
    for iid in image_ids:
        # image_id begins with star<id>_
        # Extract star_id for split
        try:
            star_id_str = iid.split("_")[0].replace("star", "")
            sid = int(star_id_str)
        except Exception:
            sid = 0
        if sid % 10 == 0:
            val_ids.append(iid)
        else:
            train_ids.append(iid)

    (SPLITS_DIR / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (SPLITS_DIR / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")


def main():
    ensure_dirs()
    star_df = pd.read_csv(CATALOG_PATH)

    image_ids = []
    n_processed = 0
    attempts = 0

    for star in tqdm(star_df.itertuples(), total=len(star_df)):
        attempts += 1
        if MAX_ATTEMPTS is not None and attempts > MAX_ATTEMPTS:
            break

        if SKIP_SUN and star.id == 0:
            continue
        if MAX_STARS is not None and n_processed >= MAX_STARS:
            break

        # HYG RA is in hours; convert to degrees
        ra_deg = float(star.ra) * 15.0
        dec_deg = float(star.dec)
        mag = float(star.mag)

        xid, radius = find_reasonable_matches(ra_deg, dec_deg, mag)
        if xid is None:
            continue

        try:
            images = SDSS.get_images(matches=xid, band=BAND)
        except TimeoutError:
            continue
        except Exception:
            # Network hiccups, server errors, etc.
            continue

        if not images:
            continue

        images_reduced = dedupe_images(images)

        # For v1: save at most one sample per star (first usable image)
        saved_one = False

        for (run, camcol, frame), hdul in images_reduced.items():
            hdu0 = hdul[0]
            data = hdu0.data
            if data is None:
                continue

            # SDSS frames are typically 2D
            if data.ndim != 2:
                continue

            wcs = WCS(hdu0.header)

            try:
                x, y = world_to_pixel(wcs, ra_deg, dec_deg)
            except Exception:
                continue

            h, w = data.shape
            if not (0 <= x < w and 0 <= y < h):
                continue

            if MAKE_PATCHES:
                patch, pxy = crop_patch(data, x, y, PATCH_SIZE)
                if patch is None or pxy is None:
                    continue
                px, py = pxy

                image_id = f"star{star.id}_run{run}_cam{camcol}_frame{frame}_{BAND}_p{PATCH_SIZE}"

                label_path = LABELS_DIR / f"{image_id}.json"
                if label_path.exists():
                    continue

                image_path = IMAGES_DIR / f"{image_id}.npy"
                if image_path.exists():
                    continue

                label = {
                    "image_id": image_id,
                    "source": "SDSS",
                    "band": BAND,
                    "image_shape": [int(PATCH_SIZE), int(PATCH_SIZE)],
                    "objects": [
                        {
                            "star_id": int(star.id),
                            "ra_deg": ra_deg,
                            "dec_deg": dec_deg,
                            "x": px,
                            "y": py,
                            "mag": mag,
                        }
                    ],
                }
                save_sample(image_id, patch.astype(np.float32), label)
            else:
                image_id = f"star{star.id}_run{run}_cam{camcol}_frame{frame}_{BAND}"
                label = {
                    "image_id": image_id,
                    "source": "SDSS",
                    "band": BAND,
                    "image_shape": [int(h), int(w)],
                    "objects": [
                        {
                            "star_id": int(star.id),
                            "ra_deg": ra_deg,
                            "dec_deg": dec_deg,
                            "x": x,
                            "y": y,
                            "mag": mag,
                        }
                    ],
                }
                save_sample(image_id, data.astype(np.float32), label)

            image_ids.append(image_id)
            saved_one = True
            break

        if saved_one:
            n_processed += 1

    write_splits(image_ids)
    print(f"Done. Saved {len(image_ids)} samples.")


if __name__ == "__main__":
    main()
