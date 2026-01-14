import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.utils.data import conf as astropy_conf
astropy_conf.use_download_cache = True



# =======================
# CONFIG
# =======================
CATALOG_PATH = "./star_data/hygdata_v41.csv"

OUT_DIR = Path("./dataset")
IMAGES_DIR = OUT_DIR / "images"
LABELS_DIR = OUT_DIR / "labels"
SPLITS_DIR = OUT_DIR / "splits"

BAND = "r"

# Run controls
SKIP_SUN = True

# Stop conditions:
# - MAX_SAVED_FRAMES limits how many unique frames are stored (good for testing)
# - MAX_ATTEMPTS limits how many stars from the catalog are tried per run
MAX_SAVED_FRAMES = 300     # set None for unlimited
MAX_ATTEMPTS = 3000        # set None for unlimited

# SDSS query controls
MAX_XID = 10
MAX_ARCMIN = 120
MIN_ARCMIN = 0.2
MAX_ITERS = 60

# Network robustness
REMOTE_TIMEOUT_SECONDS = 60  # astropy download timeout; increase if needed

# Labeling rules
# When adding a star to a frame, skip if same star_id already exists in that frame label
SKIP_DUPLICATE_STAR_ID = True


# =======================
# Utility functions
# =======================
def ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_dynamic_radius(mag: float) -> float:
    """Returns a radius in arcminutes based on star brightness."""
    if mag < 1.0:
        return 60.0
    elif mag < 3.0:
        return 30.0
    elif mag < 6.0:
        return 15.0
    else:
        return 5.0


def query_sdss_near(ra_deg: float, dec_deg: float, radius_arcmin: float):
    query = f"""
    SELECT DISTINCT p.ra, p.dec, p.objID, p.run, p.rerun, p.camcol, p.field, p.u, p.g, p.r, p.i, p.z
    FROM PhotoObj AS p
    JOIN dbo.fGetNearbyObjEq({ra_deg}, {dec_deg}, {radius_arcmin}) AS r ON p.objID = r.objID
    """
    return SDSS.query_sql(query)


def find_reasonable_matches(ra_deg: float, dec_deg: float, mag: float):
    """Find a nearby SDSS object match list with size between 1 and MAX_XID if possible."""
    radius = calculate_dynamic_radius(mag)
    xid = None

    # Grow radius until at least 1 match or cap reached
    while True:
        xid = query_sdss_near(ra_deg, dec_deg, radius)
        n = 0 if xid is None else len(xid)
        if n >= 1 or radius >= MAX_ARCMIN:
            break
        radius *= 1.2

    if xid is None or len(xid) < 1:
        return None, radius

    # Shrink radius if too many
    iters = 0
    while len(xid) > MAX_XID and iters < MAX_ITERS and radius > MIN_ARCMIN:
        radius *= 0.9
        xid = query_sdss_near(ra_deg, dec_deg, radius)
        if xid is None:
            xid = []
        iters += 1

    if xid is None or len(xid) < 1 or len(xid) > MAX_XID:
        return None, radius

    return xid, radius


def dedupe_images(images):
    """
    Deduplicate within a single SDSS.get_images() result by (RUN, CAMCOL, FRAME).
    """
    reduced = {}
    for hdul in images:
        hdu0 = hdul[0]
        run = hdu0.header.get("RUN")
        camcol = hdu0.header.get("CAMCOL")
        frame = hdu0.header.get("FRAME")
        if run is None or camcol is None or frame is None:
            continue
        key = (int(run), int(camcol), int(frame))
        if key not in reduced:
            reduced[key] = hdul
    return reduced


def world_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float) -> Tuple[float, float]:
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(coord)
    return float(x), float(y)


def frame_radec_bounds(wcs: WCS, width: int, height: int) -> Tuple[float, float, float, float]:
    """
    Approximate RA/Dec bounds of the frame using the four image corners.
    Returns (ra_min, ra_max, dec_min, dec_max) in degrees.

    Note: RA wrap-around (near 0/360) is not handled in this simple version.
    """
    xs = np.array([0, width - 1, 0, width - 1], dtype=float)
    ys = np.array([0, 0, height - 1, height - 1], dtype=float)

    sky = wcs.pixel_to_world(xs, ys)  # SkyCoord array
    ra = sky.ra.deg
    dec = sky.dec.deg

    return float(np.min(ra)), float(np.max(ra)), float(np.min(dec)), float(np.max(dec))


def frame_id(run: int, camcol: int, frame: int, band: str) -> str:
    return f"run{run}_cam{camcol}_frame{frame}_{band}"


def load_label(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_label(path: Path, label: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(label, f, indent=2)
    tmp.replace(path)


def add_object_to_label(label: Dict[str, Any], obj: Dict[str, Any]) -> bool:
    """
    Append an object to label["objects"] unless already present (by star_id if configured).
    Returns True if added, False if skipped.
    """
    objs = label.setdefault("objects", [])

    if SKIP_DUPLICATE_STAR_ID and "star_id" in obj:
        sid = obj["star_id"]
        for existing in objs:
            if existing.get("star_id") == sid:
                return False

    objs.append(obj)
    return True


def write_splits_from_labels():
    """
    Deterministic train/val split based on frame_id hash.
    Uses all labels on disk, so it works even after crashes.
    """
    label_files = sorted(LABELS_DIR.glob("*.json"))
    train_ids = []
    val_ids = []

    for p in label_files:
        fid = p.stem
        # Deterministic split: simple hash modulo
        h = int(hashlib.md5(fid.encode("utf-8")).hexdigest(), 16) % 10
        if h == 0:
            val_ids.append(fid)
        else:
            train_ids.append(fid)

    (SPLITS_DIR / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (SPLITS_DIR / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    print(f"Splits written from disk: train={len(train_ids)} val={len(val_ids)} total={len(label_files)}")


# =======================
# Main generator
# =======================
def main():
    ensure_dirs()
    astropy_conf.remote_timeout = REMOTE_TIMEOUT_SECONDS

    # Load HYG catalog once
    star_df = pd.read_csv(CATALOG_PATH)

    # Precompute degrees for fast filtering (frame-centric labeling)
    star_df["ra_deg"] = star_df["ra"].astype(float) * 15.0
    star_df["dec_deg"] = star_df["dec"].astype(float)

    # Track existing frames (labels already on disk)
    existing_labels = {p.stem for p in LABELS_DIR.glob("*.json")}
    saved_frames = set(existing_labels)

    attempts = 0

    for star in tqdm(star_df.itertuples(index=False), total=len(star_df)):
        attempts += 1
        if MAX_ATTEMPTS is not None and attempts > MAX_ATTEMPTS:
            break

        if SKIP_SUN and int(star.id) == 0:
            continue

        # Stop when enough unique frames have labels
        if MAX_SAVED_FRAMES is not None and len(saved_frames) >= MAX_SAVED_FRAMES:
            break

        # HYG RA is in hours -> degrees
        ra_deg = float(star.ra_deg)
        dec_deg = float(star.dec_deg)
        mag = float(star.mag)

        xid, _radius = find_reasonable_matches(ra_deg, dec_deg, mag)
        if xid is None:
            continue

        # Download candidate frames (can timeout)
        try:
            images = SDSS.get_images(matches=xid, band=BAND)
        except TimeoutError:
            continue
        except Exception:
            continue

        if not images:
            continue

        images_reduced = dedupe_images(images)

        # For each SDSS frame, attach ALL HYG stars inside that frame
        for (run, camcol, frame), hdul in images_reduced.items():
            fid = frame_id(run, camcol, frame, BAND)

            # Respect max unique frames
            if MAX_SAVED_FRAMES is not None and len(saved_frames) >= MAX_SAVED_FRAMES:
                break

            img_path = IMAGES_DIR / f"{fid}.npz"
            lbl_path = LABELS_DIR / f"{fid}.json"

            hdu0 = hdul[0]
            data = hdu0.data
            if data is None or not hasattr(data, "ndim") or data.ndim != 2:
                continue

            h, w = data.shape

            # Build WCS
            try:
                wcs = WCS(hdu0.header)
            except Exception:
                continue

            # Save image once per frame
            if not img_path.exists():
                try:
                    np.savez_compressed(img_path, data=data.astype(np.float32))
                except Exception:
                    continue

            # Load or create label
            label = load_label(lbl_path)
            if label is None:
                label = {
                    "image_id": fid,
                    "source": "SDSS",
                    "band": BAND,
                    "image_shape": [int(h), int(w)],
                    "objects": [],
                }

            # Compute approximate RA/Dec bounds of this frame (corner-based)
            try:
                ra_min, ra_max, dec_min, dec_max = frame_radec_bounds(wcs, w, h)
            except Exception:
                continue

            # Small margin in degrees to avoid edge numerical issues
            margin = 0.05
            ra_min -= margin
            ra_max += margin
            dec_min -= margin
            dec_max += margin

            # Candidate HYG stars potentially inside this frame
            # Note: RA wrap-around near 0/360 not handled in this simple filter.
            cands = star_df[
                (star_df["dec_deg"] >= dec_min) & (star_df["dec_deg"] <= dec_max) &
                (star_df["ra_deg"] >= ra_min) & (star_df["ra_deg"] <= ra_max)
            ]

            # Add all candidate stars that actually map inside pixel bounds
            changed = False
            for row in cands.itertuples(index=False):
                sid = int(row.id)
                if SKIP_SUN and sid == 0:
                    continue

                try:
                    sx, sy = world_to_pixel(wcs, float(row.ra_deg), float(row.dec_deg))
                except Exception:
                    continue

                if not (0 <= sx < w and 0 <= sy < h):
                    continue

                obj = {
                    "star_id": sid,
                    "ra_deg": float(row.ra_deg),
                    "dec_deg": float(row.dec_deg),
                    "x": float(sx),
                    "y": float(sy),
                    "mag": float(row.mag),
                }

                # add_object_to_label should skip duplicates by star_id
                added = add_object_to_label(label, obj)
                if added:
                    changed = True

            # Save label if anything was added/updated
            if changed or (fid not in saved_frames):
                save_label(lbl_path, label)

            saved_frames.add(fid)

    # Always write splits from what exists on disk
    write_splits_from_labels()
    print(f"Done. Unique frames with labels: {len({p.stem for p in LABELS_DIR.glob('*.json')})}")

if __name__ == "__main__":
    main()
