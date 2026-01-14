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
                existing.update(obj)
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

    star_df = pd.read_csv(CATALOG_PATH)

    # Track existing frames so we can stop at MAX_SAVED_FRAMES reliably
    existing_labels = {p.stem for p in LABELS_DIR.glob("*.json")}
    saved_frames = set(existing_labels)

    attempts = 0

    for star in tqdm(star_df.itertuples(), total=len(star_df)):
        attempts += 1
        if MAX_ATTEMPTS is not None and attempts > MAX_ATTEMPTS:
            break

        if SKIP_SUN and int(star.id) == 0:
            continue

        # Stop when we have enough unique frames saved
        if MAX_SAVED_FRAMES is not None and len(saved_frames) >= MAX_SAVED_FRAMES:
            break

        # HYG RA is in hours -> degrees
        ra_deg = float(star.ra) * 15.0
        dec_deg = float(star.dec)
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

        # For each returned SDSS frame, compute star pixel and attach to that frame label
        for (run, camcol, frame), hdul in images_reduced.items():
            fid = frame_id(run, camcol, frame, BAND)
            img_path = IMAGES_DIR / f"{fid}.npz"
            lbl_path = LABELS_DIR / f"{fid}.json"

            hdu0 = hdul[0]
            data = hdu0.data
            if data is None or not hasattr(data, "ndim") or data.ndim != 2:
                continue

            # WCS -> pixel coords for this star in this frame
            try:
                wcs = WCS(hdu0.header)
                x, y = world_to_pixel(wcs, ra_deg, dec_deg)
            except Exception:
                continue

            h, w = data.shape
            if not (0 <= x < w and 0 <= y < h):
                continue

            # Ensure image exists on disk (save once per frame)
            if not img_path.exists():
                np.savez_compressed(img_path, data=data.astype(np.float32))

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

            obj = {
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "x": float(x),
                "y": float(y),
                "mag": mag,
            }

            _ = add_object_to_label(label, obj)
            save_label(lbl_path, label)
            saved_frames.add(fid)

            # If you want to limit how many frames you store, check after additions
            if MAX_SAVED_FRAMES is not None and len(saved_frames) >= MAX_SAVED_FRAMES:
                break

    # Always write splits from what exists on disk
    write_splits_from_labels()
    print(f"Done. Unique frames with labels: {len({p.stem for p in LABELS_DIR.glob('*.json')})}")


if __name__ == "__main__":
    main()
