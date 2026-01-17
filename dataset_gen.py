import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from astroquery.sdss import SDSS
from astroquery.gaia import Gaia
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

# Stop conditions
MAX_SAVED_FRAMES = 10000   # set None for unlimited
MAX_ATTEMPTS = 100000      # set None for unlimited

# SDSS query controls
MAX_XID = 10
MAX_ARCMIN = 120
MIN_ARCMIN = 0.2
MAX_ITERS = 60

# Network robustness
REMOTE_TIMEOUT_SECONDS = 90  # download timeout for SDSS frames

# =======================
# Gaia DR3 labeling (gradual switch)
# =======================
USE_GAIA_LABELS = True            # enable Gaia labeling
KEEP_HYG_LABELS = False           # keep HYG labels too (usually False once Gaia works)
GAIA_ROW_LIMIT = 2000             # max Gaia sources per frame
GAIA_RADIUS_MARGIN_FACTOR = 1.05  # slightly enlarge search radius
GAIA_MAG_LIMIT = 19.0             # None = no filter; typical 18-20 to limit size/time

# Dedupe precision for non-Gaia objects (catalog + RA/Dec rounded)
DEDUP_ROUND_DEG = 6  # 1e-6 deg ~ 0.0036 arcsec


# =======================
# Utility functions
# =======================
def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_dynamic_radius(mag: float) -> float:
    """Returns an SDSS query radius in arcminutes based on star brightness."""
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
    """Find an SDSS match list with size between 1 and MAX_XID if possible."""
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
    """Deduplicate SDSS.get_images() results by (RUN, CAMCOL, FRAME)."""
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


def _rounded_key(obj: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """
    Generic dedupe key for objects without stable IDs:
    (catalog, rounded_ra, rounded_dec)
    """
    cat = obj.get("catalog")
    ra = obj.get("ra_deg")
    dec = obj.get("dec_deg")
    if cat is None or ra is None or dec is None:
        return None
    try:
        return (
            cat,
            round(float(ra), DEDUP_ROUND_DEG),
            round(float(dec), DEDUP_ROUND_DEG),
        )
    except Exception:
        return None


def add_object_to_label(label: Dict[str, Any], obj: Dict[str, Any]) -> bool:
    """
    Append an object to label["objects"] unless already present.
    Dedupe rules:
    - Gaia: (catalog="gaia_dr3", source_id)
    - Otherwise: (catalog, rounded ra/dec)
    Returns True if appended, False if skipped.
    """
    objs = label.setdefault("objects", [])

    # Dedupe Gaia by stable source_id
    if obj.get("catalog") == "gaia_dr3" and obj.get("source_id") is not None:
        gid = obj["source_id"]
        for existing in objs:
            if existing.get("catalog") == "gaia_dr3" and existing.get("source_id") == gid:
                return False
        objs.append(obj)
        return True

    # Fallback dedupe by rounded (catalog, ra, dec)
    k = _rounded_key(obj)
    if k is not None:
        for existing in objs:
            ek = _rounded_key(existing)
            if ek == k:
                return False

    objs.append(obj)
    return True


def write_splits_from_labels() -> None:
    """Deterministic train/val split based on frame_id hash. Uses labels on disk."""
    label_files = sorted(LABELS_DIR.glob("*.json"))
    train_ids = []
    val_ids = []

    for p in label_files:
        fid = p.stem
        h = int(hashlib.md5(fid.encode("utf-8")).hexdigest(), 16) % 10
        if h == 0:
            val_ids.append(fid)
        else:
            train_ids.append(fid)

    (SPLITS_DIR / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (SPLITS_DIR / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    print(f"Splits written from disk: train={len(train_ids)} val={len(val_ids)} total={len(label_files)}")


def gaia_sources_for_frame(wcs: WCS, width: int, height: int):
    """
    Query Gaia DR3 sources for an approximate cone covering the SDSS frame.
    Returns an astropy Table (or None on failure).
    """
    # Frame center
    center = wcs.pixel_to_world(width / 2.0, height / 2.0)
    ra_c = float(center.ra.deg)
    dec_c = float(center.dec.deg)

    # Approximate half-diagonal radius (SDSS pixel scale ~0.396 arcsec/pixel)
    pix_scale_deg = 0.396 / 3600.0
    half_diag_pix = 0.5 * float((width**2 + height**2) ** 0.5)
    radius_deg = GAIA_RADIUS_MARGIN_FACTOR * half_diag_pix * pix_scale_deg

    # Control result size
    Gaia.ROW_LIMIT = GAIA_ROW_LIMIT

    coord = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg, frame="icrs")
    try:
        job = Gaia.cone_search_async(coord, radius=radius_deg * u.deg)
        tbl = job.get_results()
    except Exception:
        return None

    if tbl is None:
        return None

    # Optional magnitude filter (Gaia G)
    if GAIA_MAG_LIMIT is not None and "phot_g_mean_mag" in tbl.colnames:
        try:
            tbl = tbl[tbl["phot_g_mean_mag"] <= GAIA_MAG_LIMIT]
        except Exception:
            pass

    return tbl


# =======================
# Main generator
# =======================
def main():
    ensure_dirs()
    astropy_conf.remote_timeout = REMOTE_TIMEOUT_SECONDS

    # Load HYG catalog once (only needed if KEEP_HYG_LABELS=True)
    star_df = None
    if KEEP_HYG_LABELS:
        star_df = pd.read_csv(CATALOG_PATH)
        star_df["ra_deg"] = star_df["ra"].astype(float) * 15.0
        star_df["dec_deg"] = star_df["dec"].astype(float)

    # Track existing frames (labels already on disk)
    saved_frames = {p.stem for p in LABELS_DIR.glob("*.json")}

    attempts = 0

    # If we are not using HYG at all, we still need "attempts" to drive SDSS frame discovery.
    # We can reuse HYG catalog as a convenient list of sky positions to probe SDSS.
    # So we still read it for iteration, but we won't label with star_id.
    probe_df = pd.read_csv(CATALOG_PATH)
    probe_df["ra_deg"] = probe_df["ra"].astype(float) * 15.0
    probe_df["dec_deg"] = probe_df["dec"].astype(float)

    for star in tqdm(probe_df.itertuples(index=False), total=len(probe_df)):
        attempts += 1
        if MAX_ATTEMPTS is not None and attempts > MAX_ATTEMPTS:
            break

        if SKIP_SUN and int(star.id) == 0:
            continue

        if MAX_SAVED_FRAMES is not None and len(saved_frames) >= MAX_SAVED_FRAMES:
            break

        ra_deg = float(star.ra_deg)
        dec_deg = float(star.dec_deg)
        mag = float(star.mag)

        xid, _radius = find_reasonable_matches(ra_deg, dec_deg, mag)
        if xid is None:
            continue

        # Download candidate SDSS frames (can timeout)
        try:
            images = SDSS.get_images(matches=xid, band=BAND)
        except TimeoutError:
            continue
        except Exception:
            continue

        if not images:
            continue

        images_reduced = dedupe_images(images)

        for (run, camcol, frame), hdul in images_reduced.items():
            if MAX_SAVED_FRAMES is not None and len(saved_frames) >= MAX_SAVED_FRAMES:
                break

            fid = frame_id(run, camcol, frame, BAND)
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

            changed = False

            # -----------------------
            # Gaia labeling (primary)
            # -----------------------
            if USE_GAIA_LABELS:
                gaia_tbl = gaia_sources_for_frame(wcs, w, h)
                if gaia_tbl is not None and len(gaia_tbl) > 0:
                    for row in gaia_tbl:
                        try:
                            ra_g = float(row["ra"])
                            dec_g = float(row["dec"])
                            gx, gy = world_to_pixel(wcs, ra_g, dec_g)
                        except Exception:
                            continue

                        if not (0 <= gx < w and 0 <= gy < h):
                            continue

                        obj = {
                            "catalog": "gaia_dr3",
                            "source_id": int(row["source_id"]) if "source_id" in row.colnames else None,
                            "ra_deg": ra_g,
                            "dec_deg": dec_g,
                            "x": float(gx),
                            "y": float(gy),
                            "phot_g_mean_mag": float(row["phot_g_mean_mag"]) if "phot_g_mean_mag" in row.colnames else None,
                        }
                        if add_object_to_label(label, obj):
                            changed = True

            # -----------------------
            # Optional: HYG labeling (secondary / transitional)
            # NOTE: No star_id stored (to avoid numbering); uses catalog+ra/dec dedupe.
            # -----------------------
            if KEEP_HYG_LABELS and star_df is not None:
                # Use the frame corners to get bbox and filter candidates.
                # Simple bbox with RA wrap handling.
                try:
                    xs = np.array([0, w - 1, 0, w - 1], dtype=float)
                    ys = np.array([0, 0, h - 1, h - 1], dtype=float)
                    sky = wcs.pixel_to_world(xs, ys)
                    ra_vals = sky.ra.deg
                    dec_vals = sky.dec.deg
                    ra_min, ra_max = float(np.min(ra_vals)), float(np.max(ra_vals))
                    dec_min, dec_max = float(np.min(dec_vals)), float(np.max(dec_vals))
                except Exception:
                    ra_min = ra_max = dec_min = dec_max = None

                if ra_min is not None:
                    margin = 0.05
                    ra_min -= margin
                    ra_max += margin
                    dec_min -= margin
                    dec_max += margin

                    dec_mask = (star_df["dec_deg"] >= dec_min) & (star_df["dec_deg"] <= dec_max)
                    if ra_min <= ra_max:
                        ra_mask = (star_df["ra_deg"] >= ra_min) & (star_df["ra_deg"] <= ra_max)
                    else:
                        ra_mask = (star_df["ra_deg"] >= ra_min) | (star_df["ra_deg"] <= ra_max)

                    cands = star_df[dec_mask & ra_mask]

                    for row in cands.itertuples(index=False):
                        if SKIP_SUN and int(row.id) == 0:
                            continue
                        try:
                            sx, sy = world_to_pixel(wcs, float(row.ra_deg), float(row.dec_deg))
                        except Exception:
                            continue
                        if not (0 <= sx < w and 0 <= sy < h):
                            continue

                        obj = {
                            "catalog": "hyg",
                            "ra_deg": float(row.ra_deg),
                            "dec_deg": float(row.dec_deg),
                            "x": float(sx),
                            "y": float(sy),
                            "mag": float(row.mag),
                        }
                        if add_object_to_label(label, obj):
                            changed = True

            # Save label if changed or new frame label
            if changed or (fid not in saved_frames):
                save_label(lbl_path, label)
                saved_frames.add(fid)

    # Always write splits from what exists on disk
    write_splits_from_labels()
    print(f"Done. Unique frames with labels: {len({p.stem for p in LABELS_DIR.glob('*.json')})}")


if __name__ == "__main__":
    main()