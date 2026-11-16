"""Read-only, bitwise comparison helpers for global_check.ipynb.

The fast loader scans every global partition's positions, then reads only the
selected NATRE columns. It never reads the reference to select/order global data.
No processing, bootstrap, rounding, alignment, or data-volume writes occur here.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def natre_mask(ds):
    return (
        (ds.time >= np.datetime64("2014-01-01"))
        & (ds.time < np.datetime64("2024-01-01"))
        & (ds.latitude >= 19) & (ds.latitude < 32)
        & (ds.longitude >= -38) & (ds.longitude < -20)
    )


def canonical_order(ds):
    """Argopy time/physical-UID ordering, independent of reference values."""
    uid = (ds.platform_number.values.astype(np.int64) * 100000
           + ds.cycle_number.values.astype(np.int64))
    uid *= np.where(ds.direction.values.astype(str) == "A", 1, -1)
    return ds.isel(time=np.lexsort((uid, ds.time.values)))


def _scan_paths(paths):
    # netCDF4/HDF5 reads run in separate processes, never concurrent threads.
    from netCDF4 import Dataset, date2num

    selected, count, spatial_files = [], 0, 0
    for path in paths:
        with Dataset(path) as ds:
            latitude = np.ma.filled(ds["latitude"][:], np.nan)
            longitude = np.ma.filled(ds["longitude"][:], np.nan)
            n = len(ds.dimensions["time"])
            if latitude.shape != (n,) or longitude.shape != (n,):
                raise ValueError(f"Invalid profile coordinates: {path}")
            count += n
            spatial = ((latitude >= 19) & (latitude < 32)
                       & (longitude >= -38) & (longitude < -20))
            if not spatial.any():
                continue
            spatial_files += 1
            time = ds["time"]
            calendar = getattr(time, "calendar", "standard")
            if calendar not in ("standard", "gregorian", "proleptic_gregorian"):
                raise ValueError(f"Unexpected time calendar: {path}: {calendar}")
            start, end = date2num([datetime(2014, 1, 1), datetime(2024, 1, 1)],
                                 time.units, calendar=calendar)
            t = np.ma.filled(time[:].astype(np.float64), np.nan)
            indices = np.flatnonzero(spatial & (t >= start) & (t < end))
            if indices.size:
                selected.append((path, indices))
    return selected, count, spatial_files


def load_global_natre(manifest_path, workers=4):
    """Equivalent to concatenating the full manifest and applying natre_mask.

    Scan all files, including cells outside NATRE, so this does not assume the
    storage-cell labels are correct. Keep manifest/within-file record order.
    """
    manifest_path = Path(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest["status"] != "complete" or manifest["failed_batches"]:
        raise ValueError("Global manifest is incomplete")
    paths = [p for region in sorted(manifest["regions"])
             for p in sorted(manifest["regions"][region])]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Empty or duplicated global file inventory")
    groups = [paths[i:i + 256] for i in range(0, len(paths), 256)]
    selected, scanned_profiles, scanned_files, spatial_files = [], 0, 0, 0
    print(f"Scanning coordinates in all {len(paths):,} global files...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for group, (matches, count, n_spatial) in zip(groups, pool.map(_scan_paths, groups)):
            selected.extend(matches)
            scanned_profiles += count
            scanned_files += len(group)
            spatial_files += n_spatial
            if scanned_files % 4096 == 0 or scanned_files == len(paths):
                print(f"  {scanned_files:,}/{len(paths):,} files scanned", flush=True)
    if scanned_profiles != manifest["accepted_profiles"]:
        raise AssertionError("Global record count differs from the manifest")
    if not selected:
        raise ValueError("No NATRE profiles selected")
    print(f"Reading selected columns from {len(selected):,} files...", flush=True)
    pieces = []
    for path, indices in selected:
        with xr.open_dataset(path, engine="netcdf4") as part:
            subset = part.isel(time=indices).load()
            # Independent decoded-coordinate check of the numeric CF-time mask.
            if not bool(natre_mask(subset).all()):
                raise AssertionError(f"Selection/decoded coordinate mismatch: {path}")
            pieces.append(subset)
    combined = xr.concat(pieces, dim="time", data_vars="minimal", coords="minimal",
                         compat="override", join="exact", combine_attrs="drop_conflicts")
    audit = {
        "mode": "all_manifest_files_scanned_then_selected_columns_loaded",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_files": len(paths), "scanned_files": scanned_files,
        "manifest_profiles": manifest["accepted_profiles"],
        "scanned_profiles": scanned_profiles,
        "files_with_spatial_matches": spatial_files,
        "selected_files": len(selected), "selected_profiles": combined.sizes["time"],
        "selected_storage_cells": sorted({Path(p).parent.name for p, _ in selected}),
        "selection": "2014-01-01 <= time < 2024-01-01; 19 <= lat < 32; -38 <= lon < -20",
        "selected_paths": [p for p, _ in selected],
    }
    combined.attrs["source_manifest"] = str(manifest_path)
    return combined, audit


def coverage_proof(actual, expected):
    """Retain duplicate occurrences; the legacy file has no WMO/cycle IDs."""
    def keys(ds):
        return list(zip(ds.time.values.astype("datetime64[ns]").astype(np.int64),
                        ds.longitude.values, ds.latitude.values))

    ak, ek = keys(actual), keys(expected)
    ac, ec = Counter(ak), Counter(ek)
    return {
        "global_profiles": len(ak), "espe_profiles": len(ek),
        "missing_profiles": sum((ec - ac).values()),
        "extra_profiles": sum((ac - ec).values()),
        "global_duplicate_keys": sum(v > 1 for v in ac.values()),
        "espe_duplicate_keys": sum(v > 1 for v in ec.values()),
        "same_population_with_multiplicity": ac == ec,
        "same_order_after_independent_sort": ak == ek,
    }


def array_proof(actual, expected):
    """Check ALL reference arrays positionally, without xarray alignment.

    Bytes mean decoded NumPy arrays in logical C order. Dtypes, named dimensions,
    shapes, signed zeros and NaN payloads must agree. Hashes are an additional
    auditable fingerprint; acceptance uses direct byte equality, not hashes.
    """
    rows = []
    for name in sorted(expected.variables):
        ref = expected[name]
        row = {"variable": name, "dims": str(ref.dims), "shape": str(ref.shape),
               "espe_dtype": ref.dtype.str, "global_dtype": None,
               "same_dims_shape_dtype": False, "different_elements_bits": None,
               "bitwise_equal": False, "global_sha256": None, "espe_sha256": None}
        if name not in actual:
            rows.append(row)
            continue
        got = actual[name]
        a, b = np.asarray(got.values), np.asarray(ref.values)
        row["global_dtype"] = a.dtype.str
        schema = got.dims == ref.dims and a.shape == b.shape and a.dtype == b.dtype
        row["same_dims_shape_dtype"] = schema
        if a.dtype.hasobject or b.dtype.hasobject:
            raise TypeError(f"Object arrays cannot be compared as value bytes: {name}")
        ab, bb = a.tobytes(order="C"), b.tobytes(order="C")
        row["global_sha256"] = hashlib.sha256(ab).hexdigest()
        row["espe_sha256"] = hashlib.sha256(bb).hexdigest()
        row["bitwise_equal"] = schema and ab == bb
        if schema:
            unequal = np.any(
                np.frombuffer(ab, dtype=np.uint8).reshape(-1, a.dtype.itemsize)
                != np.frombuffer(bb, dtype=np.uint8).reshape(-1, b.dtype.itemsize), axis=1)
            row["different_elements_bits"] = int(unequal.sum())
            if unequal.any():
                flat = int(np.flatnonzero(unequal)[0])
                row["first_mismatch_index"] = str(np.unravel_index(flat, a.shape))
                row["first_global_value"] = repr(a.flat[flat])
                row["first_espe_value"] = repr(b.flat[flat])
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable")


def mean_profiles(ds):
    """Unweighted depth-wise means, with per-profile Piso = chi - Pdia.

    Use independent C-contiguous copies to give both inputs the same reduction
    order without changing any input values. No clipping or bootstrap draws.
    """
    import warnings

    def values(name):
        return np.array(ds[name].transpose("Zfs", "time").values, order="C", copy=True)

    arrays = {"Krho": values("Krho"), "Pdia": values("Pdia")}
    arrays["Piso"] = values("chi") - arrays["Pdia"]
    output = {}
    for name, a in arrays.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN depth bins
            output[name] = ("Zfs", np.nanmean(a, axis=1))
        output[name + "_count"] = ("Zfs", np.sum(~np.isnan(a), axis=1))
    return xr.Dataset(output, coords={"Zfs": ds.Zfs.values.copy()})
