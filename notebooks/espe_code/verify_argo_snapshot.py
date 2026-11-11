#!/usr/bin/env python3
"""Inspect a refreshed NATRE archive without calculating mixing diagnostics.

This is a primary-sampling, unrounded-input diagnostic, separate from the
production loader. It never authorizes global processing or writes to the
archive. All reports require an explicit, new path beneath /swot/SUM05/amf2288.

Supports GDAC mono-profile files and float multi-profile files named
<WMO>_prof.nc. Missing files are reported, never fetched. Optional threads are
bounded to four; a lock serializes NetCDF reads because netCDF4/HDF5 builds are
not necessarily thread-safe.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading

for _name, _directory in {
    "MPLCONFIGDIR": "/tmp/argo-mixing-runtime/matplotlib",
    "NUMBA_CACHE_DIR": "/tmp/argo-mixing-runtime/numba",
    "XDG_CACHE_HOME": "/tmp/argo-mixing-runtime/cache",
}.items():
    os.environ.setdefault(_name, _directory)

import numpy as np
import xarray as xr


BOUNDS = (-38.0, -20.0, 19.0, 32.0)
START = np.datetime64("2014-01-01", "ns")
END = np.datetime64("2024-01-01", "ns")
PARAMETERS = ("PRES", "TEMP", "PSAL")
NETCDF_LOCK = threading.Lock()
FILE_ID = re.compile(r"[RD](\d+)_(\d+)(D?)\.nc$")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def string(value):
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").strip()
    return str(value).strip()


def identity(platform, cycle, direction):
    # CF masking promotes integer cycle numbers to float, including valid 210.0.
    # Require exact integrality rather than silently rounding malformed IDs.
    identifiers = [float(string(value)) for value in (platform, cycle)]
    if not all(np.isfinite(value) and value.is_integer() for value in identifiers):
        raise ValueError(f"Non-integral or missing physical identity: {platform}, {cycle}")
    return (str(int(identifiers[0])), str(int(identifiers[1])), string(direction))


def file_identity(filename):
    match = FILE_ID.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"Unrecognized core profile filename: {filename}")
    return (match[1], str(int(match[2])), "D" if match[3] else "A")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def selected(longitude, latitude, time):
    return bool(np.isfinite(longitude) and np.isfinite(latitude)
                and BOUNDS[0] <= longitude < BOUNDS[1]
                and BOUNDS[2] <= latitude < BOUNDS[3]
                and not np.isnat(time) and START <= time < END)


def index_selected(row):
    try:
        return ("20140101000000" <= row["date"] < "20240101000000"
                and BOUNDS[0] <= float(row["longitude"]) < BOUNDS[1]
                and BOUNDS[2] <= float(row["latitude"]) < BOUNDS[3])
    except (ValueError, TypeError):
        return False


def load_references(run, raw_directory):
    accepted, raw, files = defaultdict(list), defaultdict(list), []
    partitions = sorted((run / "legacy_raw/profiles/region").glob("*.nc"))
    if not partitions:
        raise FileNotFoundError(f"No historical replay partitions beneath {run}")
    for path in partitions:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            values = [ds[name].values for name in ("platform_number", "cycle_number", "direction")]
            for i, fields in enumerate(zip(*values)):
                accepted[identity(*fields)].append({"partition": str(path), "index": i})
        files.append({"kind": "accepted_partition", "path": str(path), "sha256": sha256(path)})
    raw_paths = sorted(raw_directory.glob("NATRE_????_????_Argo_resmode.nc"))
    if not raw_paths:
        raise FileNotFoundError(f"No saved NATRE research raw files beneath {raw_directory}")
    excluded = 0
    for path in raw_paths:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            values = [ds[name].values for name in ("PLATFORM_NUMBER", "CYCLE_NUMBER", "DIRECTION")]
            longitude, latitude, time = ds.LONGITUDE.values, ds.LATITUDE.values, ds.TIME.values
            arrays = {p: ds[p].values for p in PARAMETERS}
            provenance = {name: ds.attrs.get(name) for name in
                          ("Fetched_date", "Fetched_from", "Fetched_uri", "Processing_history")}
            for i, fields in enumerate(zip(*values)):
                if not selected(longitude[i], latitude[i], time[i]):
                    excluded += 1
                    continue
                keep = np.isfinite(arrays["PRES"][i])
                raw[identity(*fields)].append({
                    "path": str(path), "index": i,
                    "longitude": float(longitude[i]), "latitude": float(latitude[i]),
                    "time": np.datetime64(time[i], "ns"),
                    "arrays": {p: a[i, keep] for p, a in arrays.items()},
                })
        files.append({"kind": "research_raw", "path": str(path), "sha256": sha256(path), **provenance})
    missing = set(accepted) - set(raw)
    if missing:
        raise ValueError(f"Accepted identities missing from saved raw reference: {sorted(missing)[:5]}")
    return accepted, raw, files, excluded


def scan_index(gdac, expected, raw_reference_keys):
    path = gdac / "ar_index_global_prof.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Expected full frozen core index at {path}")
    header, rows, counts = [], defaultdict(list), Counter()
    relevant = set(expected) | set(raw_reference_keys)
    with path.open(newline="") as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line.startswith("#"):
                handle.seek(position)
                break
            header.append(line.rstrip())
        for row in csv.DictReader(handle):
            counts["full_index_rows"] += 1
            key = file_identity(row["file"])
            in_box = index_selected(row)
            counts["natre_index_rows"] += in_box
            if in_box or key in relevant:
                rows[key].append({**row, "selected_by_index": in_box})
    return rows, {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
                  "header": header, **dict(counts)}


def locate_source(gdac, row):
    relative = Path(row["file"])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 4:
        raise ValueError(f"Unsafe or unexpected GDAC index path: {relative}")
    platform = relative.parts[1]
    mono = [gdac / "dac" / relative, gdac / relative]
    multi = [base / relative.parts[0] / platform / f"{platform}_prof.nc"
             for base in (gdac / "dac", gdac)]
    for kind, candidates in (("mono", mono), ("multi", multi)):
        existing = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
        if len(existing) > 1:
            raise ValueError(f"Ambiguous archive layout for {relative}: {existing}")
        if existing:
            path = existing[0]
            if not path.is_relative_to(gdac.resolve()):
                raise ValueError(f"Source symlink escapes archive: {path}")
            return kind, path
    return None, None


def compare(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    same = a.shape == b.shape
    result = {"actual_levels": a.size, "reference_levels": b.size,
              "actual_dtype": str(a.dtype), "reference_dtype": str(b.dtype),
              "exact_values": same and bool(np.array_equal(a, b, equal_nan=True)),
              "bitwise_equal": same and a.dtype == b.dtype and a.tobytes() == b.tobytes()}
    if same:
        finite = np.isfinite(a) & np.isfinite(b)
        error = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
        result.update(finite_mask_mismatches=int(np.count_nonzero(np.isfinite(a) != np.isfinite(b))),
                      mismatched_values=int(np.count_nonzero(~((a == b) | (np.isnan(a) & np.isnan(b))))),
                      max_abs_error=float(error.max()) if error.size else None)
    return result


def bounded_map(function, items, workers):
    if workers == 1:
        yield from map(function, items)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        for _ in range(2 * workers):
            item = next(iterator, None)
            if item is not None:
                pending.add(pool.submit(function, item))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                item = next(iterator, None)
                if item is not None:
                    pending.add(pool.submit(function, item))


def inspect_file(task):
    """Load one source once, including multi-profile files serving many identities."""
    source, kind, requested = task
    metadata = {"path": str(source), "layout": kind, "bytes": source.stat().st_size,
                "sha256": sha256(source)}
    names = {"PLATFORM_NUMBER", "CYCLE_NUMBER", "DIRECTION", "DATA_MODE", "DATE_UPDATE",
             "JULD", "JULD_QC", "LONGITUDE", "LATITUDE", "POSITION_QC",
             "VERTICAL_SAMPLING_SCHEME", "PRES", "PRES_ADJUSTED_ERROR"}
    names |= {f"{p}{suffix}" for p in PARAMETERS for suffix in ("_ADJUSTED", "_ADJUSTED_QC")}
    try:
        with NETCDF_LOCK:
            with xr.open_dataset(source, engine="netcdf4") as opened:
                ds = opened[[name for name in names if name in opened]].load()
        metadata["date_update"] = string(ds.DATE_UPDATE.values.item()) if "DATE_UPDATE" in ds else None
        metadata["sizes"] = dict(ds.sizes)
        by_identity = defaultdict(list)
        for i, fields in enumerate(zip(ds.PLATFORM_NUMBER.values, ds.CYCLE_NUMBER.values, ds.DIRECTION.values)):
            by_identity[identity(*fields)].append(i)
        result = []
        for key, index_row in requested:
            record = {"identity": list(key), "index": index_row, "source_file": str(source),
                      "source_layout": kind, "accepted_by_primary_diagnostic": False}
            # A float multi-profile DATE_UPDATE describes the aggregate, not the
            # individual mono-profile indexed revision; do not equate them.
            record["index_source_date_update_agree"] = (
                metadata["date_update"] == index_row.get("date_update") if kind == "mono" else None)
            indices = by_identity.get(key, [])
            if not indices:
                record["reason"] = "identity_absent_from_source"
                result.append(record)
                continue
            schemes = {i: string(ds.VERTICAL_SAMPLING_SCHEME.values[i])
                       if "VERTICAL_SAMPLING_SCHEME" in ds else "" for i in indices}
            primary = [i for i in indices if schemes[i].lower().startswith("primary sampling:")]
            if len(primary) == 1:
                chosen, policy = primary[0], "explicit_primary_sampling_label"
            elif len(indices) == 1 and not primary:
                # A sole record with an explicit secondary label is not primary.
                chosen = indices[0]
                if schemes[chosen].lower().startswith(("secondary sampling:", "near-surface sampling:")):
                    record.update(reason="only_secondary_sampling_available", sampling_schemes=schemes)
                    result.append(record)
                    continue
                policy = "sole_sampling_record_without_primary_label"
            else:
                record.update(reason="ambiguous_primary_sampling", sampling_schemes=schemes)
                result.append(record)
                continue
            record.update(sampling_profile_index=chosen, sampling_selection=policy,
                          sampling_schemes=schemes, sampling_profiles_for_identity=len(indices))
            row = ds.isel(N_PROF=chosen)
            required = {"DATA_MODE", "POSITION_QC", "JULD_QC", "JULD", "LONGITUDE", "LATITUDE",
                        "PRES_ADJUSTED_ERROR"}
            required |= {f"{p}{suffix}" for p in PARAMETERS for suffix in ("_ADJUSTED", "_ADJUSTED_QC")}
            missing = required - set(row.variables)
            if missing:
                record.update(reason="missing_required_variables", missing_variables=sorted(missing))
                result.append(record)
                continue
            mode = string(row.DATA_MODE.values.item())
            longitude, latitude = float(row.LONGITUDE), float(row.LATITUDE)
            time = np.datetime64(row.JULD.values, "ns")
            record.update(data_mode=mode, longitude=longitude, latitude=latitude,
                          time=str(time), position_qc=string(row.POSITION_QC.values.item()),
                          time_qc=string(row.JULD_QC.values.item()))
            if not selected(longitude, latitude, time):
                record["reason"] = "outside_actual_region_or_time"
            elif mode != "D":
                record["reason"] = "not_delayed_mode"
            elif record["position_qc"] != "1" or record["time_qc"] != "1":
                record["reason"] = "position_or_time_qc"
            else:
                arrays = {p: np.asarray(row[f"{p}_ADJUSTED"].values).reshape(-1) for p in PARAMETERS}
                keep = np.ones(arrays["PRES"].shape, dtype=bool)
                for p in PARAMETERS:
                    keep &= np.isfinite(arrays[p])
                    keep &= np.asarray(row[f"{p}_ADJUSTED_QC"].values).astype(str).reshape(-1) == "1"
                keep &= np.asarray(row.PRES_ADJUSTED_ERROR.values).reshape(-1) < 20
                # Saved research ERDDAP queries used inclusive pressure bounds.
                keep &= (arrays["PRES"] >= 0) & (arrays["PRES"] <= 2200)
                retained = {p: a[keep] for p, a in arrays.items()}
                order = np.argsort(retained["PRES"], kind="stable")
                record["pressure_sort_changed_order"] = not np.array_equal(order, np.arange(len(order)))
                retained = {p: a[order] for p, a in retained.items()}
                p = retained["PRES"]
                delta = np.abs(np.diff(p))[p[:-1] < 1000]
                spacing = float(np.mean(delta)) if delta.size else None
                record.update(retained_levels=len(p), mean_spacing_upper_1000_dbar=spacing,
                              repeated_pressure_levels=int(np.count_nonzero(np.diff(p) == 0)),
                              accepted_by_primary_diagnostic=spacing is not None and spacing <= 5,
                              reason="accepted" if spacing is not None and spacing <= 5 else "spacing_or_insufficient_levels",
                              _arrays=retained)
            result.append(record)
        return metadata, result
    except Exception as exc:
        metadata["read_error"] = f"{type(exc).__name__}: {exc}"
        return metadata, [{"identity": list(key), "index": row, "source_file": str(source),
                           "accepted_by_primary_diagnostic": False, "reason": "source_read_error",
                           "error": metadata["read_error"]} for key, row in requested]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdac", type=Path, required=True, help="Extracted archive root containing ar_index_global_prof.txt")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    gdac, output = args.gdac.resolve(), args.output.resolve()
    if not output.is_relative_to(Path("/swot/SUM05/amf2288")) or output.is_relative_to(gdac):
        raise ValueError("Report must be beneath /swot/SUM05/amf2288 and outside the inspected archive")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Report parent directory must already exist: {output.parent}")
    expected, reference_raw, reference_files, excluded = load_references(args.reference_run, args.reference_raw)
    inventory, index_provenance = scan_index(gdac, expected, reference_raw)
    groups, sources, records = {}, [], []
    missing_files, duplicate_index = [], []
    for key, index_rows in sorted(inventory.items()):
        if len(index_rows) != 1:
            duplicate_index.append({"identity": list(key), "rows": index_rows})
            continue
        row = index_rows[0]
        kind, path = locate_source(gdac, row)
        if path is None:
            missing_files.append({"identity": list(key), "index": row})
            continue
        if path not in groups:
            groups[path] = (path, kind, [])
        groups[path][2].append((key, row))
    progress = 0
    for source, inspected in bounded_map(inspect_file, groups.values(), args.workers):
        sources.append(source)
        for record in inspected:
            key = tuple(record["identity"])
            arrays = record.pop("_arrays", None)
            if arrays is not None and key in reference_raw:
                comparisons = []
                for reference in reference_raw[key]:
                    comparison = {
                        "reference_path": reference["path"], "reference_profile_index": reference["index"],
                        "variables": {p: compare(arrays[p], reference["arrays"][p]) for p in PARAMETERS},
                        "longitude_exact": record["longitude"] == reference["longitude"],
                        "latitude_exact": record["latitude"] == reference["latitude"],
                        "time_delta_ns": int(np.datetime64(record["time"], "ns").astype(np.int64)
                                             - reference["time"].astype(np.int64)),
                    }
                    comparison["all_input_values_exact"] = (
                        all(v["exact_values"] for v in comparison["variables"].values())
                        and comparison["longitude_exact"] and comparison["latitude_exact"]
                        and comparison["time_delta_ns"] == 0)
                    comparisons.append(comparison)
                record["saved_raw_comparisons"] = comparisons
            records.append(record)
        progress += 1
        if progress % 100 == 0:
            print(f"Inspected {progress}/{len(groups)} source files", file=sys.stderr, flush=True)
    records.sort(key=lambda r: r["identity"])
    sources.sort(key=lambda s: s["path"])
    accepted = Counter(tuple(r["identity"]) for r in records if r["accepted_by_primary_diagnostic"])
    wanted = Counter({key: len(rows) for key, rows in expected.items()})
    missing_expected, extra = wanted - accepted, accepted - wanted
    comparisons = [comparison for record in records for comparison in record.get("saved_raw_comparisons", [])]
    source_errors = [s for s in sources if "read_error" in s]
    coverage_complete = not missing_files and not duplicate_index and not source_errors
    absent_index = sorted(set(expected) - set(inventory))
    summary = {
        "expected_accepted_profiles": sum(wanted.values()), "expected_unique_identities": len(wanted),
        "saved_in_range_raw_profiles": sum(map(len, reference_raw.values())), "saved_raw_profiles_excluded": excluded,
        "full_index_natre_candidates": index_provenance.get("natre_index_rows", 0),
        "inspected_source_files": len(sources), "inspected_identities": len(records),
        "source_layout_counts": dict(Counter(s["layout"] for s in sources)),
        "missing_source_files": len(missing_files), "duplicate_index_identities": len(duplicate_index),
        "source_read_errors": len(source_errors), "expected_identities_absent_from_full_index": len(absent_index),
        "expected_identities_in_index_outside_natre_selection": sum(
            key in expected and not any(r["selected_by_index"] for r in rows) for key, rows in inventory.items()),
        "primary_diagnostic_accepted_profiles": sum(accepted.values()),
        "expected_profiles_missing_after_primary_diagnostic": sum(missing_expected.values()),
        "extra_profiles_after_primary_diagnostic": sum(extra.values()),
        "identity_population_matches": coverage_complete and not missing_expected and not extra,
        "reason_counts": dict(Counter(r["reason"] for r in records)),
        "primary_data_mode_counts": dict(Counter(r.get("data_mode", "unavailable") for r in records)),
        "mono_file_index_date_update_mismatches": sum(r.get("index_source_date_update_agree") is False for r in records),
        "raw_profile_comparisons": len(comparisons),
        "all_raw_input_values_exact_comparisons": sum(c["all_input_values_exact"] for c in comparisons),
        "raw_array_exact_counts": {p: sum(c["variables"][p]["exact_values"] for c in comparisons) for p in PARAMETERS},
        "raw_array_bitwise_counts": {p: sum(c["variables"][p]["bitwise_equal"] for c in comparisons) for p in PARAMETERS},
    }
    report = {
        "diagnostic_only": True, "global_authorized": False, "scientific_validation_gate_evaluated": False,
        "production_modified": False, "archive_modified": False, "summary": summary,
        "method": {
            "population": "All NATRE index rows plus every saved raw/accepted reference identity independently located in the full index",
            "bounds": list(BOUNDS), "time_start_inclusive": str(START), "time_end_exclusive": str(END),
            "sampling": "One explicitly labeled primary entry per physical identity, or a sole entry without an explicit secondary label; ambiguities reported",
            "research_qc": "DATA_MODE D; POSITION_QC and JULD_QC 1; adjusted P/T/S finite and all adjusted QC 1; unrounded PRES_ADJUSTED_ERROR <20; 0<=PRES_ADJUSTED<=2200",
            "pressure": "Unrounded adjusted pressure, stably sorted ascending; no duplicate-row removal",
            "spacing": "mean(abs(diff(PRES))[PRES[:-1]<1000]) <=5 dbar using source dtype",
            "precision": "No timestamp normalization, salinity rounding, or tolerance relaxation",
            "reference_snapshot": "Saved ERDDAP response bytes define the reference; Fetched_date is not an immutable GDAC snapshot identifier",
            "global_availability": "Only selected candidate/reference files are checked; this does not establish global archive completeness",
            "mutable_source_warning": "A frozen index alone does not freeze subsequently downloaded files; content hashes and mono-profile DATE_UPDATE comparisons are recorded",
            "multi_profile_version_note": "Aggregate float DATE_UPDATE is not comparable to individual indexed mono-profile DATE_UPDATE",
            "scope_limit": "Does not inspect other globally indexed identities outside NATRE/reference selections; does not reproduce all server implementation details such as distinct()",
        },
        "provenance": {
            "gdac": str(gdac), "index": index_provenance, "reference_files": reference_files,
            "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
            "python": sys.version, "numpy": np.__version__, "xarray": xr.__version__, "workers": args.workers,
        },
        "expected_identities_absent_from_full_index": [list(k) for k in absent_index],
        "missing_expected_profiles": [{"identity": list(k), "occurrences": n} for k, n in sorted(missing_expected.items())],
        "extra_profiles": [{"identity": list(k), "occurrences": n} for k, n in sorted(extra.items())],
        "missing_files": missing_files, "duplicate_index_identities": duplicate_index,
        "sources": sources, "profiles": records,
    }
    with output.open("x") as handle:
        json.dump(json_safe(report), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(json_safe({"output": str(output), "summary": summary}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
