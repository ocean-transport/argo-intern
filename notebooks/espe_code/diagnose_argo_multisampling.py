#!/usr/bin/env python3
"""Audit NATRE profiles lost when primary and secondary sampling are merged.

This diagnostic reads completed results and source files. It calls the current
local loader, but does not calculate mixing metrics or alter production outputs.
Its optional JSON report must be written outside the home directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for _name, _directory in {
    "MPLCONFIGDIR": "/tmp/argo-mixing-runtime/matplotlib",
    "NUMBA_CACHE_DIR": "/tmp/argo-mixing-runtime/numba",
    "XDG_CACHE_HOME": "/tmp/argo-mixing-runtime/cache",
}.items():
    os.environ.setdefault(_name, _directory)

import numpy as np
import xarray as xr


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spacing_info(values):
    """Use the production mean-spacing formula, retaining existing row order."""
    values = np.asarray(values).reshape(-1)
    delta = np.abs(np.diff(values))
    candidates = delta[values[:-1] < 1000]
    finite = values[np.isfinite(values)]
    mean = float(np.nanmean(candidates)) if np.any(np.isfinite(candidates)) else None
    jumps = np.flatnonzero(np.diff(finite) < 0)
    return {
        "finite_levels": len(finite),
        "mean_spacing_upper_1000_dbar": mean,
        "passes_5_dbar_limit": mean is not None and mean <= 5,
        "negative_pressure_jumps": len(jumps),
        "negative_jump_examples": [
            {"index": int(i), "before": float(finite[i]), "after": float(finite[i + 1])}
            for i in jumps[:3]
        ],
        "head": finite[:7].tolist(),
        "tail": finite[-7:].tolist(),
    }


def compare(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    same_shape = actual.shape == expected.shape
    same_dtype = actual.dtype == expected.dtype
    answer = {
        "actual_shape": list(actual.shape),
        "reference_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(expected.dtype),
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "exact_values": same_shape and bool(np.array_equal(actual, expected, equal_nan=True)),
        "bitwise_equal": same_shape and same_dtype and actual.tobytes() == expected.tobytes(),
    }
    if same_shape:
        finite = np.isfinite(actual) & np.isfinite(expected)
        difference = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
        answer.update(
            mismatched_values=int(np.count_nonzero(~((actual == expected) | (np.isnan(actual) & np.isnan(expected))))),
            max_abs_error=float(difference.max()) if len(difference) else None,
            finite_mask_mismatches=int(np.count_nonzero(np.isfinite(actual) != np.isfinite(expected))),
        )
    return answer


def load_references(directory, wanted):
    result, files = {}, []
    for path in sorted(directory.glob("NATRE_????_????_Argo_resmode.nc")):
        with xr.open_dataset(path) as ds:
            keys = zip(ds.PLATFORM_NUMBER.values, ds.CYCLE_NUMBER.values, ds.DIRECTION.values)
            for i, (platform, cycle, direction) in enumerate(keys):
                key = (str(platform), str(cycle), str(direction))
                if key not in wanted:
                    continue
                arrays = {v: ds[v].isel(N_PROF=i).values for v in ("PRES", "TEMP", "PSAL")}
                mask = np.isfinite(arrays["PRES"])
                if key in result:
                    raise ValueError(f"Ambiguous duplicate reference identity: {key}")
                result[key] = {
                    "arrays": {v: a[mask] for v, a in arrays.items()},
                    "file": str(path), "profile_index": i,
                }
        files.append({"path": str(path), "sha256": digest(path)})
    missing = wanted - result.keys()
    if missing:
        raise ValueError(f"Missing raw reference identities: {sorted(missing)[:5]}")
    return result, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--gdac", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--current-samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.output.resolve().is_relative_to(Path.home()):
        raise ValueError("Write diagnostic reports outside the home directory")

    script = Path(__file__).with_name("process_argo_mixing.py")
    spec = importlib.util.spec_from_file_location("argo_mixing_audit_target", script)
    mixing = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mixing
    spec.loader.exec_module(mixing)
    audit = json.loads(args.identity_report.read_text())
    rows = [row for row in audit["reference_only"]
            if row["reason"] == "selected_but_not_accepted_by_profile_kernel"]
    references, reference_files = load_references(args.reference_dir, {tuple(r["identity"]) for r in rows})
    config = {"bounds": [-38, -20, 19, 32], "start": "2014-01-01",
              "end": "2024-01-01", "max_pressure": 2200}
    counts, records = Counter(), []
    for number, row in enumerate(rows, start=1):
        key = tuple(row["identity"])
        relative = row["local_index"]["path"]
        path = args.gdac / "dac" / relative
        local, accounting = mixing._load_local_records(
            [{"path": str(path), "source_file": relative}], config
        )
        if local is None or local.sizes["N_PROF"] != 1:
            raise ValueError(f"Expected one merged local profile for {key}")
        reference = references[key]
        with xr.open_dataset(path) as ds:
            arrays = {v: ds[f"{v}_ADJUSTED"].values for v in ("PRES", "TEMP", "PSAL")}
            # Reconstruct the existing loader's QC/pressure mask, but retain the
            # source sampling axis and unrounded scientific arrays for diagnosis.
            mask = np.isfinite(ds.PRES.values)
            mask &= (np.round(arrays["PRES"], 1) >= 0) & (np.round(arrays["PRES"], 1) < 2200)
            mask &= np.round(ds.PRES_ADJUSTED_ERROR.values, 1) < 20
            for variable in ("PRES", "TEMP", "PSAL"):
                mask &= ds[f"{variable}_ADJUSTED_QC"].values.astype(str) == "1"
            for variable in ("POSITION_QC", "JULD_QC"):
                mask &= (ds[variable].values.astype(str) == "1")[:, None]
            mask &= (ds.DATA_MODE.values.astype(str) == "D")[:, None]
            primary = {v: a[0, mask[0]] for v, a in arrays.items()}
            reconstructed = np.round(arrays["PRES"][mask], 1)
            merged = local.PRES.values.reshape(-1)
            merged = merged[np.isfinite(merged)]
            if not np.array_equal(reconstructed, merged):
                raise ValueError(f"QC reconstruction disagrees with actual loader for {key}")
            schemes = [str(x).strip() for x in ds.VERTICAL_SAMPLING_SCHEME.values.astype(str)]
            if not schemes[0].startswith("Primary sampling:"):
                raise ValueError(f"First sampling profile is not primary: {key}")
            record = {
                "identity": list(key), "source_file": str(path), "source_sha256": digest(path),
                "reference_file": reference["file"], "reference_profile_index": reference["profile_index"],
                "raw_N_PROF": ds.sizes["N_PROF"], "raw_N_LEVELS": ds.sizes["N_LEVELS"],
                "data_mode": ds.DATA_MODE.values.astype(str).tolist(),
                "sampling_schemes": schemes, "qc_retained_levels_by_sampling_profile": mask.sum(axis=1).tolist(),
                "actual_loader_pressure_matches_reconstructed_mask": True,
                "actual_local_loader": spacing_info(local.PRES.values),
                "unrounded_primary": spacing_info(primary["PRES"]),
                "reference": spacing_info(reference["arrays"]["PRES"]),
                "unrounded_primary_vs_reference": {
                    v: compare(primary[v], reference["arrays"][v]) for v in arrays
                },
                "loader_accounting": accounting,
            }
            if args.current_samples and (args.current_samples / path.name).exists():
                current_path = args.current_samples / path.name
                current_digest = digest(current_path)
                record["current_download"] = {
                    "path": str(current_path), "sha256": current_digest,
                    "byte_identical_to_old_local_file": current_digest == record["source_sha256"],
                }
            # This is a diagnostic hypothesis, never a production transform.
            a, b = primary["PSAL"], reference["arrays"]["PSAL"]
            if a.shape == b.shape:
                rounded = np.round(a.astype(np.float64), 5).astype(a.dtype)
                record["salinity_five_decimal_probe"] = compare(rounded, b)
                record["salinity_five_decimal_probe"]["remaining_examples"] = [
                    {"index": int(i), "raw": float(a[i]), "rounded": float(rounded[i]), "reference": float(b[i])}
                    for i in np.flatnonzero(rounded != b)[:3]
                ]
        counts["profiles"] += 1
        counts["two_sampling_profiles"] += record["raw_N_PROF"] == 2
        counts["two_retained_secondary_points"] += sum(record["qc_retained_levels_by_sampling_profile"][1:]) == 2
        counts["actual_local_loader_rejects_spacing"] += not record["actual_local_loader"]["passes_5_dbar_limit"]
        counts["unrounded_primary_passes_spacing"] += record["unrounded_primary"]["passes_5_dbar_limit"]
        for variable, comparison in record["unrounded_primary_vs_reference"].items():
            counts[f"primary_{variable}_exact_values"] += comparison["exact_values"]
            counts[f"primary_{variable}_bitwise_equal"] += comparison["bitwise_equal"]
        records.append(record)
        if number % 50 == 0:
            print(f"Audited {number}/{len(rows)} profiles", flush=True)

    report = {
        "scope": "All historical accepted identities selected locally but rejected by the profile spacing kernel",
        "production_outputs_modified": False,
        "method": {
            "local": "Actual process_argo_mixing._load_local_records, scientific metrics not executed",
            "primary": "Source N_PROF=0 confirmed Primary sampling; existing rounded-pressure/QC mask reconstructed and checked against actual loader; unrounded adjusted P/T/S retained",
            "spacing": "nanmean(abs(diff(PRES))[PRES[:-1] < 1000]), accepted when <=5 dbar",
            "equality": "Exact numeric equality with equal NaNs, plus separate dtype-and-byte equality",
            "caveat": "Reference selection of primary sampling is observed in this population; the server-side rule is not independently proven by this audit",
        },
        "provenance": {
            "diagnostic_script": str(Path(__file__).resolve()), "diagnostic_script_sha256": digest(__file__),
            "production_script": str(script.resolve()), "production_script_sha256": digest(script),
            "identity_report": str(args.identity_report), "identity_report_sha256": digest(args.identity_report),
            "reference_files": reference_files, "python": sys.version,
            "numpy": np.__version__, "xarray": xr.__version__, "loader_config": config,
        },
        "summary": dict(counts),
        "spacing_ranges_dbar": {
            label: [min(r[label]["mean_spacing_upper_1000_dbar"] for r in records),
                    max(r[label]["mean_spacing_upper_1000_dbar"] for r in records)]
            for label in ("actual_local_loader", "unrounded_primary", "reference")
        },
        "profiles": records,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "summary": report["summary"],
                      "spacing_ranges_dbar": report["spacing_ranges_dbar"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
