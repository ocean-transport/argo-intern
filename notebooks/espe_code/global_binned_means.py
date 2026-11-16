"""Stream saved global mixing profiles into 3-degree pressure/density bins.

No profile processing, interpolation onto surfaces, or bootstrap is performed.
Each finite saved Zfs estimate contributes once to its spatial/vertical bin.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import xarray as xr

METRICS = ("Krho", "epsilon", "chi", "Pdia", "Piso", "chiS", "PSdia", "PSiso")
UNITS = ("m2 s-1", "W kg-1", "K2 s-1", "K2 s-1", "K2 s-1",
         "(g kg-1)2 s-1", "(g kg-1)2 s-1", "(g kg-1)2 s-1")
SOURCE_METRICS = ("Krho", "epsilon", "chi", "Pdia", "chiS", "PdiaS")
NLAT, NLON = 60, 120
NCELL = NLAT * NLON
AXES = ("pressure_bin", "neutral_density_bin")


def horizontal_cells(latitude, longitude):
    """Global 3-degree cells; wrap longitude, include +90 in the last row."""
    lat, lon = np.asarray(latitude), np.asarray(longitude)
    valid = (np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
             & (lon >= -180) & (lon <= 360))
    cells = np.full(lat.shape, -1, dtype=np.int64)
    iy = np.minimum(np.floor((lat[valid] + 90) / 3).astype(np.int64), NLAT - 1)
    ix = np.floor((((lon[valid] + 180) % 360)) / 3).astype(np.int64)
    cells[valid] = iy * NLON + ix
    return cells, valid


def pressure_on_finescale(pressure, zmeas, zfs):
    """Mean pressure on the SAME local support used for stored gamman_fs.

    Production uses abs(Zmeas - Zfs) < 25 m (50-m center spacing). A bincount
    replaces the equivalent loop over centers. This is not depth-as-pressure.
    """
    p, z, centers = map(np.asarray, (pressure, zmeas, zfs))
    if p.shape != z.shape or p.ndim != 2:
        raise ValueError("pressure and Zmeas must have equal (Z, time) shapes")
    if centers.ndim != 1 or len(centers) < 2 or not np.all(np.diff(centers) == 50):
        raise ValueError("Expected the production 50-m Zfs spacing")
    nprof = p.shape[1]
    iz, ip = np.nonzero(np.isfinite(p) & np.isfinite(z))
    index = np.floor((z[iz, ip] - centers[0] + 25) / 50).astype(np.int64)
    inside = (index >= 0) & (index < len(centers))
    iz, ip, index = iz[inside], ip[inside], index[inside]
    inside = np.abs(z[iz, ip] - centers[index]) < 25  # both boundaries excluded
    iz, ip, index = iz[inside], ip[inside], index[inside]
    group = index * nprof + ip
    count = np.bincount(group, minlength=len(centers) * nprof)
    total = np.bincount(group, weights=p[iz, ip], minlength=len(centers) * nprof)
    out = np.full(count.shape, np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out.reshape(len(centers), nprof)


def vertical_indices(values, width):
    """Half-open bins [k*width, (k+1)*width), anchored at zero.

    Check against the actual floating-point edges to avoid a division rounding
    to an adjacent bin at an edge. No rounding of measured values is used.
    """
    values = np.asarray(values)
    index = np.floor(values / width).astype(np.int64)
    index -= values < index * width
    index += values >= (index + 1) * width
    return index


def reduce_samples(coordinate, values, cells, width):
    """Sparse sums/counts for one file; distinct profile records counted once/bin."""
    nprof = coordinate.shape[1]
    take = np.flatnonzero(np.isfinite(coordinate) & (cells[None, :] >= 0))
    profile = take % nprof
    bins = vertical_indices(coordinate.ravel()[take], width)
    keys, inverse = np.unique(bins * NCELL + cells[profile], return_inverse=True)
    sums = np.zeros((len(keys), len(METRICS)), dtype=np.float64)
    counts = np.zeros(sums.shape, dtype=np.int64)
    for k, value in enumerate(values):
        selected = np.asarray(value).ravel()[take]
        finite = np.isfinite(selected)
        sums[:, k] = np.bincount(inverse[finite], weights=selected[finite], minlength=len(keys))
        counts[:, k] = np.bincount(inverse[finite], minlength=len(keys))
    observations = np.bincount(inverse, minlength=len(keys)).astype(np.int64)
    distinct = np.unique(inverse * nprof + profile) // nprof
    profiles = np.bincount(distinct, minlength=len(keys)).astype(np.int64)
    return keys, sums, counts, observations, profiles


def _combine(blocks):
    keys = np.concatenate([b[0] for b in blocks])
    unique, inverse = np.unique(keys, return_inverse=True)
    result = [unique]
    for field in range(1, 5):
        values = np.concatenate([b[field] for b in blocks])
        total = np.zeros((len(unique),) + values.shape[1:], dtype=values.dtype)
        np.add.at(total, inverse, values)
        result.append(total)
    return tuple(result)


def mixing_values(raw):
    """Saved production terms and paired thermal/haline residuals before binning."""
    values = dict(raw)
    values['PSdia'] = values['PdiaS']
    for out, dissipation, production in [('Piso', 'chi', 'Pdia'), ('PSiso', 'chiS', 'PSdia')]:
        a, b = values[dissipation], values[production]
        with np.errstate(invalid='ignore', over='ignore'):
            residual = a - b
        residual[~(np.isfinite(a) & np.isfinite(b))] = np.nan
        values[out] = residual
    return [values[name] for name in METRICS]


def _reduce_files(task):
    from netCDF4 import Dataset

    paths, widths = task
    blocks = [[], []]
    profiles, invalid_positions, samples = 0, 0, 0
    cell_profiles = np.zeros(NCELL, dtype=np.int64)
    eligible = np.zeros((2, len(METRICS)), dtype=np.int64)
    coordinate_counts = np.zeros(2, dtype=np.int64)
    for path in paths:
        with Dataset(path) as d:
            def read(name):
                return np.asarray(np.ma.filled(d[name][:], np.nan), dtype=np.float64)

            lat, lon, zfs = read("latitude"), read("longitude"), read("Zfs")
            cells, valid = horizontal_cells(lat, lon)
            profiles += len(lat)
            invalid_positions += int((~valid).sum())
            cell_profiles += np.bincount(cells[valid], minlength=NCELL)
            pressure = pressure_on_finescale(read("pressure"), read("Zmeas"), zfs)
            density = read("gamman_fs")
            values = mixing_values({name: read(name) for name in SOURCE_METRICS})
            shape = (len(zfs), len(lat))
            if any(a.shape != shape for a in [pressure, density, *values]):
                raise ValueError(f"Unexpected scientific array dimensions: {path}")
            samples += int(np.prod(shape))
            for axis, (coordinate, width) in enumerate(zip((pressure, density), widths)):
                ok = np.isfinite(coordinate) & valid[None, :]
                coordinate_counts[axis] += int(ok.sum())
                eligible[axis] += [np.count_nonzero(ok & np.isfinite(v)) for v in values]
                blocks[axis].append(reduce_samples(coordinate, values, cells, width))
    return {
        "blocks": [_combine(b) for b in blocks], "files": len(paths),
        "profiles": profiles, "invalid_positions": invalid_positions,
        "samples": samples, "cell_profiles": cell_profiles,
        "eligible": eligible, "coordinate_counts": coordinate_counts,
    }


def _accumulate(storage, block):
    keys, sums, counts, observations, profiles = block
    vb, cells = np.divmod(keys, NCELL)
    for index in np.unique(vb):
        selected = vb == index
        if int(index) not in storage:
            storage[int(index)] = (
                np.zeros((NCELL, len(METRICS)), dtype=np.float64),
                np.zeros((NCELL, len(METRICS)), dtype=np.int64),
                np.zeros(NCELL, dtype=np.int64), np.zeros(NCELL, dtype=np.int64),
            )
        for target, value in zip(storage[int(index)], (sums, counts, observations, profiles)):
            target[cells[selected]] += value[selected]


def _dataset(storage, axis, width, cell_profiles):
    if not storage:
        raise ValueError(f"No finite coordinates for {axis}")
    first, last = min(storage), max(storage)
    if last - first > 2000:
        raise ValueError(f"More than 2,000 {axis} bins; inspect coordinate outliers/bin width")
    indices = np.arange(first, last + 1)
    shape = (len(indices), NLAT, NLON)
    sums = np.zeros(shape + (len(METRICS),))
    counts = np.zeros(sums.shape, dtype=np.int64)
    observations, profiles = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    for index, block in storage.items():
        k = index - first
        sums[k] = block[0].reshape(NLAT, NLON, len(METRICS))
        counts[k] = block[1].reshape(NLAT, NLON, len(METRICS))
        observations[k] = block[2].reshape(NLAT, NLON)
        profiles[k] = block[3].reshape(NLAT, NLON)
    dims = (axis, "latitude", "longitude")
    ds = xr.Dataset(coords={
        axis: (indices + .5) * width,
        "latitude": np.arange(-88.5, 90, 3), "longitude": np.arange(-178.5, 180, 3),
    })
    ds[axis + "_bounds"] = ((axis, "bounds"), np.column_stack((indices * width, (indices + 1) * width)))
    ds[axis].attrs.update(units="dbar" if axis == AXES[0] else "kg m-3",
                          bounds=axis + "_bounds", long_name=axis.replace("_", " ") + " center")
    for name, start, stop, units in [("latitude", -90, 90, "degrees_north"),
                                      ("longitude", -180, 180, "degrees_east")]:
        edges = np.arange(start, stop + 3, 3)
        ds[name + "_bounds"] = ((name, "bounds"), np.column_stack((edges[:-1], edges[1:])))
        ds[name].attrs.update(units=units, bounds=name + "_bounds")
    for k, (name, unit) in enumerate(zip(METRICS, UNITS)):
        mean = np.full(shape, np.nan)
        np.divide(sums[..., k], counts[..., k], out=mean, where=counts[..., k] > 0)
        if not np.all(np.isfinite(mean[counts[..., k] > 0])):
            raise ArithmeticError(f"Non-finite populated mean: {name}")
        ds[name] = (dims, mean, {"units": unit, "cell_methods": "time: mean Zfs: mean",
                                "weighting": "equal weight per finite saved Zfs observation"})
        ds[name + "_sum"] = (dims, sums[..., k], {"units": unit})
        ds[name + "_count"] = (dims, counts[..., k], {"units": "1"})
    ds["observation_count"] = (dims, observations, {"description": "Zfs observations with valid position and vertical coordinate, regardless of metric availability"})
    ds["profile_count"] = (dims, profiles, {"description": "Profile records with at least one valid vertical coordinate in this bin; not independent samples"})
    ds["cell_profile_count"] = (("latitude", "longitude"), cell_profiles.reshape(NLAT, NLON))
    ds.attrs.update(
        horizontal_resolution_degrees=3., vertical_bin_width=float(width),
        bin_rule="lower-inclusive, upper-exclusive; longitude wraps at 180; latitude +90 included",
        vertical_range="all observed finite coordinate bins, including empty intervening bins",
        weighting="one vote per finite Zfs observation, not one per float or per profile",
        pressure_coordinate="mean pressure where abs(Zmeas-Zfs)<25 m, matching gamman_fs local support",
        density_coordinate="saved gamman_fs; no 1000 offset added",
        Piso_definition="chi - Pdia at each saved Zfs/profile, before binning; negative values retained",
        PSdia_definition="saved PdiaS (salinity-variance diapycnal production)",
        PSiso_definition="chiS - PdiaS at each saved Zfs/profile, before binning; negative values retained",
        missing_values="NaN/Inf excluded per metric; empty means NaN, counts and sums zero",
        bootstrap="none", time_selection="all profiles in recovered global manifest",
        sampling_note="200-m mixing windows overlap at 50-m spacing; counts are not independent observations",
    )
    return ds


def _sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def compute_global_binned_means(manifest_path, output_root, pressure_width=50., density_width=.05,
                               workers=8, files_per_task=64, reuse=True):
    """Return pressure dataset, neutral-density dataset and a conservation report.

    Full NetCDF products and a complete report are saved in a content-keyed run
    directory. Matching verified outputs are reopened on subsequent executions.
    File/task reduction order is deterministic and independent of worker count.
    """
    widths = (float(pressure_width), float(density_width))
    if not all(np.isfinite(w) and w > 0 for w in widths):
        raise ValueError("Bin widths must be positive and finite")
    if workers < 1 or files_per_task < 1:
        raise ValueError("workers and files_per_task must be positive")
    manifest_path, output_root = Path(manifest_path).resolve(), Path(output_root).resolve()
    if not output_root.is_relative_to(Path('/swot/SUM05/amf2288')):
        raise ValueError("Save scientific outputs under /swot/SUM05/amf2288")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest["status"] != "complete" or manifest["failed_batches"]:
        raise ValueError("Global manifest is incomplete")
    paths = [p for region in sorted(manifest["regions"]) for p in sorted(manifest["regions"][region])]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Empty or duplicated partition inventory")
    config = {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "helper_sha256": _sha256(__file__), "widths": widths,
              "files_per_task": files_per_task, "horizontal_degrees": 3., "metrics": METRICS}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = output_root / signature[:16]
    report_path = output / 'report.json'
    if report_path.exists() and reuse:
        report = json.loads(report_path.read_text())
        if report['status'] != 'complete' or report['signature'] != signature:
            raise ValueError("Cached output provenance mismatch")
        for name, digest in report['output_sha256'].items():
            if _sha256(output / name) != digest:
                raise ValueError(f"Cached output checksum mismatch: {name}")
        print(f"Reusing verified 3-degree averages: {output}", flush=True)
        return (xr.open_dataset(output / 'pressure_means.nc', engine='netcdf4'),
                xr.open_dataset(output / 'neutral_density_means.nc', engine='netcdf4'), report)

    started = time.monotonic()
    storage = [{}, {}]
    scanned = profiles = samples = invalid = 0
    cell_profiles = np.zeros(NCELL, dtype=np.int64)
    eligible = np.zeros((2, len(METRICS)), dtype=np.int64)
    coordinate_counts = np.zeros(2, dtype=np.int64)
    groups = [paths[i:i + files_per_task] for i in range(0, len(paths), files_per_task)]
    tasks = [(group, widths) for group in groups]
    print(f"Binning {manifest['accepted_profiles']:,} profiles from {len(paths):,} files with {workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, result in enumerate(pool.map(_reduce_files, tasks)):
            scanned += result['files']; profiles += result['profiles']
            samples += result['samples']; invalid += result['invalid_positions']
            cell_profiles += result['cell_profiles']
            eligible += result['eligible']; coordinate_counts += result['coordinate_counts']
            for target, block in zip(storage, result['blocks']):
                _accumulate(target, block)
            if (k + 1) % 32 == 0 or k + 1 == len(tasks):
                print(f"  {scanned:,}/{len(paths):,} files; {profiles:,} profiles; {time.monotonic()-started:.1f} s", flush=True)
    if profiles != manifest['accepted_profiles'] or scanned != len(paths):
        raise AssertionError("Input coverage differs from the complete manifest")
    assert int(cell_profiles.sum()) == profiles - invalid
    datasets = [_dataset(s, name, width, cell_profiles) for s, name, width in zip(storage, AXES, widths)]
    conservation = {}
    for i, (name, ds) in enumerate(zip(AXES, datasets)):
        assert int(ds.observation_count.sum()) == int(coordinate_counts[i])
        assert bool((ds.profile_count <= ds.cell_profile_count).all())
        details = {}
        for j, metric in enumerate(METRICS):
            count = ds[metric + '_count']
            assert int(count.sum()) == int(eligible[i, j])
            assert bool((count <= ds.observation_count).all())
            assert np.array_equal(np.isnan(ds[metric].values), count.values == 0)
            details[metric] = {"eligible_input_observations": int(eligible[i, j]),
                               "binned_observations": int(count.sum()), "passed": True}
        conservation[name] = details
        ds.attrs.update(source_manifest=str(manifest_path), source_manifest_sha256=config['manifest_sha256'],
                        signature=signature, accepted_profiles=profiles, source_partitions=scanned)
    output.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for ds, name in zip(datasets, ('pressure_means.nc', 'neutral_density_means.nc')):
        temporary = output / (name + '.partial')
        ds.to_netcdf(temporary, engine='netcdf4', encoding={
            name: {"zlib": True, "complevel": 2, "_FillValue": None}
            for name in ds.data_vars
        })
        temporary.replace(output / name)
        output_hashes[name] = _sha256(output / name)
    report = {"status": "complete", "signature": signature, "config": config,
              "source_manifest": str(manifest_path), "output_directory": str(output),
              "files_scanned": scanned, "profiles_scanned": profiles,
              "invalid_position_profiles": invalid, "saved_Zfs_observations": samples,
              "valid_coordinate_observations": dict(zip(AXES, map(int, coordinate_counts))),
              "coordinate_unavailable_or_invalid_position": dict(zip(AXES, map(int, samples - coordinate_counts))),
              "conservation": conservation, "output_sha256": output_hashes,
              "elapsed_seconds": time.monotonic() - started}
    tmp = report_path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(report_path)
    print(f"PASS: full inventory and all observation counts conserved. Saved {output}", flush=True)
    return *datasets, report
