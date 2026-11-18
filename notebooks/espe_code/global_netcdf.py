"""Build the two standalone NetCDF products used by global_check.ipynb.

Profile consolidation streams bounded batches, preserving all source variables.
The analysis half of the notebook never calls these builders or opens partitions.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from xarray.coding.times import decode_cf_datetime


def _digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _json_write(path, value):
    tmp = Path(str(path) + '.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def _json_numpy(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Cannot encode {type(value)}')


def _schema(path):
    with Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        dimensions = {k: len(v) for k, v in ds.dimensions.items()}
        variables = {}
        for name, v in ds.variables.items():
            variables[name] = {
                'dtype': 'str' if v.dtype is str else np.dtype(v.dtype).str,
                'dims': v.dimensions, 'attrs': {a: v.getncattr(a) for a in v.ncattrs()},
            }
        static = {k: np.asarray(v[:]) for k, v in ds.variables.items() if 'time' not in v.dimensions}
        attrs = {a: ds.getncattr(a) for a in ('Description', 'scientific_mode', 'scientific_parameters',
                                            'legacy_conventions', 'failure_handling', 'source_mode')
                 if a in ds.ncattrs()}
    return dimensions, variables, static, attrs


def _read_profile_group(task):
    paths, schema, static = task
    pieces = {name: [] for name, spec in schema.items() if 'time' in spec['dims']}
    lengths = []
    for path in paths:
        with Dataset(path) as ds:
            ds.set_auto_maskandscale(False)
            if set(ds.variables) != set(schema):
                raise ValueError(f'Variable inventory differs: {path}')
            lengths.append(len(ds.dimensions['time']))
            for name, spec in schema.items():
                v = ds[name]
                dtype = 'str' if v.dtype is str else np.dtype(v.dtype).str
                if v.dimensions != spec['dims'] or dtype != spec['dtype']:
                    raise ValueError(f'Source dimensions/dtype differ: {path}: {name}')
                v.set_var_chunk_cache(2 * 1024**2, 1009, .75)
                values = np.asarray(v[:])
                if name in static:
                    if values.shape != static[name].shape or values.tobytes() != static[name].tobytes():
                        raise ValueError(f'Source coordinate differs: {path}: {name}')
                    continue
                if name == 'time':
                    values = np.asarray(decode_cf_datetime(
                        values, v.units, getattr(v, 'calendar', 'standard')),
                        dtype='datetime64[ns]').view(np.int64)
                    if np.any(values == np.iinfo(np.int64).min):
                        raise ValueError(f'Missing profile time: {path}')
                pieces[name].append(values)
    arrays = {name: np.concatenate(parts, axis=schema[name]['dims'].index('time'))
              for name, parts in pieces.items()}
    return arrays, lengths


def _create_profile_file(path, nprofiles, dimensions, schema, static, attrs, signature, manifest_path):
    ds = Dataset(path, 'w', format='NETCDF4')
    ds.set_fill_off()
    for name, length in dimensions.items():
        ds.createDimension(name, nprofiles if name == 'time' else length)
    for name, spec in schema.items():
        dtype = str if spec['dtype'] == 'str' else np.dtype(spec['dtype'])
        kwargs = {}
        if 'time' in spec['dims']:
            kwargs['chunksizes'] = tuple(min(nprofiles, 8192 if len(spec['dims']) == 1 else 64)
                                         if dim == 'time' else dimensions[dim] for dim in spec['dims'])
        if '_FillValue' in spec['attrs']:
            kwargs['fill_value'] = spec['attrs']['_FillValue']
        v = ds.createVariable(name, dtype, spec['dims'], **kwargs)
        v.setncatts({k: value for k, value in spec['attrs'].items() if k != '_FillValue'})
        v.set_auto_maskandscale(False)
        v.set_var_chunk_cache(2 * 1024**2, 1009, .75)
        if name in static:
            v[:] = static[name]
    # Per-partition time origins differ. Use exact integer nanoseconds throughout.
    ds['time'].setncatts({'units': 'nanoseconds since 1970-01-01 00:00:00',
                         'calendar': 'proleptic_gregorian'})
    ds['latitude'].setncattr('Description', 'Latitude')
    ds['gamma_n_status'].setncatts({
        'flag_values': np.arange(6, dtype=np.int16),
        'flag_meanings': 'computed unsupported_latitude invalid_position no_samples failed_samples arithmetic_failure',
    })
    ds.setncatts(attrs)
    ds.setncatts({
        'source_manifest': str(manifest_path), 'accepted_profiles': nprofiles,
        'consolidation_signature': signature, 'processing_stage': 'profiles',
        'profile_order': 'sorted manifest regions, sorted partition paths, original within-file order',
        'storage': 'uncompressed NetCDF4 chunks; original numeric precision and all profile variables retained',
        'time_encoding': 'int64 nanoseconds since Unix epoch; decoded timestamps preserved exactly',
    })
    return ds


def consolidate_profiles(manifest_path, output_path, workers=4, files_per_task=8):
    """Create/reuse one global profile file, verifying each written variable block.

    Bounded prefetch limits memory; one process writes the uncompressed output.
    No raw Argo science is recomputed. Successful products are reused by their
    manifest/code signature, file size and completion report, without rereading
    the entire 140-GiB file on each notebook execution.
    """
    manifest_path, output_path = Path(manifest_path).resolve(), Path(output_path).resolve()
    if not output_path.is_relative_to(Path('/swot/SUM05/amf2288')):
        raise ValueError('Save consolidated data under /swot/SUM05/amf2288')
    if workers < 1 or files_per_task < 1:
        raise ValueError('workers/files_per_task must be positive')
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest['status'] != 'complete' or manifest['failed_batches']:
        raise ValueError('The global manifest is incomplete')
    paths = [p for r in sorted(manifest['regions']) for p in sorted(manifest['regions'][r])]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Empty or duplicate source file inventory')
    config = {'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'helper_sha256': _digest(__file__),
              'files_per_task': files_per_task, 'encoding': 'lossless_uncompressed_time_ns_v1'}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    report_path = output_path.with_suffix('.json')
    if output_path.exists():
        if not report_path.exists():
            raise FileExistsError(f'Existing file has no completion report: {output_path}')
        report = json.loads(report_path.read_text())
        with Dataset(output_path) as ds:
            valid = (report['status'] == 'complete' and report['signature'] == signature
                     and ds.consolidation_signature == signature
                     and len(ds.dimensions['time']) == manifest['accepted_profiles']
                     and output_path.stat().st_size == report['file_bytes'])
        if not valid:
            raise FileExistsError(f'Existing product does not match this build; use a new output path: {output_path}')
        print(f'Reusing complete profile NetCDF: {output_path}', flush=True)
        return report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix('.nc.partial')
    if temporary.exists():
        raise FileExistsError(f'Incomplete export already exists; inspect before retrying: {temporary}')
    dimensions, schema, static, attrs = _schema(paths[0])
    groups = [paths[i:i + files_per_task] for i in range(0, len(paths), files_per_task)]
    tasks = iter((group, schema, static) for group in groups)
    first_samples, last_samples = None, None
    repaired_samples = []
    offset = nfiles = verified_blocks = 0
    started = time.monotonic()
    print(f'Consolidating {len(paths):,} files / {manifest["accepted_profiles"]:,} profiles into {output_path}', flush=True)
    print('Every numeric block is compared byte-for-byte after writing; string values are checked exactly.', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        for _ in range(workers * 2):
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(_read_profile_group, task))
        with _create_profile_file(temporary, manifest['accepted_profiles'], dimensions, schema, static,
                                  attrs, signature, manifest_path) as target:
            for group in groups:
                values, lengths = pending.popleft().result()
                task = next(tasks, None)
                if task is not None:
                    pending.append(pool.submit(_read_profile_group, task))
                n = sum(lengths)
                if offset + n > manifest['accepted_profiles']:
                    raise AssertionError('More input profiles than the manifest total')
                for name, array in values.items():
                    spec = schema[name]
                    slices = tuple(slice(offset, offset + n) if dim == 'time' else slice(None)
                                   for dim in spec['dims'])
                    target[name][slices] = array
                    reread = np.asarray(target[name][slices])
                    if spec['dtype'] == 'str':
                        equal = np.array_equal(array, reread)
                    else:
                        equal = array.dtype == reread.dtype and array.tobytes() == reread.tobytes()
                    if not equal:
                        raise AssertionError(f'Write/read mismatch: {name}, profiles {offset}:{offset+n}')
                    verified_blocks += 1
                local_offset = offset
                for path, length in zip(group, lengths):
                    sample = {'path': path, 'start': local_offset, 'stop': local_offset + length}
                    if first_samples is None:
                        first_samples = sample
                    last_samples = sample
                    if '/repaired_batch/' in path:
                        repaired_samples.append(sample)
                    local_offset += length
                offset += n
                nfiles += len(group)
                if nfiles % 512 == 0 or nfiles == len(paths):
                    elapsed = time.monotonic() - started
                    eta = elapsed * (manifest['accepted_profiles'] / offset - 1)
                    print(f'  {nfiles:,}/{len(paths):,} files; {offset:,} profiles; '
                          f'{elapsed:.1f}s elapsed; ~{eta/60:.1f} min remaining', flush=True)
            if offset != manifest['accepted_profiles'] or nfiles != len(paths):
                raise AssertionError('Consolidated coverage differs from complete manifest')
            target.sync()
    # Reopen the closed on-disk file and check complete boundary/repaired partitions.
    reopen_samples = [first_samples, last_samples, *repaired_samples]
    with Dataset(temporary) as target:
        target.set_auto_maskandscale(False)
        for sample in reopen_samples:
            values, _ = _read_profile_group(([sample['path']], schema, static))
            for name, expected in values.items():
                slices = tuple(slice(sample['start'], sample['stop']) if dim == 'time' else slice(None)
                               for dim in schema[name]['dims'])
                actual = np.asarray(target[name][slices])
                if schema[name]['dtype'] == 'str':
                    assert np.array_equal(expected, actual), name
                else:
                    assert expected.dtype == actual.dtype and expected.tobytes() == actual.tobytes(), name
        for name, expected in static.items():
            assert np.asarray(target[name][:]).tobytes() == expected.tobytes(), name
    report = {
        'status': 'complete', 'signature': signature, 'config': config,
        'source_manifest': str(manifest_path), 'output_file': str(output_path),
        'source_files': nfiles, 'profiles': offset, 'variables_including_coordinates': len(schema),
        'verified_variable_blocks': verified_blocks,
        'numeric_verification': 'same dtype and bytes for every time-dependent block immediately after writing; static coordinates checked in every source',
        'string_verification': 'exact value equality for every time-dependent block',
        'closed_file_verification': reopen_samples,
        'file_bytes': temporary.stat().st_size, 'elapsed_seconds': time.monotonic() - started,
    }
    temporary.replace(output_path)
    _json_write(report_path, report)
    print(f'PASS: {offset:,} profiles saved and verified; {report["file_bytes"]/1024**3:.2f} GiB; '
          f'{report["elapsed_seconds"]/60:.1f} minutes.', flush=True)
    return report


def merge_binned_grids(pressure, density):
    """Two independent vertical grids in one Dataset, with unambiguous names."""
    for name in ('latitude', 'longitude', 'latitude_bounds', 'longitude_bounds', 'cell_profile_count'):
        a, b = pressure[name], density[name]
        if a.dims != b.dims or a.dtype != b.dtype or a.values.tobytes() != b.values.tobytes():
            raise ValueError(f'Shared binned coordinate/count differs: {name}')
    pnames = {k: k + '_pressure' for k, v in pressure.data_vars.items()
              if 'pressure_bin' in v.dims and k != 'pressure_bin_bounds'}
    dnames = {k: k + '_density' for k, v in density.data_vars.items()
              if 'neutral_density_bin' in v.dims and k != 'neutral_density_bin_bounds'}
    merged = xr.merge([pressure.rename(pnames), density.rename(dnames)],
                      compat='identical', join='exact', combine_attrs='override')
    # combine_attrs also applies to variables; replace only the Dataset attrs.
    merged.attrs = dict(
        description='Global 3-degree observation-weighted means on separate pressure and neutral-density bins',
        pressure_variable_suffix='_pressure', density_variable_suffix='_density',
        pressure_grid_attrs_json=json.dumps(dict(pressure.attrs), default=_json_numpy),
        density_grid_attrs_json=json.dumps(dict(density.attrs), default=_json_numpy),
    )
    return merged


def split_binned_grids(combined):
    """Views of the two grids; no new computation or disk read until requested."""
    pressure = combined.drop_dims('neutral_density_bin')
    density = combined.drop_dims('pressure_bin')
    pressure = pressure.rename({k: k.removesuffix('_pressure') for k in pressure.data_vars if k.endswith('_pressure')})
    density = density.rename({k: k.removesuffix('_density') for k in density.data_vars if k.endswith('_density')})
    pressure.attrs = json.loads(combined.attrs['pressure_grid_attrs_json'])
    density.attrs = json.loads(combined.attrs['density_grid_attrs_json'])
    return pressure, density


def save_binned_grids(pressure, density, output_path, binning_report):
    """Write/reuse the single binned Dataset, verifying both original grids."""
    output_path = Path(output_path).resolve()
    if not output_path.is_relative_to(Path('/swot/SUM05/amf2288')):
        raise ValueError('Save consolidated data under /swot/SUM05/amf2288')
    combined = merge_binned_grids(pressure, density)
    combined.attrs.update(binning_report_json=json.dumps(binning_report),
                          source_manifest=binning_report['source_manifest'],
                          binning_signature=binning_report['signature'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with xr.open_dataset(output_path, engine='netcdf4') as previous:
            if previous.attrs.get('binning_signature') != binning_report['signature']:
                raise FileExistsError(f'Existing binned file has different settings: {output_path}')
            for actual, expected in zip(split_binned_grids(previous), (pressure, density)):
                _check_arrays(actual, expected)
        print(f'Reusing verified binned NetCDF: {output_path}', flush=True)
        return {'status': 'complete', 'output_file': str(output_path), 'file_bytes': output_path.stat().st_size}
    temporary = output_path.with_suffix('.nc.partial')
    if temporary.exists():
        raise FileExistsError(f'Incomplete binned export already exists: {temporary}')
    combined.to_netcdf(temporary, engine='netcdf4', encoding={
        k: {'zlib': True, 'complevel': 2, '_FillValue': None} for k in combined.data_vars})
    with xr.open_dataset(temporary, engine='netcdf4') as reread:
        for actual, expected in zip(split_binned_grids(reread), (pressure, density)):
            _check_arrays(actual, expected)
    temporary.replace(output_path)
    report = {'status': 'complete', 'output_file': str(output_path),
              'file_bytes': output_path.stat().st_size, 'sha256': _digest(output_path),
              'source_binning_signature': binning_report['signature'],
              'verification': 'all variables, coordinates, dimensions, dtypes, bytes and variable attributes match both source grids'}
    _json_write(output_path.with_suffix('.json'), report)
    print(f'PASS: both binned grids saved bitwise in {output_path}', flush=True)
    return report


def _check_arrays(actual, expected):
    if set(actual.variables) != set(expected.variables):
        raise AssertionError('Variable inventory differs')
    for name in expected.variables:
        a, b = actual[name], expected[name]
        if a.dims != b.dims or a.shape != b.shape or a.dtype != b.dtype or a.values.tobytes() != b.values.tobytes():
            raise AssertionError(f'Binned roundtrip differs: {name}')
        # Compare metadata independently, without another read of the arrays.
        try:
            xr.testing.assert_identical(xr.Dataset(attrs=a.attrs), xr.Dataset(attrs=b.attrs))
        except AssertionError as exc:
            raise AssertionError(f'Binned variable attributes differ: {name}') from exc
