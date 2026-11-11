#!/usr/bin/env python3
"""Compare installed Argopy ingestion on six fixed local NATRE input files.

Run this script separately in each package environment. It does not modify
Argopy, download inputs, execute mixing calculations, or change validation gates.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from unittest.mock import patch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def string(value):
    return value.decode().strip() if isinstance(value, bytes) else str(value).strip()


def safe(value):
    import math
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def array_digest(values, np):
    values = np.asarray(values)
    finite = np.isfinite(values)
    return dict(shape=list(values.shape), dtype=str(values.dtype),
                finite_count=int(finite.sum()), nan_count=int(np.isnan(values).sum()),
                finite_float64_sha256=hashlib.sha256(values[finite].astype('<f8').tobytes()).hexdigest(),
                finite_mask_sha256=hashlib.sha256(finite.astype('u1').tobytes()).hexdigest())


def profile_rows(dataset, np):
    rows = []
    for i in range(dataset.sizes.get('N_PROF', 0)):
        profile = dataset.isel(N_PROF=i)
        arrays = {name: np.asarray(profile[name].values).reshape(-1) for name in ['PRES', 'TEMP', 'PSAL']}
        # Remove only common trailing padding. Preserve internal missing values
        # and the measurement order used by the production resolution filter.
        length = max((int(np.flatnonzero(np.isfinite(v))[-1]) + 1
                      for v in arrays.values() if np.isfinite(v).any()), default=0)
        arrays = {name: values[:length] for name, values in arrays.items()}
        pressure = arrays['PRES']
        delta = np.abs(np.diff(pressure))
        upper_delta = delta[pressure[:-1] < 1000]
        spacing = float(np.nanmean(upper_delta)) if np.isfinite(upper_delta).any() else None
        rows.append(dict(identity=[string(profile[name].values.item()) for name in ['PLATFORM_NUMBER', 'CYCLE_NUMBER', 'DIRECTION']],
                         time=np.datetime_as_string(profile.TIME.values, unit='ns'),
                         longitude=float(profile.LONGITUDE.values), latitude=float(profile.LATITUDE.values),
                         mean_pressure_spacing_upper_1000=spacing,
                         passes_legacy_spacing_filter=spacing is not None and spacing <= 5,
                         arrays=arrays,
                         pressure_first_8=pressure[:8].astype(float).tolist(),
                         pressure_last_8=pressure[-8:].astype(float).tolist()))
    return rows


def compare(actual, reference, np):
    result = dict(identity_equal=actual['identity'] == reference['identity'],
                  time_equal=actual['time'] == reference['time'],
                  time_delta_ns=int((np.datetime64(actual['time'], 'ns') - np.datetime64(reference['time'], 'ns')) / np.timedelta64(1, 'ns')),
                  longitude_equal=actual['longitude'] == reference['longitude'],
                  latitude_equal=actual['latitude'] == reference['latitude'], variables={})
    for name in ['PRES', 'TEMP', 'PSAL']:
        a, b = actual['arrays'][name], reference['arrays'][name]
        value = dict(actual_shape=list(a.shape), reference_shape=list(b.shape), shape_equal=a.shape == b.shape)
        if a.shape == b.shape:
            equal = (a == b) | (np.isnan(a) & np.isnan(b))
            finite = np.isfinite(a) & np.isfinite(b)
            value.update(equal=bool(equal.all()), mismatched_elements=int((~equal).sum()),
                         nan_mask_mismatches=int((np.isnan(a) != np.isnan(b)).sum()),
                         max_abs_error=float(np.max(np.abs(a[finite].astype(float) - b[finite].astype(float)))) if finite.any() else None)
        else:
            value['equal'] = False
        result['variables'][name] = value
    result['all_input_values_equal'] = all(result[key] for key in ['identity_equal', 'time_equal', 'longitude_equal', 'latitude_equal']) and all(v['equal'] for v in result['variables'].values())
    return result


def summarize_profile(row, np):
    return {**{name: value for name, value in row.items() if name != 'arrays'},
            'array_digests': {name: array_digest(value, np) for name, value in row['arrays'].items()}}


def probe(args):
    import numpy as np
    import xarray as xr
    cache = Path('/tmp/argo-mixing-runtime/argopy')
    cache.mkdir(parents=True, exist_ok=True)
    expanduser = os.path.expanduser

    def import_cache_path(path):
        return str(cache) if path == '~/.cache/argopy' else expanduser(path)

    # Published Argopy 1.4 creates this hardcoded cache during import. Redirect
    # only that exact cache path, without changing HOME or release source code.
    with patch('os.path.expanduser', side_effect=import_cache_path):
        import argopy
        from argopy.data_fetchers import gdac_data_processors
    argopy.set_options(cachedir=str(cache))

    package_root = Path(argopy.__file__).parent
    sources = [package_root / 'data_fetchers/gdac_data_processors.py', package_root / 'xarray.py']
    dependencies = {}
    for package in ['argopy', 'numpy', 'pandas', 'xarray', 'netCDF4', 'scipy', 'erddapy']:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, executable=sys.executable, platform=platform.platform(),
                  argopy_import_version=argopy.__version__, argopy_import_path=str(Path(argopy.__file__).resolve()),
                  dependencies=dependencies, source_sha256={str(p): sha256(p) for p in sources if p.is_file()},
                  probe_sha256=sha256(__file__),
                  import_cache_redirect=dict(original='~/.cache/argopy', actual=str(cache),
                                             method='Narrow os.path.expanduser patch during import only; release package source and HOME unchanged.'),
                  procedure='Production GDAC preprocessor, research QC, NATRE bounds/time/pressure mask, production variable subset, point2profile. Drop unused SCIENTIFIC_CALIB_DATE before Argopy casting, matching production loader.',
                  reference_padding='Remove common trailing missing-value padding only; preserve internal missing values and measurement order.',
                  cases=[])
    samples = [('1901694', '210', '2019', 'aoml/1901694/profiles/D1901694_210.nc', 'D1901694_210.nc'),
               ('1901730', '259', '2021', 'aoml/1901730/profiles/R1901730_259.nc', 'D1901730_259.nc'),
               ('6901151', '107', '2014', 'bodc/6901151/profiles/D6901151_107.nc', 'D6901151_107.nc')]
    for wmo, cycle, year, old_path, new_path in samples:
        reference_path = args.reference / f'NATRE_{year}_{int(year)+1}_Argo_resmode.nc'
        with xr.open_dataset(reference_path, engine='netcdf4') as opened:
            mask = ((opened.PLATFORM_NUMBER.values.astype(str) == wmo) &
                    (opened.CYCLE_NUMBER.values.astype(str) == cycle) &
                    (opened.DIRECTION.values.astype(str) == 'A'))
            reference = profile_rows(opened.isel(N_PROF=np.flatnonzero(mask)).load(), np)
        if len(reference) != 1:
            raise ValueError(f'Expected exactly one saved raw reference for {wmo}/{cycle}, found {len(reference)}')
        for archive, path in [('local_2023', args.gdac / 'dac' / old_path), ('current_sample', args.current_dir / new_path)]:
            case = dict(identity=[wmo, cycle, 'A'], archive=archive, input_path=str(path),
                        reference_path=str(reference_path), reference_sha256=sha256(reference_path),
                        reference=summarize_profile(reference[0], np))
            try:
                case['input_sha256'] = sha256(path)
                with xr.open_dataset(path, engine='netcdf4') as opened:
                    raw = opened.drop_vars('SCIENTIFIC_CALIB_DATE', errors='ignore').load()
                    case['raw_sizes'] = dict(raw.sizes)
                    case['raw_data_modes'] = [string(v) for v in raw.DATA_MODE.values]
                    case['raw_date_update'] = string(raw.DATE_UPDATE.values.item())
                    case['raw_sampling_scheme'] = [string(v) for v in raw.VERTICAL_SAMPLING_SCHEME.values]
                points = gdac_data_processors.pre_process_multiprof(raw, access_point='BOX', access_point_opts={})
                case['preprocessed_points'] = points.sizes.get('N_POINTS', 0)
                points = points.argo.filter_researchmode()
                case['research_qc_points'] = points.sizes.get('N_POINTS', 0)
                if not points.sizes.get('N_POINTS', 0):
                    case.update(status='research_qc_rejected', profiles=[], comparisons=[])
                else:
                    mask = ((points.LONGITUDE >= -38) & (points.LONGITUDE < -20) &
                            (points.LATITUDE >= 19) & (points.LATITUDE < 32) &
                            (points.PRES >= 0) & (points.PRES < 2200) &
                            (points.TIME >= np.datetime64('2014-01-01')) &
                            (points.TIME < np.datetime64('2024-01-01')))
                    points = points.where(mask, drop=True)
                    needed = ['PRES', 'TEMP', 'PSAL', 'TIME', 'LATITUDE', 'LONGITUDE',
                              'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'DIRECTION']
                    profiles = points[[name for name in needed if name in points]].argo.point2profile()
                    for name in ['PRES', 'TEMP', 'PSAL']:
                        if 'N_LEVELS' not in profiles[name].dims:
                            profiles[name] = xr.broadcast(profiles[name], profiles.N_LEVELS)[0].transpose('N_PROF', 'N_LEVELS')
                    rows = profile_rows(profiles, np)
                    case.update(status='loaded', profiles=[summarize_profile(row, np) for row in rows],
                                comparisons=[compare(row, reference[0], np) for row in rows])
            except Exception:
                case.update(status='error', traceback=traceback.format_exc())
            report['cases'].append(case)
    report['summary'] = dict(cases=len(report['cases']), errors=sum(c['status'] == 'error' for c in report['cases']),
                             research_qc_rejections=sum(c['status'] == 'research_qc_rejected' for c in report['cases']),
                             exact_reference_profiles=sum(v['all_input_values_equal'] for c in report['cases'] for v in c.get('comparisons', [])))
    return safe(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--argopy-dir', type=Path, help='Parent of an extracted published argopy package; inserted first on sys.path.')
    parser.add_argument('--gdac', type=Path, default=Path('/swot/SUM05/dbalwada/Argo_sync'))
    parser.add_argument('--reference', type=Path, default=Path('/swot/SUM05/amf2288/espe_code/data_files'))
    parser.add_argument('--current-dir', type=Path, default=Path('/swot/SUM05/amf2288/argo_mixing/natre_investigation_20261111/current_gdac_samples'))
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path.home()):
        parser.error('Reports must be outside the home directory.')
    if args.output.exists():
        parser.error('Use a fresh output path; existing reports are not overwritten.')
    for key, suffix in [('MPLCONFIGDIR', 'matplotlib'), ('NUMBA_CACHE_DIR', 'numba'), ('XDG_CACHE_HOME', 'cache')]:
        os.environ.setdefault(key, '/tmp/argo-mixing-runtime/' + suffix)
    proj_data = Path('/home/amf2288/miniconda3/envs/Argo_Feb_25/share/proj')
    if proj_data.is_dir():
        os.environ.setdefault('PROJ_DATA', str(proj_data))
    if args.argopy_dir:
        sys.path.insert(0, str(args.argopy_dir.resolve()))
    report = probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(argopy_import_version=report['argopy_import_version'], argopy_import_path=report['argopy_import_path'], **report['summary']), indent=2))
    return 1 if report['summary']['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
