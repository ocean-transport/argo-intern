#!/usr/bin/env python3
"""Bounded, read-only ingestion experiments; never changes the validation gate.

Compare shared float/cycle/direction profiles from the completed NATRE run with
the saved research-mode ERDDAP inputs. Removing the installed GDAC pressure
rounding is an isolated diagnostic, not a production preprocessing change.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
import process_argo_mixing as mixing
import numpy as np
import xarray as xr
os.environ.setdefault('PROJ_DATA', str(Path(sys.base_prefix) / 'share/proj'))
_argopy_cache = Path('/tmp/argo-mixing-runtime/argopy')
_argopy_cache.mkdir(parents=True, exist_ok=True)
_expanduser = os.path.expanduser
# Published 1.4.0 opens a cached static table during import, before set_options
# is available. Redirect only that cache path; leave package code and HOME intact.
with patch('os.path.expanduser', side_effect=lambda path: str(_argopy_cache)
           if path == '~/.cache/argopy' else _expanduser(path)):
    import argopy
argopy.set_options(cachedir=str(_argopy_cache))
from argopy.data_fetchers import gdac_data_processors as gdac


def key(ds, i, names):
    return tuple(str(ds[n].values[i]).strip() for n in names)


def compare_values(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = {'actual_levels': len(a), 'reference_levels': len(b),
           'equal': bool(np.array_equal(a, b))}
    if a.shape == b.shape:
        delta = np.abs(a.astype('float64') - b.astype('float64'))
        out.update(mismatched=int(np.count_nonzero(a != b)),
                   max_abs_difference=float(delta.max()) if delta.size else 0.)
    return out


def unrounded_preprocessor():
    source = inspect.getsource(gdac.pre_process_multiprof)
    tree = ast.parse(source)
    function = tree.body[0]
    loops = [n for n in function.body if isinstance(n, ast.For)
             and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                     and c.func.attr == 'round' for c in ast.walk(n))]
    if len(loops) != 1:
        raise ValueError('Installed rounding implementation changed; inspect it first')
    function.body.remove(loops[0])
    namespace = dict(vars(gdac))
    exec(compile(ast.fix_missing_locations(tree), '<diagnostic_no_round>', 'exec'), namespace)
    return namespace['pre_process_multiprof'], hashlib.sha256(source.encode()).hexdigest()


def run(args):
    config = json.loads((args.run / 'local/manifest.json').read_text())['config']
    refs = defaultdict(list)
    raw_names = ['PLATFORM_NUMBER', 'CYCLE_NUMBER', 'DIRECTION']
    out_names = ['platform_number', 'cycle_number', 'direction']
    for path in sorted(args.reference.glob('NATRE_????_????_Argo_resmode.nc')):
        with xr.open_dataset(path) as ds:
            meta = ds[raw_names].load()
            for i in range(ds.sizes['N_PROF']):
                refs[key(meta, i, raw_names)].append((path, i))
    candidates = defaultdict(dict)
    for path in sorted((args.run / 'local/profiles/region').glob('*.nc')):
        with xr.open_dataset(path) as ds:
            meta = ds[out_names + ['source_file']].load()
            for i in range(ds.sizes['time']):
                identity = key(meta, i, out_names)
                if len(refs[identity]) != 1:
                    continue
                year = str(meta.time.values[i])[:4]
                candidates[year].setdefault(identity[0], (identity, str(meta.source_file.values[i])))
    # Spread the small sample over both years and floats; preserve reproducibility.
    selected = []
    for year, floats in sorted(candidates.items()):
        pool = sorted(floats.values())
        indices = np.linspace(0, len(pool) - 1, min(args.per_year, len(pool)), dtype=int)
        selected.extend((year, *pool[i]) for i in indices)
    no_round, source_hash = unrounded_preprocessor()
    records = []
    for year, identity, relative in selected:
        ref_path, index = refs[identity][0]
        with xr.open_dataset(ref_path) as ds:
            ref = ds.isel(N_PROF=index).load()
        local_path = Path(config['gdac']) / 'dac' / relative
        with xr.open_dataset(local_path, engine='netcdf4') as ds:
            raw = ds.drop_vars('SCIENTIFIC_CALIB_DATE', errors='ignore').load()
        record = {'identity': identity, 'year': year, 'local_path': str(local_path),
                  'reference_path': str(ref_path), 'reference_index': index,
                  'local_date_update': str(raw.DATE_UPDATE.values) if 'DATE_UPDATE' in raw else None,
                  'local_data_mode': raw.DATA_MODE.values.astype(str).tolist(), 'variants': {}}
        for label, processor in [('installed_gdac', gdac.pre_process_multiprof), ('without_pressure_rounding', no_round)]:
            points = processor(raw.copy(deep=True), access_point='BOX', access_point_opts={})
            points = points.argo.filter_researchmode()
            points = points.where((points.PRES >= 0) & (points.PRES < config['max_pressure']), drop=True)
            mask = np.ones(points.sizes['N_POINTS'], dtype=bool)
            for name, value in zip(raw_names, identity):
                mask &= (points[name].values == float(value) if name != 'DIRECTION'
                         else np.char.strip(points[name].values.astype(str)) == value)
            points = points.isel(N_POINTS=np.flatnonzero(mask))
            if not points.sizes['N_POINTS']:
                raise ValueError(f'Previously accepted identity unexpectedly empty: {identity}')
            # Reconstruct the same profile consumed by production, including order.
            points = points.argo.point2profile()
            if points.sizes['N_PROF'] != 1:
                raise ValueError(f'Ambiguous identity after conversion: {identity}')
            points = points.isel(N_PROF=0)
            result = {name: compare_values(points[name].values, ref[name].values)
                      for name in ['PRES', 'TEMP', 'PSAL']}
            for name in ['PRES', 'TEMP', 'PSAL']:
                result[name]['local_first_five'] = points[name].values[:5].astype(float).tolist()
                result[name]['reference_first_five'] = ref[name].values[:5].astype(float).tolist()
            result['time_offset_ns'] = int((points.TIME.values - ref.TIME.values).astype('timedelta64[ns]').astype('int64'))
            result['longitude_offset'] = float(points.LONGITUDE.values - ref.LONGITUDE.values)
            result['latitude_offset'] = float(points.LATITUDE.values - ref.LATITUDE.values)
            record['variants'][label] = result
        records.append(record)
    summary = {}
    for label in ['installed_gdac', 'without_pressure_rounding']:
        summary[label] = {name + '_equal_profiles': sum(r['variants'][label][name]['equal'] for r in records)
                          for name in ['PRES', 'TEMP', 'PSAL']}
        summary[label]['all_three_equal_profiles'] = sum(all(r['variants'][label][v]['equal'] for v in ['PRES', 'TEMP', 'PSAL']) for r in records)
    report = {'purpose': 'Diagnostic only; does not relax or pass historical validation',
              'argopy_version': argopy.__version__, 'argopy_path': argopy.__file__,
              'xarray_version': xr.__version__, 'numpy_version': np.__version__,
              'sample_design': 'One shared accepted profile per selected float; up to per_year evenly spaced float IDs each year; unique reference identities only',
              'sample_profiles': len(records), 'per_year': args.per_year,
              'pressure_rounding_source_sha256': source_hash,
              'diagnostic_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'comparison_note': 'Finite values after production point2profile conversion; differing level counts are reported, not interpolated. Sample does not establish full population equivalence.',
              'summary': summary, 'profiles': records}
    output = mixing._data_directory(args.output)
    mixing._json_write(output / 'ingestion_experiments.json', report)
    print(json.dumps({'sample_profiles': len(records), 'summary': summary, 'report': str(output / 'ingestion_experiments.json')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('/swot/SUM05/amf2288/argo_mixing/natre_validation_20261110'))
    parser.add_argument('--reference', type=Path, default=Path('/swot/SUM05/amf2288/espe_code/data_files'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-year', type=int, default=4)
    run(parser.parse_args())
