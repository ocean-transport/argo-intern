#!/usr/bin/env python3
"""Read-only NATRE population audit, keyed by float/cycle/direction.

This diagnoses coverage without changing the strict numerical validation gate.
Write JSON reports only to an explicit output outside the home directory.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re

os.environ.setdefault('MPLCONFIGDIR', '/tmp/argo-mixing-runtime/matplotlib')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/argo-mixing-runtime/numba')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/argo-mixing-runtime/cache')

import numpy as np
import pandas as pd
import xarray as xr


def text(value):
    if isinstance(value, bytes):
        return value.decode().strip()
    return str(value).strip()


def key_file(path):
    match = re.fullmatch(r'[RD](\d+)_(\d+)(D?)\.nc', Path(path).name)
    if match is None:
        raise ValueError(f'Unrecognized profile filename: {path}')
    return (match[1], str(int(match[2])), 'D' if match[3] else 'A')


def records(paths, raw=False):
    rows = []
    names = ['PLATFORM_NUMBER', 'CYCLE_NUMBER', 'DIRECTION', 'TIME', 'LONGITUDE', 'LATITUDE'] if raw else [
        'platform_number', 'cycle_number', 'direction', 'time', 'longitude', 'latitude']
    for path in sorted(paths):
        with xr.open_dataset(path, engine='netcdf4') as ds:
            vals = [ds[name].values for name in names]
            for i in range(len(vals[0])):
                identity = tuple(text(v[i]) for v in vals[:3])
                rows.append(dict(key=identity, time=np.datetime_as_string(vals[3][i], unit='ns'),
                                 longitude=float(vals[4][i]), latitude=float(vals[5][i]),
                                 partition=str(path), index=i))
    return rows


def grouped(rows):
    result = defaultdict(list)
    for row in rows:
        result[row['key']].append(row)
    return result


def year_counts(rows):
    return dict(sorted(Counter(r['time'][:4] for r in rows).items()))


def scalar_summary(values):
    values = np.asarray(values)
    return dict(count=len(values), nonzero=int(np.count_nonzero(values)),
                min=float(values.min()), max=float(values.max()),
                median_abs=float(np.median(np.abs(values))),
                max_abs=float(np.max(np.abs(values))))


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def audit(run, gdac, reference):
    local = records((run / 'local/profiles/region').glob('*.nc'))
    historical = records((run / 'legacy_raw/profiles/region').glob('*.nc'))
    raw = records(reference.glob('NATRE_????_????_Argo_resmode.nc'), raw=True)
    lg, hg, rg = grouped(local), grouped(historical), grouped(raw)
    lk, hk, rk = set(lg), set(hg), set(rg)
    shared, only_ref, only_local = lk & hk, hk - lk, lk - hk

    selected_index, full_index_for_relevant = {}, {}
    relevant = hk | lk
    with (gdac / 'ar_index_global_prof.txt').open() as stream:
        headers = [next(stream).strip() for _ in range(8)]
    for frame in pd.read_csv(gdac / 'ar_index_global_prof.txt', comment='#', dtype=str, chunksize=100_000):
        for row in frame.itertuples(index=False):
            key = key_file(row.file)
            if key not in relevant:
                continue
            record = dict(path=row.file, date=row.date, longitude=float(row.longitude), latitude=float(row.latitude))
            full_index_for_relevant[key] = record
            if ('20140101000000' <= row.date < '20240101000000' and
                    -38 <= record['longitude'] < -20 and 19 <= record['latitude'] < 32):
                selected_index[key] = record

    rejects = {}
    rejection_counts = Counter()
    for status_path in sorted((run / 'local/profiles/region').glob('*.json')):
        status = json.loads(status_path.read_text())
        for item in status.get('accounting', {}).get('rejections', []):
            rejects[key_file(item['source_file'])] = item['reason']
            rejection_counts[item['reason']] += 1

    classified_ref = []
    for key in sorted(only_ref):
        if key not in full_index_for_relevant:
            reason = 'absent_from_local_index'
        elif key not in selected_index:
            reason = 'local_index_outside_selection'
        elif key in rejects:
            reason = 'loader_' + rejects[key]
        else:
            reason = 'selected_but_not_accepted_by_profile_kernel'
        classified_ref.append(dict(identity=key, reason=reason, reference=hg[key],
                                   local_index=full_index_for_relevant.get(key)))

    classified_local = [dict(identity=key,
                             reason='absent_from_saved_research_raw' if key not in rk else 'saved_research_raw_not_accepted_by_profile_kernel',
                             local=lg[key], reference_raw=rg.get(key, [])) for key in sorted(only_local)]
    unique_shared = sorted(key for key in shared if len(lg[key]) == len(hg[key]) == 1)
    deltas = []
    for key in unique_shared:
        l, h = lg[key][0], hg[key][0]
        dt_ns = int((np.datetime64(l['time'], 'ns') - np.datetime64(h['time'], 'ns')) / np.timedelta64(1, 'ns'))
        deltas.append(dict(identity=key, time_delta_ns=dt_ns,
                           longitude_delta=l['longitude'] - h['longitude'],
                           latitude_delta=l['latitude'] - h['latitude'], local=l, reference=h))
    coordinates_exact = [r for r in deltas if r['longitude_delta'] == r['latitude_delta'] == 0]
    years = sorted(set(year_counts(local)) | set(year_counts(historical)))
    by_year = {}
    for year in years:
        by_year[year] = dict(local_accepted=sum(r['time'].startswith(year) for r in local),
                             historical_accepted=sum(r['time'].startswith(year) for r in historical),
                             historical_only_unique=sum(hg[k][0]['time'].startswith(year) for k in only_ref),
                             local_only_unique=sum(lg[k][0]['time'].startswith(year) for k in only_local),
                             shared_unique=sum(hg[k][0]['time'].startswith(year) for k in shared))
    summary = dict(local_profiles=len(local), historical_profiles=len(historical),
                   local_unique_identities=len(lk), historical_unique_identities=len(hk),
                   shared_unique_identities=len(shared), historical_only_unique_identities=len(only_ref),
                   local_only_unique_identities=len(only_local),
                   shared_occurrences=sum(min(len(lg[k]), len(hg[k])) for k in shared),
                   local_duplicate_identities={':'.join(k):len(v) for k,v in lg.items() if len(v)>1},
                   historical_duplicate_identities={':'.join(k):len(v) for k,v in hg.items() if len(v)>1},
                   historical_only_reasons=dict(Counter(r['reason'] for r in classified_ref)),
                   local_only_reasons=dict(Counter(r['reason'] for r in classified_local)),
                   local_loader_rejection_counts=dict(rejection_counts), by_year=by_year,
                   unique_shared_pairs=len(deltas),
                   unique_shared_exact_coordinates=len(coordinates_exact),
                   unique_shared_exact_time_coordinates=sum(r['time_delta_ns']==0 for r in coordinates_exact),
                   time_delta_ns=scalar_summary([r['time_delta_ns'] for r in deltas]),
                   longitude_delta=scalar_summary([r['longitude_delta'] for r in deltas]),
                   latitude_delta=scalar_summary([r['latitude_delta'] for r in deltas]),
                   unique_shared_time_delta_abs_at_most_1ms=sum(abs(r['time_delta_ns'])<=1e6 for r in deltas),
                   unique_shared_time_delta_abs_above_1s=sum(abs(r['time_delta_ns'])>1e9 for r in deltas),
                   reference_only_after_local_index_cutoff=sum(hg[k][0]['time']>'2023-08-04T09:50:20' for k in only_ref))
    return dict(summary=summary, provenance=dict(run=str(run), gdac=str(gdac), reference=str(reference), local_index_header=headers),
                identity='(PLATFORM_NUMBER, CYCLE_NUMBER, DIRECTION); duplicate occurrences retained and reported separately',
                numerical_gate='Diagnostic only. Neither modifies nor relaxes existing strict comparisons.',
                reference_only=classified_ref, local_only=classified_local,
                unique_shared_differences=deltas,
                duplicate_reference_records={':'.join(k):v for k,v in hg.items() if len(v)>1})


def inspect_missing_metadata(existing):
    """Inspect only small metadata and QC arrays in reference-only local files."""
    gdac = Path(existing['provenance']['gdac'])
    rows = []
    for record in existing['reference_only']:
        if record['local_index'] is None:
            continue
        path = gdac / 'dac' / record['local_index']['path']
        with xr.open_dataset(path, engine='netcdf4') as ds:
            row = dict(identity=record['identity'], reason=record['reason'], path=str(path),
                       sizes=dict(ds.sizes),
                       data_modes=[text(v) for v in ds.DATA_MODE.values],
                       date_update=text(ds.DATE_UPDATE.values.item()),
                       juld=[str(v) for v in ds.JULD.values])
            for name in ['JULD_QC', 'POSITION_QC', 'PRES_ADJUSTED_QC', 'TEMP_ADJUSTED_QC', 'PSAL_ADJUSTED_QC']:
                if name in ds:
                    row[name.lower() + '_counts'] = dict(Counter(text(v) for v in ds[name].values.ravel()))
            if 'PRES_ADJUSTED_ERROR' in ds:
                error = ds.PRES_ADJUSTED_ERROR.values
                row['pressure_adjusted_error_finite_count'] = int(np.isfinite(error).sum())
                row['pressure_adjusted_error_above_20_count'] = int((error > 20).sum())
            rows.append(row)
    groups = sorted(set(r['reason'] for r in rows))
    summary = {reason: dict(files=sum(r['reason'] == reason for r in rows),
                            data_mode_combinations=dict(Counter(','.join(r['data_modes']) for r in rows if r['reason'] == reason)),
                            no_pressure_adjusted_error_files=sum(r['pressure_adjusted_error_finite_count'] == 0 for r in rows if r['reason'] == reason))
               for reason in groups}
    return dict(summary=summary, files=rows,
                note='Historical research-mode raw files omit DATA_MODE and QC fields. Their previous data mode cannot be read directly from those saved arrays.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('/swot/SUM05/amf2288/argo_mixing/natre_validation_20261110'))
    parser.add_argument('--gdac', type=Path, default=Path('/swot/SUM05/dbalwada/Argo_sync'))
    parser.add_argument('--reference', type=Path, default=Path('/swot/SUM05/amf2288/espe_code/data_files'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--metadata-from', type=Path, help='Inspect reference-only local files listed in an existing identity audit JSON.')
    args = parser.parse_args()
    if args.output and args.output.resolve().is_relative_to(Path.home()):
        parser.error('Report output must be outside the home directory.')
    result = inspect_missing_metadata(json.loads(args.metadata_from.read_text())) if args.metadata_from else audit(args.run, args.gdac, args.reference)
    result = json_safe(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result['summary'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
