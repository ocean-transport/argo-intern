"""Pooled, deterministic Espe-style isopycnal diffusivities from saved profiles.

Each 3-degree box uses all dates, observation-weighted overlapping neutral-density
windows, and separate zonal/meridional regressions. No bootstrap is called.
The existing profile and binned products are read-only during computation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import warnings

import gsw
import numpy as np
from netCDF4 import Dataset
from scipy.stats import linregress
import xarray as xr

from global_binned_means import horizontal_cells

DENSITY = np.round(24.75 + np.arange(67) * 0.05, 2)
HALF_WIDTH = 0.05
GAMMA_MIX = 0.2
INPUT_FIELDS = ('CT', 'SA', 'gamman', 'pressure', 'dCTdz', 'dSAdz',
                'chi', 'chiS', 'epsilon')
BASE_FIELDS = {
    'iso_pressure_mean': 'dbar', 'iso_CT': 'degC', 'iso_SA': 'g kg-1',
    'iso_chi': 'K2 s-1', 'iso_chiS': '(g kg-1)2 s-1',
    'iso_epsilon': 'W kg-1', 'iso_N2': 's-2',
    'iso_dCTdz': 'K m-1', 'iso_dSAdz': 'g kg-1 m-1',
    'iso_Pdia': 'K2 s-1', 'iso_PSdia': '(g kg-1)2 s-1',
    'iso_Piso': 'K2 s-1', 'iso_PSiso': '(g kg-1)2 s-1',
    'KisoT': 'm2 s-1', 'KisoS': 'm2 s-1',
    'hgradCT': 'K m-1', 'hgradSA': 'g kg-1 m-1',
}
for _tracer, _unit in [('CT', 'K m-1'), ('SA', 'g kg-1 m-1')]:
    for _direction in ('x', 'y'):
        _name = f'd{_tracer}d{_direction}'
        BASE_FIELDS.update({_name: _unit, _name + '_stderr': _unit,
                            _name + '_pvalue': '1', _name + '_rvalue': '1'})
COUNT_FIELDS = ('iso_observation_count', 'iso_profile_count',
                'iso_epsilon_count', 'iso_regression_count')
PRESSURE_FIELDS = ('KisoT', 'KisoS', 'dCTdx', 'dCTdy', 'dSAdx', 'dSAdy',
                   'hgradCT', 'hgradSA')
PRESSURE_MAPPING = (
    'Linear interpolation at pressure_bin centers between adjacent density '
    'layers using iso_pressure_mean. Both endpoint estimates must be finite '
    'and pressures strictly increasing. No extrapolation, no bridging missing '
    'layers, and ambiguous pressure crossings are masked. These are point '
    'estimates at bin centers, not averages over pressure-bin bounds.'
)


def _read(variable, selection=slice(None)):
    return np.asarray(np.ma.filled(variable[selection], np.nan), dtype=float)


def log_interpolate(values, source_depth, target_depth):
    """Legacy log-depth interpolation, with finite positive input support."""
    out = np.full((len(target_depth), values.shape[1]), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log10(values)
    for j in range(values.shape[1]):
        valid = np.isfinite(logs[:, j])
        if valid.any():
            out[:, j] = 10 ** np.interp(
                target_depth, source_depth[valid], logs[valid, j],
                left=np.nan, right=np.nan)
    return out


def prepare_mixing(fields, source_depth, target_depth):
    """Clip the upper 3-sigma log-epsilon tail per box/depth, then interpolate."""
    epsilon = fields['epsilon']
    with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        logs = np.log10(epsilon)
        threshold = 10 ** (np.nanmean(logs, axis=1, keepdims=True)
                           + 3 * np.nanstd(logs, axis=1, keepdims=True))
    bad = epsilon > threshold
    result = {}
    for name in ('chi', 'chiS', 'epsilon'):
        values = fields[name].copy()
        values[bad] = np.nan
        result[name] = log_interpolate(values, source_depth, target_depth)
    return result, int(bad.sum())


def _finite_mean(values, minimum=1):
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size >= minimum else np.nan


def calculate_box(fields, latitude, longitude, depth, finescale_depth):
    """One all-date box. Inputs are independent of the existing binned means."""
    result = {name: np.full(len(DENSITY), np.nan) for name in BASE_FIELDS}
    result.update({name: np.zeros(len(DENSITY), dtype=np.int64) for name in COUNT_FIELDS})
    if not len(latitude):
        return result, 0
    # Unwrap around the local circular mean, including the dateline.
    center = np.degrees(np.arctan2(np.mean(np.sin(np.radians(longitude))),
                                   np.mean(np.cos(np.radians(longitude)))))
    lon = center + (longitude - center + 180) % 360 - 180
    mix, clipped = prepare_mixing(fields, finescale_depth, depth)
    common = np.isfinite(mix['chi'] + mix['chiS'])
    cell_latitude = float(np.mean(latitude))
    lat_span, lon_span = np.ptp(latitude), np.ptp(lon)
    for k, density in enumerate(DENSITY):
        use = (np.abs(fields['gamman'] - density) <= HALF_WIDTH) & common
        iz, ip = np.nonzero(use)
        n = len(ip)
        result['iso_observation_count'][k] = n
        result['iso_profile_count'][k] = np.count_nonzero(np.any(use, axis=0))
        if not n:
            continue
        ct, sa = fields['CT'][use], fields['SA'][use]
        # CT/SA use ordinary means, matching the original regional calculation.
        result['iso_CT'][k], result['iso_SA'][k] = np.mean(ct), np.mean(sa)
        pressure = _finite_mean(fields['pressure'][use])
        result['iso_pressure_mean'][k] = pressure
        result['iso_chi'][k] = _finite_mean(mix['chi'][use], minimum=5)
        result['iso_chiS'][k] = _finite_mean(mix['chiS'][use], minimum=5)
        eps = mix['epsilon'][use]
        result['iso_epsilon_count'][k] = np.isfinite(eps).sum()
        epsilon = _finite_mean(eps, minimum=5)
        result['iso_epsilon'][k] = epsilon
        dctdz = -_finite_mean(fields['dCTdz'][use])
        dsadz = -_finite_mean(fields['dSAdz'][use])
        result['iso_dCTdz'][k], result['iso_dSAdz'][k] = dctdz, dsadz
        grav = gsw.grav(cell_latitude, pressure)
        alpha = gsw.alpha(result['iso_SA'][k], result['iso_CT'][k], pressure)
        beta = gsw.beta(result['iso_SA'][k], result['iso_CT'][k], pressure)
        n2 = -grav * (-dctdz * alpha + dsadz * beta)
        result['iso_N2'][k] = n2
        if np.isfinite(n2) and n2 > 0 and np.isfinite(epsilon):
            result['iso_Pdia'][k] = 2 * GAMMA_MIX * epsilon * dctdz**2 / n2
            result['iso_PSdia'][k] = 2 * GAMMA_MIX * epsilon * dsadz**2 / n2
        # Retain signed residuals; nonpositive production cannot yield Kiso.
        result['iso_Piso'][k] = result['iso_chi'][k] - result['iso_Pdia'][k]
        result['iso_PSiso'][k] = result['iso_chiS'][k] - result['iso_PSdia'][k]
        x, y = lon[ip], latitude[ip]
        # Preserve Espe's checks: >10 profiles in the box, >10 layer samples,
        # and layer coverage exceeding a third of each observed box span.
        coverage = (n > 10 and len(latitude) > 10
                    and np.ptp(x) > lon_span / 3 and np.ptp(y) > lat_span / 3)
        if not coverage:
            continue
        result['iso_regression_count'][k] = n
        factorx = 111e3 * np.cos(np.radians(np.mean(y)))
        for tracer, values, residual, output in [
            ('CT', ct, 'iso_Piso', 'KisoT'),
            ('SA', sa, 'iso_PSiso', 'KisoS'),
        ]:
            if not np.all(np.isfinite(values)):
                continue
            for direction, coordinate, factor in [('x', x, factorx), ('y', y, 111e3)]:
                fit = linregress(coordinate, values)
                name = f'd{tracer}d{direction}'
                result[name][k] = fit.slope / factor
                result[name + '_stderr'][k] = fit.stderr / factor
                result[name + '_pvalue'][k] = fit.pvalue
                result[name + '_rvalue'][k] = fit.rvalue
            gradient2 = result[f'd{tracer}dx'][k]**2 + result[f'd{tracer}dy'][k]**2
            result[f'hgrad{tracer}'][k] = np.sqrt(gradient2)
            production = result[residual][k]
            if np.isfinite(gradient2) and gradient2 > 0 and production > 0:
                value = production / (2 * gradient2)
                if np.isfinite(value):
                    result[output][k] = value
    return result, clipped


def pressure_interpolate(pressure, values, targets):
    """Interpolate adjacent density layers only; retain gaps and mask folds."""
    out = np.full(len(targets), np.nan)
    valid = np.isfinite(pressure) & np.isfinite(values)
    for j, target in enumerate(targets):
        exact = np.flatnonzero(valid & (pressure == target))
        if len(exact) == 1:
            out[j] = values[exact[0]]
            continue
        if len(exact) > 1:
            continue
        candidates = np.flatnonzero(valid[:-1] & valid[1:]
                                   & (pressure[:-1] < target) & (pressure[1:] > target))
        if len(candidates) == 1:
            i = candidates[0]
            fraction = (target - pressure[i]) / (pressure[i + 1] - pressure[i])
            out[j] = values[i] + fraction * (values[i + 1] - values[i])
    return out


def _box_task(task):
    profile_path, checkpoint, cell, indices = task
    destination = Path(checkpoint) / f'{cell:04d}.npz'
    if destination.exists():
        return cell, True
    with Dataset(profile_path) as source:
        fields = {name: _read(source[name], (slice(None), indices)) for name in INPUT_FIELDS}
        result, clipped = calculate_box(
            fields, _read(source['latitude'], indices), _read(source['longitude'], indices),
            _read(source['Z']), _read(source['Zfs']))
    np.savez_compressed(destination, **result, clipped_epsilon=clipped, profiles=len(indices))
    return cell, False


def assemble(checkpoint, cells, latitude, longitude, pressure, provenance):
    shape = (len(DENSITY), 60, 120)
    arrays = {name: np.full(shape, np.nan) for name in BASE_FIELDS}
    arrays.update({name: np.zeros(shape, dtype=np.int64) for name in COUNT_FIELDS})
    profiles = np.zeros((60, 120), dtype=np.int64)
    clipped = np.zeros_like(profiles)
    for cell in cells:
        iy, ix = divmod(int(cell), 120)
        with np.load(Path(checkpoint) / f'{cell:04d}.npz') as result:
            for name in arrays:
                arrays[name][:, iy, ix] = result[name]
            profiles[iy, ix], clipped[iy, ix] = result['profiles'], result['clipped_epsilon']
    dims = ('isopycnal_density', 'latitude', 'longitude')
    ds = xr.Dataset(coords={'isopycnal_density': DENSITY, 'latitude': latitude,
                            'longitude': longitude, 'pressure_bin': pressure})
    ds.isopycnal_density.attrs.update(units='kg m-3', long_name='Neutral-density layer center',
                                      window_half_width=HALF_WIDTH)
    for name, array in arrays.items():
        ds[name] = (dims, array, {'units': BASE_FIELDS.get(name, '1'),
                                'analysis': 'All-date pooled isopycnal calculation; no bootstrap'})
    for tracer, unit in [('CT', 'K m-1'), ('SA', 'g kg-1 m-1')]:
        ds[f'hgrad{tracer}'].attrs['long_name'] = f'Magnitude of horizontal {tracer} gradient on neutral-density layers'
        for direction, label in [('x', 'zonal'), ('y', 'meridional')]:
            name = f'd{tracer}d{direction}'
            ds[name].attrs['long_name'] = f'{label.capitalize()} {tracer} gradient on neutral-density layers'
            ds[name + '_stderr'].attrs['long_name'] = 'Regression slope standard error (no bootstrap)'
            ds[name + '_pvalue'].attrs['long_name'] = 'Two-sided regression slope p value (no bootstrap)'
            ds[name + '_rvalue'].attrs['long_name'] = 'Regression Pearson correlation coefficient'
    for name, tracer, residual in [('KisoT', 'CT', 'iso_Piso'), ('KisoS', 'SA', 'iso_PSiso')]:
        ds[name].attrs.update(long_name=f'Isopycnal diffusivity from {tracer}',
                              formula=f'{residual} / (2 * (d{tracer}dx**2 + d{tracer}dy**2))',
                              nonpositive_production='masked; signed residual is saved separately')
    ds.iso_pressure_mean.attrs['long_name'] = 'Mean measured pressure of common-support layer samples'
    ds.iso_profile_count.attrs['long_name'] = 'Distinct saved profile records contributing common-support samples'
    ds.iso_observation_count.attrs['long_name'] = 'Common-support depth samples; overlapping layers are not independent'
    ds.iso_regression_count.attrs['long_name'] = 'Layer samples passing the spatial-coverage screen'
    ds['iso_cell_profile_count'] = (('latitude', 'longitude'), profiles)
    ds['iso_clipped_epsilon_count'] = (('latitude', 'longitude'), clipped)
    for name in PRESSURE_FIELDS:
        mapped = np.full((len(pressure), 60, 120), np.nan)
        for cell in cells:
            iy, ix = divmod(int(cell), 120)
            mapped[:, iy, ix] = pressure_interpolate(
                arrays['iso_pressure_mean'][:, iy, ix], arrays[name][:, iy, ix], pressure)
        ds[name + '_pressure'] = (('pressure_bin', 'latitude', 'longitude'), mapped,
                                  dict(ds[name].attrs, pressure_mapping=PRESSURE_MAPPING))
    ds.attrs['isopycnal_analysis_json'] = json.dumps(provenance, sort_keys=True)
    ds.attrs['isopycnal_pressure_mapping'] = PRESSURE_MAPPING
    return ds


def compute(profile_path, binned_path, output_path, checkpoint, workers=4, selected_cells=None):
    """Compute a separate addition file; preserve the original NetCDF unchanged."""
    profile_path, binned_path = Path(profile_path).resolve(), Path(binned_path).resolve()
    output_path, checkpoint = Path(output_path).resolve(), Path(checkpoint).resolve()
    if output_path in (profile_path, binned_path):
        raise ValueError('Computation output must be separate from both input products')
    with Dataset(profile_path) as source:
        lat, lon = _read(source['latitude']), _read(source['longitude'])
        profile_cells, valid = horizontal_cells(lat, lon)
        times = np.asarray(source['time'][:], dtype=np.int64)
        time_units = source['time'].units
        source_manifest = source.getncattr('source_manifest')
    if time_units != 'nanoseconds since 1970-01-01 00:00:00':
        raise ValueError('Unexpected source time units')
    with xr.open_dataset(binned_path, engine='netcdf4') as binned:
        latitude, longitude, pressure = (binned[n].values.copy() for n in ('latitude', 'longitude', 'pressure_bin'))
    np.testing.assert_array_equal(latitude, np.arange(-88.5, 90, 3))
    np.testing.assert_array_equal(longitude, np.arange(-178.5, 180, 3))
    cells = np.unique(profile_cells[valid])
    if selected_cells is not None:
        cells = np.intersect1d(cells, selected_cells)
    stat = profile_path.stat()
    provenance = {
        'method': 'Pooled deterministic adaptation of Espe aggregate_region',
        'source_profiles': str(profile_path), 'source_manifest': source_manifest,
        'source_bytes': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'horizontal_degrees': 3.0, 'density_centers': DENSITY.tolist(),
        'density_half_width': HALF_WIDTH, 'gamma_mix': GAMMA_MIX, 'bootstrap_count': 0,
        'time_selection': 'all saved dates in one pooled window',
        'time_start': str(times.min().astype('datetime64[ns]')),
        'time_end': str(times.max().astype('datetime64[ns]')),
        'weighting': 'equal weight per valid depth sample within each box/layer',
        'common_support': 'abs(gamman-center)<=0.05 and finite interpolated chi+chiS',
        'filter': 'per-box/depth epsilon > 10**(mean(log10(epsilon))+3*std(log10(epsilon))) excluded',
        'interpolation': 'log-linear from Zfs to Z; finite positive support; no extrapolation',
        'Pdia': '2*gamma_mix*mean(epsilon)*mean(dCTdz)**2/N2; analogous for SA',
        'N2': 'TEOS-10 from pooled mean CT, SA, pressure, vertical gradients, and box mean latitude',
        'negative_production': 'signed residual retained; Kiso masked for nonpositive residual',
        'regression': 'separate longitude and latitude OLS, metres per degree 111000; zonal cos(sample mean latitude)',
        'coverage': '>10 box profiles, >10 layer samples, each layer spatial span > one third of box observed span',
        'missing_gradient': 'invalid or zero squared horizontal gradient gives missing Kiso; no small-gradient cutoff',
        'pressure_mapping': PRESSURE_MAPPING,
        'profiles_in_source': len(lat), 'invalid_position_profiles': int((~valid).sum()),
        'populated_boxes': len(cells), 'full_global_run': selected_cells is None,
    }
    checkpoint.mkdir(parents=True, exist_ok=True)
    config_file = checkpoint / 'config.json'
    if config_file.exists():
        if json.loads(config_file.read_text()) != provenance:
            raise ValueError('Checkpoint configuration/source/code changed; use a new checkpoint directory')
    else:
        config_file.write_text(json.dumps(provenance, indent=2) + '\n')
    order = np.argsort(profile_cells, kind='stable')
    starts = np.searchsorted(profile_cells[order], cells)
    stops = np.searchsorted(profile_cells[order], cells, side='right')
    tasks = [(str(profile_path), str(checkpoint), int(cell), order[start:stop])
             for cell, start, stop in zip(cells, starts, stops)]
    started = time.monotonic()
    print(f'Computing {len(cells)} boxes, {sum(len(t[3]) for t in tasks):,} profiles, {workers} workers; no bootstrap.', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for completed, _ in enumerate(pool.map(_box_task, tasks, chunksize=1), 1):
            if completed % 100 == 0 or completed == len(tasks):
                elapsed = time.monotonic() - started
                print(f'{completed}/{len(tasks)} boxes; {elapsed:.1f}s elapsed; '
                      f'~{elapsed*(len(tasks)/completed-1)/60:.1f} min remaining', flush=True)
    result = assemble(checkpoint, cells, latitude, longitude, pressure, provenance)
    scanned = int(result.iso_cell_profile_count.sum())
    assert scanned == sum(len(t[3]) for t in tasks)
    report = dict(provenance, profiles_scanned=scanned, elapsed_seconds=time.monotonic()-started,
                  clipped_epsilon_samples=int(result.iso_clipped_epsilon_count.sum()),
                  finite_estimates={n: int(np.isfinite(result[n]).sum()) for n in (*PRESSURE_FIELDS, 'KisoT_pressure', 'KisoS_pressure')})
    result.attrs['isopycnal_report_json'] = json.dumps(report, sort_keys=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix('.partial.nc')
    result.to_netcdf(temporary, engine='netcdf4', encoding={
        n: {'zlib': True, 'complevel': 2, '_FillValue': None} for n in result.data_vars})
    with xr.open_dataset(temporary, engine='netcdf4') as reread:
        xr.testing.assert_identical(result, reread.load())
    temporary.replace(output_path)
    output_path.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output_path), 'profiles_scanned': scanned,
                      'finite_estimates': report['finite_estimates']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', required=True)
    parser.add_argument('--binned', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cells', type=int, nargs='+', help='Optional pilot boxes; never a full global product')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    compute(args.profiles, args.binned, args.output, args.checkpoint, args.workers, args.cells)


if __name__ == '__main__':
    main()
