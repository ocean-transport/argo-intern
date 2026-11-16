"""Consolidation must preserve bits, timestamps, strings and independent grids."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from global_netcdf import consolidate_profiles, merge_binned_grids, split_binned_grids, save_binned_grids


class GlobalNetCDFTest(unittest.TestCase):
    def test_profiles_roundtrip_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            values = np.array([0, 0x8000000000000000, 0x7ff8000000000001, 0x7ff8000000000002],
                              dtype=np.uint64).view(np.float64).reshape(2, 2)
            for i in range(2):
                path = root / f'part_{i}.nc'
                files.append(str(path))
                with Dataset(path, 'w') as d:
                    d.createDimension('time', 2); d.createDimension('Z', 2)
                    z = d.createVariable('Z', 'f8', ('Z',)); z[:] = [2, 4]
                    t = d.createVariable('time', 'i8', ('time',))
                    t.units = ('seconds' if i == 0 else 'milliseconds') + ' since 2014-01-01 00:00:00'
                    t.calendar = 'proleptic_gregorian'; t[:] = [0, 1] if i == 0 else [2000, 2001]
                    for name in ['latitude', 'longitude']:
                        d.createVariable(name, 'f8', ('time',))[:] = [20., 21.]
                    d.createVariable('gamma_n_status', 'i2', ('time',))[:] = [0, 5]
                    d.createVariable('value', 'f8', ('Z', 'time'))[:] = values
                    d.createVariable('profile_id', str, ('time',))[:] = np.array(['aß', 'γ'], dtype=object)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'status': 'complete', 'failed_batches': [],
                                            'accepted_profiles': 4, 'regions': {'cell': files}}))
            out = root / 'global.nc'
            with patch.object(Path, 'is_relative_to', return_value=True):
                report = consolidate_profiles(manifest, out, workers=1, files_per_task=1)
                reused = consolidate_profiles(manifest, out, workers=1, files_per_task=1)
            self.assertEqual(report, reused)
            with Dataset(out) as d:
                d.set_auto_maskandscale(False)
                self.assertEqual(d['value'][:].tobytes(), np.concatenate([values, values], axis=1).tobytes())
                self.assertEqual(d['profile_id'][:].tolist(), ['aß', 'γ', 'aß', 'γ'])
            with xr.open_dataset(out, engine='netcdf4') as d:
                expected = np.datetime64('2014-01-01', 'ns') + np.array([0, 1000000000, 2000000000, 2001000000], dtype='timedelta64[ns]')
                np.testing.assert_array_equal(d.time.values, expected)

    def test_binned_two_vertical_grids_roundtrip(self):
        coords = {'latitude': [1.5], 'longitude': [1.5]}
        grids = []
        for dim, levels in [('pressure_bin', [25., 75.]), ('neutral_density_bin', [26.025, 26.075, 26.125])]:
            n = len(levels)
            ds = xr.Dataset({
                'Pdia': ((dim, 'latitude', 'longitude'), np.arange(n, dtype=float).reshape(n, 1, 1)),
                'Pdia_count': ((dim, 'latitude', 'longitude'), np.ones((n, 1, 1), dtype=np.int64)),
                dim + '_bounds': ((dim, 'bounds'), np.column_stack([levels, levels])),
                'latitude_bounds': (('latitude', 'bounds'), [[0., 3.]]),
                'longitude_bounds': (('longitude', 'bounds'), [[0., 3.]]),
                'cell_profile_count': (('latitude', 'longitude'), [[1]]),
            }, coords={**coords, dim: levels}, attrs={'accepted_profiles': np.int64(1)})
            grids.append(ds)
        merged = merge_binned_grids(*grids)
        self.assertEqual(merged.Pdia_pressure.dims, ('pressure_bin', 'latitude', 'longitude'))
        self.assertEqual(merged.Pdia_density.dims, ('neutral_density_bin', 'latitude', 'longitude'))
        for actual, expected in zip(split_binned_grids(merged), grids):
            xr.testing.assert_identical(actual, expected)
        with tempfile.TemporaryDirectory() as tmp, patch.object(Path, 'is_relative_to', return_value=True):
            out = Path(tmp) / 'bins.nc'
            report = {'source_manifest': 'test.json', 'signature': 'test_signature'}
            save_binned_grids(*grids, out, report)
            save_binned_grids(*grids, out, report)
            with xr.open_dataset(out, engine='netcdf4') as d:
                for actual, expected in zip(split_binned_grids(d), grids):
                    xr.testing.assert_identical(actual, expected)


if __name__ == '__main__':
    unittest.main()
