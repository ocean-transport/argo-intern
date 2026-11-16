"""Boundary, weighting and conservation checks for the new global averages."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import xarray as xr

from global_binned_means import (
    AXES, METRICS, SOURCE_METRICS, NCELL, _accumulate, _combine, _dataset, _reduce_files,
    horizontal_cells, mixing_values, pressure_on_finescale, reduce_samples, vertical_indices,
)


class GlobalBinsTest(unittest.TestCase):
    def test_horizontal_boundaries_and_wrap(self):
        cells, valid = horizontal_cells(
            [-90, -90, -87, 90, 90.01, 0, 0, 0],
            [-180, 180, -177, 360, 0, np.nan, 361, -181])
        np.testing.assert_array_equal(valid, [1, 1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(cells, [0, 0, 121, 59 * 120 + 60, -1, -1, -1, -1])

    def test_vertical_edges_are_half_open(self):
        for width in [50., .05]:
            for k in [-4, 0, 1, 534]:
                edge = k * width
                vals = np.array([np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)])
                np.testing.assert_array_equal(vertical_indices(vals, width), [k - 1, k, k])

    def test_pressure_uses_observed_depth_support_not_depth_as_pressure(self):
        z = np.array([25, 26, 50, 74, 75, 76, 100, 124, 125.])[:, None]
        z = np.column_stack([z, z])
        p = z * 1.07 + 2
        p[2, 1] = np.nan
        centers = np.array([50, 100])
        expected = np.full((2, 2), np.nan)
        for i, center in enumerate(centers):
            for j in range(2):
                chosen = (np.abs(z[:, j] - center) < 25) & np.isfinite(p[:, j])
                expected[i, j] = np.mean(p[chosen, j])
        np.testing.assert_array_equal(pressure_on_finescale(p, z, centers), expected)
        self.assertFalse(np.array_equal(expected[:, 0], centers))

    def test_counts_weight_observations_and_retain_negative_residuals(self):
        coordinate = np.array([[26.01, 26.02], [26.03, 26.08], [np.nan, 26.07]])
        values = [np.array([[1., 10.], [3., np.nan], [99., 20.]]) for _ in METRICS]
        values[-1] = np.array([[-2., 5.], [-4., 7.], [99., 9.]])
        cells = np.array([10, 10])
        block = reduce_samples(coordinate, values, cells, .05)
        k, total, n, observations, profiles = block
        np.testing.assert_array_equal(k, [520 * NCELL + 10, 521 * NCELL + 10])
        np.testing.assert_array_equal(n[:, 0], [3, 1])
        np.testing.assert_array_equal(total[:, 0], [14, 20])
        np.testing.assert_array_equal(total[:, -1], [-1, 16])
        np.testing.assert_array_equal(observations, [3, 2])
        np.testing.assert_array_equal(profiles, [2, 1])
        # Missing vertical coordinates and an invalid position cannot contribute.
        invalid = reduce_samples(coordinate, values, np.array([-1, -1]), .05)
        self.assertEqual(len(invalid[0]), 0)

    def test_streamed_files_match_independent_observation_loop(self):
        z = np.array([[26., 26.], [50., 50.], [74., 74.],
                      [76., 76.], [100., 100.], [124., 124.]])
        centers = np.array([50, 100])
        metric = np.array([[1., 4.], [3., 8.]])
        density = np.array([[26.01, np.nan], [26.02, 26.08]])
        source = xr.Dataset({
            'latitude': ('time', [0., 0.]), 'longitude': ('time', [0., 0.]),
            'Zmeas': (('Z', 'time'), z), 'pressure': (('Z', 'time'), z * 1.01),
            'gamman_fs': (('Zfs', 'time'), density),
            **{name: (('Zfs', 'time'), metric.copy()) for name in SOURCE_METRICS},
        }, coords={'Z': np.arange(6), 'Zfs': centers, 'time': np.arange(2)})
        source['chi'].values[0, 0] = -1  # negative Piso retained
        source['chiS'].values[0, 0] = -2  # negative PSiso retained
        source['Krho'].values[1, 1] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            paths = [str(Path(tmp) / f'{i}.nc') for i in range(2)]
            for path in paths:
                source.to_netcdf(path, engine='netcdf4')
            full = _reduce_files((paths, (50., .05)))
            chunks = [_reduce_files(([path], (50., .05))) for path in paths]
            for axis in range(2):
                merged = _combine([c['blocks'][axis] for c in chunks])
                for a, b in zip(full['blocks'][axis], merged):
                    np.testing.assert_array_equal(a, b)
                store = {}
                _accumulate(store, full['blocks'][axis])
                ds = _dataset(store, AXES[axis], (50., .05)[axis], full['cell_profiles'])
                coordinate = pressure_on_finescale(source.pressure.values, z, centers) if axis == 0 else density
                for name in METRICS:
                    if name == 'Piso':
                        a = (source.chi - source.Pdia).values
                    elif name == 'PSiso':
                        a = (source.chiS - source.PdiaS).values
                    else:
                        a = source['PdiaS' if name == 'PSdia' else name].values
                    expected = {}
                    for _ in paths:
                        for i in range(2):
                            for j in range(2):
                                if np.isfinite(coordinate[i, j]) and np.isfinite(a[i, j]):
                                    key = int(np.floor(coordinate[i, j] / (50., .05)[axis]))
                                    expected.setdefault(key, []).append(a[i, j])
                    for key, values in expected.items():
                        center = (key + .5) * (50., .05)[axis]
                        got = ds.sel({AXES[axis]: center, 'latitude': 1.5, 'longitude': 1.5})
                        self.assertEqual(got[name + '_count'].item(), len(values))
                        self.assertEqual(got[name].item(), np.mean(values))
                    self.assertEqual(int(ds[name + '_count'].sum()), sum(map(len, expected.values())))
                # The serialized products retain float64 means and integer counts.
                out = Path(tmp) / f'grid_{axis}.nc'
                ds.to_netcdf(out, engine='netcdf4')
                with xr.open_dataset(out, engine='netcdf4') as reread:
                    xr.testing.assert_identical(ds, reread)

    def test_halinity_names_and_paired_residuals(self):
        raw = {name: np.array([[1., 2., 3.]]) for name in SOURCE_METRICS}
        raw['chiS'] = np.array([[2., np.nan, 1.]])
        raw['PdiaS'] = np.array([[1., 4., 5.]])
        values = dict(zip(METRICS, mixing_values(raw)))
        np.testing.assert_array_equal(values['PSdia'], [[1., 4., 5.]])
        np.testing.assert_array_equal(values['PSiso'], [[1., np.nan, -4.]])


if __name__ == '__main__':
    unittest.main()
