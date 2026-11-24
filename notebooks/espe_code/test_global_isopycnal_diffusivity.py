"""Scientific checks for pooled diffusivity, screening, and pressure mapping."""
import unittest

import gsw
import numpy as np

from global_isopycnal_diffusivity import (
    DENSITY, GAMMA_MIX, calculate_box, log_interpolate,
    prepare_mixing, pressure_interpolate,
)


def synthetic_box():
    lon, lat = np.meshgrid(np.linspace(-32.9, -30.1, 4), np.linspace(24.1, 26.9, 4))
    lon, lat = lon.ravel(), lat.ravel()
    z = np.array([50., 100., 150.])
    xx, yy = lon[None, :] + 31.5, lat[None, :] - 25.5
    zz = z[:, None] - 50
    shape = (3, len(lat))
    fields = {
        'CT': 12 + .03 * xx + .02 * yy - .01 * zz,
        'SA': 35 + .003 * xx - .002 * yy + .0005 * zz,
        'gamman': np.broadcast_to(np.array([25., 25.05, 25.1])[:, None], shape).copy(),
        'pressure': np.broadcast_to(z[:, None] * 1.03, shape).copy(),
        'dCTdz': np.full(shape, -.01), 'dSAdz': np.full(shape, .0005),
        'chi': np.full(shape, 1e-7), 'chiS': np.full(shape, 1e-9),
        'epsilon': np.full(shape, 1e-10),
    }
    return fields, lat, lon, z


class IsopycnalTest(unittest.TestCase):
    def test_known_slopes_and_diffusivity_identity(self):
        fields, lat, lon, z = synthetic_box()
        original = {k: v.copy() for k, v in fields.items()}
        result, clipped = calculate_box(fields, lat, lon, z, z)
        k = np.flatnonzero(DENSITY == 25.05)[0]
        selected = abs(fields['gamman'] - 25.05) <= .05
        n = selected.sum()
        self.assertEqual(result['iso_observation_count'][k], n)
        self.assertEqual(result['iso_profile_count'][k], 16)
        self.assertEqual(result['iso_regression_count'][k], n)
        self.assertEqual(clipped, 0)
        factor = 111e3 * np.cos(np.radians(lat.mean()))
        for name, slope in [('dCTdx', .03/factor), ('dCTdy', .02/111e3),
                            ('dSAdx', .003/factor), ('dSAdy', -.002/111e3)]:
            np.testing.assert_allclose(result[name][k], slope, rtol=2e-12)
        ct, sa = fields['CT'][selected].mean(), fields['SA'][selected].mean()
        pressure = fields['pressure'][selected].mean()
        n2 = gsw.grav(lat.mean(), pressure) * (
            .01 * gsw.alpha(sa, ct, pressure) + .0005 * gsw.beta(sa, ct, pressure))
        for tracer, production, slope, name in [('CT', 1e-7, .01, 'KisoT'),
                                               ('SA', 1e-9, .0005, 'KisoS')]:
            expected_piso = production - 2*GAMMA_MIX*1e-10*slope**2/n2
            denom = 2*(result[f'd{tracer}dx'][k]**2 + result[f'd{tracer}dy'][k]**2)
            np.testing.assert_allclose(result[name][k], expected_piso/denom, rtol=1e-12)
        for name in fields:
            np.testing.assert_array_equal(fields[name], original[name])

    def test_nonpositive_residual_and_zero_gradient_are_missing(self):
        fields, lat, lon, z = synthetic_box()
        fields['chiS'][:] = 1e-20
        fields['CT'][:] = 12
        result, _ = calculate_box(fields, lat, lon, z, z)
        self.assertTrue(np.any(result['iso_PSiso'] < 0))
        self.assertTrue(np.isnan(result['KisoS']).all())
        self.assertTrue(np.isnan(result['KisoT']).all())

    def test_common_support_counts_and_sparse_spatial_screen(self):
        fields, lat, lon, z = synthetic_box()
        # Missing salinity dissipation excludes this profile from both tracers.
        fields['chiS'][:, 0] = np.nan
        result, _ = calculate_box(fields, lat, lon, z, z)
        k = np.flatnonzero(DENSITY == 25.05)[0]
        self.assertEqual(result['iso_profile_count'][k], 15)
        result, _ = calculate_box(fields, lat, np.full_like(lon, lon.mean()), z, z)
        self.assertTrue(np.isnan(result['dCTdx']).all())
        self.assertTrue(np.isnan(result['KisoT']).all())

    def test_clipping_and_log_interpolation(self):
        fields, lat, lon, z = synthetic_box()
        fields['epsilon'][1, 0] = 1
        fields['chi'][1, 0] = 1
        mix, count = prepare_mixing(fields, z, z)
        self.assertEqual(count, 1)
        # The clipped interior sample is bridged in log space, as in Espe's code.
        np.testing.assert_allclose(mix['chi'][1, 0], 1e-7)
        a = log_interpolate(np.array([[1.], [100.]]), np.array([0., 100.]),
                            np.array([-1., 0., 50., 100., 101.]))[:, 0]
        np.testing.assert_allclose(a, [np.nan, 1, 10, 100, np.nan], equal_nan=True)

    def test_dateline_unwrap(self):
        fields, lat, lon, z = synthetic_box()
        before, _ = calculate_box(fields, lat, lon, z, z)
        # Shift the complete box across the dateline without changing its geometry.
        shifted_lon = (lon + 211.5 + 180) % 360 - 180
        after, _ = calculate_box(fields, lat, shifted_lon, z, z)
        for name in ('dCTdx', 'dSAdx', 'KisoT', 'KisoS'):
            np.testing.assert_allclose(before[name], after[name], rtol=1e-12, equal_nan=True)

    def test_pressure_mapping_preserves_gaps_and_no_extrapolation(self):
        p = np.array([100., 200., 300., 400.])
        v = np.array([1., 2., np.nan, 4.])
        targets = np.array([50., 100., 150., 200., 250., 350., 400., 450.])
        expected = [np.nan, 1., 1.5, 2., np.nan, np.nan, 4., np.nan]
        np.testing.assert_allclose(pressure_interpolate(p, v, targets), expected, equal_nan=True)
        # Two increasing segments cross this pressure: leave the estimate missing.
        folded = pressure_interpolate(np.array([100., 300., 150., 400.]),
                                      np.array([1., 2., 3., 4.]), np.array([200.]))
        self.assertTrue(np.isnan(folded[0]))


if __name__ == '__main__':
    unittest.main()
