"""Regression checks for local-source selection and strict replication acceptance."""
import unittest

import process_argo_mixing as mixing
import natre_figure_review as review
import numpy as np
import xarray as xr


class BitwiseAcceptance(unittest.TestCase):
    def check(self, actual, expected, passed):
        report = mixing._comparison_array(actual, expected, 1, 1)
        self.assertEqual(report['passed'], passed, report)

    def test_tolerance_cannot_pass_a_different_bit(self):
        self.check(np.array([np.nextafter(1., 2.)]), np.array([1.]), False)

    def test_signed_zero_is_different(self):
        self.check(np.array([-0.]), np.array([0.]), False)

    def test_nan_payload_is_different(self):
        a = np.array([0x7ff8000000000001], dtype=np.uint64).view(np.float64)
        b = np.array([0x7ff8000000000002], dtype=np.uint64).view(np.float64)
        self.check(a, b, False)
        self.check(a, a.copy(), True)

    def test_dtype_is_required(self):
        self.check(np.array([1.], dtype='f4'), np.array([1.], dtype='f8'), False)

    def test_datetime_precision_and_nat(self):
        a = np.array(['2020-01-01', 'NaT'], dtype='datetime64[ns]')
        self.check(a, a.copy(), True)
        self.check(a, a.astype('datetime64[us]'), False)

    def test_review_reports_value_and_byte_equality_separately(self):
        a = xr.Dataset({'a': ('x', [-0.])})
        b = xr.Dataset({'a': ('x', [0.])})
        report = review.compare_datasets(a, b)['summary']
        self.assertTrue(report['all_reference_variables_exact'])
        self.assertFalse(report['all_reference_variables_bitwise'])

    def test_variable_dimension_order_is_required(self):
        expected = xr.Dataset({'CT': (('SG', 'TIME'), np.zeros((2, 2)))})
        actual = xr.Dataset({'CT': (('TIME', 'SG'), np.zeros((2, 2)))})
        self.assertFalse(mixing.compare_summary_datasets(actual, expected)['deterministic_passed'])

    def test_coordinate_dimensions_are_required(self):
        expected = xr.Dataset(coords={'SG': ('SG', [1., 2.])})
        actual = xr.Dataset(coords={'SG': ('different_dimension', [1., 2.])})
        self.assertFalse(mixing.compare_summary_datasets(actual, expected)['passed'])

    def test_profile_reordering_cannot_pass(self):
        ds = xr.Dataset({'longitude': ('time', [-30., -29.]),
                         'latitude': ('time', [25., 26.]), 'CT': ('time', [3., 4.])},
                        coords={'time': np.array(['2020-01-01', '2020-01-02'], dtype='datetime64[ns]')})
        self.assertTrue(mixing.compare_profile_datasets(ds, ds)['passed'])
        report = mixing.compare_profile_datasets(ds.isel(time=[1, 0]), ds)
        self.assertTrue(report['coverage']['passed'])
        self.assertFalse(report['profile_order_matches'])
        self.assertFalse(report['passed'])
        self.assertFalse(mixing.compare_profile_datasets(ds.isel(time=[0]), ds)['passed'])


class LocalSourceSelection(unittest.TestCase):
    def raw(self, schemes):
        return xr.Dataset({'PLATFORM_NUMBER': ('N_PROF', [b'1234567'] * len(schemes)),
                           'CYCLE_NUMBER': ('N_PROF', [4] * len(schemes)),
                           'DIRECTION': ('N_PROF', [b'A'] * len(schemes)),
                           'VERTICAL_SAMPLING_SCHEME': ('N_PROF', schemes)})

    def records(self):
        return [{'path': '/unused/1234567_prof.nc', 'source_file': 'dac/1234567/1234567_prof.nc',
                 'index_file': 'dac/1234567/profiles/D1234567_004.nc'}]

    def test_primary_need_not_be_first(self):
        indices, rejected = mixing._primary_profile_indices(
            self.raw([b'Near-surface sampling: discrete', b'Primary sampling: averaged']), self.records())
        self.assertEqual(indices, [1])
        self.assertFalse(rejected)

    def test_ambiguous_primary_fails(self):
        with self.assertRaises(ValueError):
            mixing._primary_profile_indices(self.raw([b'Primary sampling:', b'Primary sampling:']), self.records())

    def test_secondary_only_is_excluded(self):
        indices, rejected = mixing._primary_profile_indices(self.raw([b'Secondary sampling:']), self.records())
        self.assertFalse(indices)
        self.assertEqual(rejected[0]['reason'], 'only_secondary_sampling_available')

    def test_unknown_sole_sampling_and_missing_identity(self):
        self.assertEqual(mixing._primary_profile_indices(self.raw([b'']), self.records())[0], [0])
        ds = self.raw([b''])
        ds['CYCLE_NUMBER'] = ('N_PROF', [5])
        with self.assertRaises(ValueError):
            mixing._primary_profile_indices(ds, self.records())

    def test_cf_decoder_preserves_fractional_seconds(self):
        variable = xr.DataArray([0.25, np.nan], dims='N_PROF',
                                attrs={'units': 'seconds since 2020-01-01 00:00:00'})
        decoded = mixing._decode_cf_time(variable)
        self.assertEqual(decoded[0], np.datetime64('2020-01-01T00:00:00.250000000'))
        self.assertTrue(np.isnat(decoded[1]))

    def test_tie_order_is_independent_of_batch_order(self):
        ds = xr.Dataset({'platform_number': ('time', ['2000000', '1000000']),
                         'cycle_number': ('time', ['4', '3']), 'direction': ('time', ['A', 'A'])},
                        coords={'time': np.array(['2020-01-01'] * 2, dtype='datetime64[ns]')})
        a = mixing._canonical_profile_order(ds)
        b = mixing._canonical_profile_order(ds.isel(time=[1, 0]))
        xr.testing.assert_identical(a, b)
        self.assertEqual(a.platform_number.values.tolist(), ['1000000', '2000000'])


if __name__ == '__main__':
    unittest.main()
