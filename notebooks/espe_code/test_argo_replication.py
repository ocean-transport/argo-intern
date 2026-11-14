"""Regression checks for local-source selection and strict replication acceptance."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import tempfile

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


class NeutralDensityFailureIsolation(unittest.TestCase):
    def raw(self):
        pressure = np.arange(2., 102., 2.)
        return xr.Dataset({
            'PRES': (('N_PROF', 'N_LEVELS'), np.tile(pressure, (2, 1))),
            'TEMP': (('N_PROF', 'N_LEVELS'), np.tile(20. - pressure * .02, (2, 1))),
            'PSAL': (('N_PROF', 'N_LEVELS'), np.full((2, len(pressure)), 35.)),
            'TIME': ('N_PROF', np.array(['2020-01-01', '2020-01-02'], dtype='datetime64[ns]')),
            'LONGITUDE': ('N_PROF', [-30., -29.]),
            'LATITUDE': ('N_PROF', [30., 30.]),
        }, coords={'N_PROF': [0, 1], 'N_LEVELS': np.arange(len(pressure))})

    def test_gamma_arithmetic_failure_keeps_profiles_and_flags_missing_density(self):
        def gamma(s, t, p, lon, lat):
            if lon == -30.:
                raise ZeroDivisionError('division by zero')
            values = 26. + p * .001
            return values, np.zeros_like(values), np.zeros_like(values)
        with patch.object(mixing.nds, 'gamma_n', side_effect=gamma):
            result = mixing.process_profiles(self.raw(), {'max_depth': 100.})
        self.assertEqual(result.sizes['time'], 2)
        np.testing.assert_array_equal(result.gamma_n_status, [5, 0])
        self.assertTrue(np.isnan(result.gamman[:, 0]).all())
        self.assertTrue(np.isnan(result.N2[:, 0]).all())
        self.assertTrue(np.isfinite(result.CT[:, 0]).any())
        self.assertTrue(np.isfinite(result.gamman[:, 1]).any())
        self.assertEqual(int(result.gamma_n_failed_sample_count[0]),
                         int(np.isfinite(result.SP[:, 0]).sum()))
        self.assertIn('gamma_n ZeroDivisionError', str(result.stage_message.values[0]))
        self.assertEqual(int(result.n2_status[0]), 1)
        self.assertEqual(int(result.polynomial_status[0]), 1)

    def test_unexpected_gamma_errors_still_fail(self):
        with patch.object(mixing.nds, 'gamma_n', side_effect=ValueError('unexpected')):
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                mixing.process_profiles(self.raw(), {'max_depth': 100.})


class ProfilesOnlyWorkflow(unittest.TestCase):
    def test_cli_skips_summaries(self):
        with patch.object(mixing, 'run_pipeline') as run:
            self.assertEqual(mixing.main(['run', '--bounds', '-38', '-20', '19', '32',
                                          '--profiles-only', '--workers', '50']), 0)
        self.assertFalse(run.call_args.kwargs['summaries'])
        self.assertEqual(run.call_args.args[0]['workers'], 50)

    def test_profile_only_manifest_and_no_summary_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'output': directory, 'source': 'local', 'workers': 1}
            result = {'status': 'complete', 'accepted_profiles': 2, 'region': 'region',
                      'path': str(Path(directory) / 'profiles.nc')}
            with patch.object(mixing, '_data_directory', side_effect=Path), \
                    patch.object(mixing, '_source_snapshot', return_value={}), \
                    patch.object(mixing, '_profile_tasks', return_value=[{}]), \
                    patch.object(mixing, '_profile_task', return_value=result), \
                    patch.object(mixing, '_summary_task', side_effect=AssertionError('Summary ran')):
                manifest = mixing.run_pipeline(config, summaries=False)
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['processing_stage'], 'profiles')
            self.assertEqual(manifest['summaries_status'], 'deferred')
            self.assertEqual(manifest['accepted_profiles'], 2)
            self.assertNotIn('summaries', manifest)
            self.assertGreater(manifest['timings_seconds']['profiles_wall'], 0)

    def test_validation_never_bootstraps_even_on_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            for year in range(2014, 2024):
                (Path(directory) / f'Mixing_Argo_NATRE_2m_{year}_resmode.nc').touch()
            args = SimpleNamespace(profiles_only=True, reference_dir=directory)
            config = {'output': directory, 'gdac': '/snapshot', 'summary_config': {}}
            manifest = {'regions': {'region': ['unused.nc']}, 'inventory': {}, 'timings_seconds': {}}
            for passed in [True, False]:
                with self.subTest(passed=passed), \
                        patch.object(mixing, '_data_directory', side_effect=Path), \
                        patch.object(mixing, '_source_snapshot', return_value={}), \
                        patch.object(mixing, 'run_pipeline', return_value=manifest) as run, \
                        patch.object(mixing, '_open_profiles', side_effect=lambda paths: xr.Dataset()), \
                        patch.object(mixing, 'compare_profile_datasets', return_value={'passed': passed}), \
                        patch.object(mixing, 'summary_regression', side_effect=AssertionError('Bootstrap diagnostic ran')), \
                        patch.object(mixing, '_summary_task', side_effect=AssertionError('Summary ran')):
                    report = mixing._validate_natre(config, args)
                self.assertEqual(report['passed'], passed)
                self.assertEqual(report['global_authorized'], passed)
                self.assertEqual(report['validation_scope'], 'profiles')
                self.assertEqual(report['summaries_status'], 'deferred')
                self.assertTrue(all(not call.kwargs['summaries'] for call in run.call_args_list))

    def test_global_gate_enforces_scope_and_code_identity(self):
        config = {'code_sha256': 'same', 'gdac': '/snapshot', 'local_layout': 'aggregate'}
        validation = {'passed': True, 'global_authorized': True, 'script_sha256': 'same',
                      'gdac': '/snapshot', 'local_layout': 'aggregate', 'source_snapshot': {},
                      'comparison_policy': 'bitwise_reference_arrays_with_full_population_and_order',
                      'validation_scope': 'profiles'}
        with patch.object(mixing, '_source_snapshot', return_value={}):
            mixing._require_global_validation(config, validation, profiles_only=True)
            with self.assertRaisesRegex(ValueError, 'only validates profiles'):
                mixing._require_global_validation(config, validation, profiles_only=False)
            for changed in [{'script_sha256': 'different'}, {'source_snapshot': {'changed': True}},
                            {'local_layout': 'mono'}, {'passed': False}]:
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    mixing._require_global_validation(config, dict(validation, **changed), profiles_only=True)
            validation['validation_scope'] = 'profiles_and_summaries'
            mixing._require_global_validation(config, validation, profiles_only=True)
            mixing._require_global_validation(config, validation, profiles_only=False)


if __name__ == '__main__':
    unittest.main()
