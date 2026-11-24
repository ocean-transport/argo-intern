"""Installation preserves old arrays/metadata and rejects incomplete products."""
import json
from pathlib import Path
import tempfile
import unittest

from netCDF4 import Dataset
import numpy as np
import xarray as xr

from install_isopycnal_diffusivity import digest, install, verify_preserved


class InstallTest(unittest.TestCase):
    def test_atomic_addition_backup_and_pilot_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, extra = root/'original.nc', root/'extra.nc'
            with Dataset(original, 'w') as d:
                for name, size in [('latitude', 1), ('longitude', 1), ('pressure_bin', 2)]:
                    d.createDimension(name, size)
                    d.createVariable(name, 'f8', (name,))[:] = [25., 75.] if size == 2 else [1.5]
                d.createVariable('cell_profile_count', 'i8', ('latitude', 'longitude'))[:] = [[20]]
                data = d.createVariable('old_metric', 'f8', ('pressure_bin',))
                data[:] = np.array([0x8000000000000000, 0x7ff8000000000001], dtype=np.uint64).view('f8')
                data.units = 'original units'
                d.description = 'original global attributes'
            original.with_suffix('.json').write_text(json.dumps({'sha256': digest(original)}))
            ds = xr.Dataset({
                'KisoT': (('isopycnal_density', 'latitude', 'longitude'), np.full((2, 1, 1), 100.)),
                'KisoT_pressure': (('pressure_bin', 'latitude', 'longitude'), np.full((2, 1, 1), 200.)),
                'iso_cell_profile_count': (('latitude', 'longitude'), [[20]]),
            }, coords={'isopycnal_density': [25., 25.05], 'latitude': [1.5],
                       'longitude': [1.5], 'pressure_bin': [25., 75.]})
            provenance = {'full_global_run': False, 'bootstrap_count': 0, 'profiles_scanned': 20}
            ds.attrs['isopycnal_report_json'] = json.dumps(provenance)
            ds.to_netcdf(extra, engine='netcdf4')
            original_hash = digest(original)
            with self.assertRaisesRegex(ValueError, 'completed global'):
                install(original, extra)
            self.assertEqual(digest(original), original_hash)
            provenance['full_global_run'] = True
            ds.attrs['isopycnal_report_json'] = json.dumps(provenance)
            ds.to_netcdf(extra, engine='netcdf4')
            report = install(original, extra)
            backup = root/'original.before_kiso.nc'
            self.assertEqual(digest(backup), original_hash)
            self.assertEqual(digest(original), report['sha256'])
            verify_preserved(backup, original)
            with Dataset(original) as d:
                np.testing.assert_array_equal(d['KisoT_pressure'][:], np.full((2, 1, 1), 200.))
            with self.assertRaises(FileExistsError):
                install(original, extra)


if __name__ == '__main__':
    unittest.main()
