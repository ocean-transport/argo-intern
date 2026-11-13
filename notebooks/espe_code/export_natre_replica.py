#!/usr/bin/env python3
"""Export validated LOCAL profile arrays in the historical notebook's schema.

No profile or mixing calculation occurs here. Historical values are comparison
targets only. Variable selection and descriptive metadata follow the reference
schema; local-source provenance and extra diagnostics remain in the run and the
adjacent JSON report. NetCDF container-byte identity is not asserted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import process_argo_mixing as mixing
import xarray as xr


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path,
                        default=Path('/swot/SUM05/amf2288/espe_code/data_files/Mixing_Argo_NATRE_2m_resmode_ALL.nc'))
    args = parser.parse_args()
    run = args.run.resolve()
    validation = json.loads((run / 'validation.json').read_text())
    manifest = json.loads((run / 'local/manifest.json').read_text())
    if not (validation['local_profiles']['passed'] and manifest['status'] == 'complete'
            and manifest['config']['source'] == 'local'
            and manifest['config']['gdac'] == validation['gdac']
            and manifest['config']['code_sha256'] == validation['script_sha256']):
        raise ValueError('Require a completed, bitwise-validated local profile stage')
    directory = mixing._data_directory(run / 'local')
    target = directory / 'NATRE_profiles_replica.nc'
    report_path = target.with_suffix('.json')
    if target.exists() or report_path.exists():
        raise FileExistsError('Refusing to replace an existing replica or report')
    paths = sorted(manifest['regions']['region'])
    with xr.open_dataset(args.reference, engine='netcdf4') as expected:
        names = list(expected.data_vars)
        parts = []
        # Load bounded partitions before global ordering to avoid a large lazy
        # shuffle graph. The final NATRE export is approximately 800 MiB.
        for path in paths:
            with xr.open_dataset(path, engine='netcdf4') as opened:
                parts.append(opened[names + ['platform_number', 'cycle_number', 'direction']].load())
        joined = xr.concat(parts, dim='time', data_vars='minimal', coords='minimal', compat='equals')
        replica = mixing._canonical_profile_order(joined)[names]
        del parts, joined
        for name in expected.variables:
            if set(expected[name].attrs) - {'Description', 'Units'}:
                raise ValueError(f'Review non-descriptive reference attributes before copying: {name}')
            replica[name].attrs = dict(expected[name].attrs)
        if set(expected.attrs) - {'Description'}:
            raise ValueError('Review non-descriptive global reference attributes before copying')
        replica.attrs = dict(expected.attrs)
        before = mixing.compare_profile_datasets(replica, expected)
        if not before['passed'] or before['additional_variables']:
            raise ValueError(f'Local export failed bitwise reference-schema validation: {before}')
        xr.testing.assert_identical(replica, expected)
        mixing._netcdf_write(replica, target)
        with xr.open_dataset(target, engine='netcdf4') as saved:
            after = mixing.compare_profile_datasets(saved, expected)
            if not after['passed']:
                raise ValueError(f'Serialized replica failed read-back validation: {after}')
            xr.testing.assert_identical(saved, expected)
    report = {'passed': True, 'replica': str(target), 'replica_sha256': digest(target),
              'reference': str(args.reference.resolve()), 'reference_sha256': digest(args.reference),
              'validation_report': str(run / 'validation.json'),
              'local_partition_paths': paths, 'gdac': validation['gdac'],
              'producing_script_sha256': validation['script_sha256'],
              'export_script_sha256': digest(__file__),
              'array_origin': 'new local-source profile partitions only',
              'reference_usage': 'schema, descriptive metadata, and comparison targets only; no reference data copied',
              'all_reference_arrays_bitwise': True, 'schema_and_descriptive_attributes_identical': True,
              'serialized_file_bytes_identical': False, 'read_back_comparison': after}
    with report_path.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'passed': True, 'dataset': str(target), 'report': str(report_path)}, indent=2))


if __name__ == '__main__':
    main()
