"""Add a validated global Kiso product to the existing binned NetCDF atomically.

Preserves every original variable, dimension and attribute. Keeps independent
backups of the original NetCDF and its JSON report before replacing the file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from netCDF4 import Dataset


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_preserved(original, updated):
    """Compare original raw arrays bitwise, including NaN payloads and metadata."""
    with Dataset(original) as before, Dataset(updated) as after:
        before.set_auto_maskandscale(False)
        after.set_auto_maskandscale(False)
        for name, dim in before.dimensions.items():
            assert name in after.dimensions and len(dim) == len(after.dimensions[name]), name
        for name in before.ncattrs():
            np.testing.assert_equal(before.getncattr(name), after.getncattr(name))
        for name, variable in before.variables.items():
            other = after[name]
            assert variable.dimensions == other.dimensions and variable.dtype == other.dtype, name
            assert set(variable.ncattrs()) == set(other.ncattrs()), name
            for attr in variable.ncattrs():
                np.testing.assert_equal(variable.getncattr(attr), other.getncattr(attr))
            assert np.asarray(variable[:]).tobytes() == np.asarray(other[:]).tobytes(), name
        return len(before.variables)


def install(binned_path, additions_path):
    target, additions = Path(binned_path).resolve(), Path(additions_path).resolve()
    backup = target.with_name(target.stem + '.before_kiso.nc')
    metadata_path = target.with_suffix('.json')
    metadata_backup = target.with_name(target.stem + '.before_kiso.json')
    temporary = target.with_name(target.stem + '.kiso.partial.nc')
    if any(p.exists() for p in (backup, metadata_backup, temporary)):
        raise FileExistsError('Kiso backup or partial file already exists; inspect before reinstalling')
    original_stat = target.stat()
    original_digest, addition_digest = digest(target), digest(additions)
    with Dataset(additions) as extra, Dataset(target) as existing:
        provenance = json.loads(extra.getncattr('isopycnal_report_json'))
        if not provenance.get('full_global_run') or provenance.get('bootstrap_count') != 0:
            raise ValueError('Only a completed global, non-bootstrap product can be installed')
        expected_profiles = int(np.asarray(existing['cell_profile_count'][:]).sum())
        if provenance['profiles_scanned'] != expected_profiles:
            raise ValueError('New analysis does not cover the existing global profile population')
        np.testing.assert_array_equal(extra['iso_cell_profile_count'][:], existing['cell_profile_count'][:])
        shared = set(existing.variables) & set(extra.variables)
        if shared != {'latitude', 'longitude', 'pressure_bin'}:
            raise ValueError(f'Unexpected variable collision: {sorted(shared)}')
        for name in shared:
            np.testing.assert_array_equal(existing[name][:], extra[name][:])
        if any(attr in existing.ncattrs() for attr in extra.ncattrs()):
            raise ValueError('New global attributes would overwrite existing metadata')
        new_names = sorted(set(extra.variables) - shared)
    # Build and verify the complete replacement before changing the original path.
    shutil.copy2(target, temporary)
    with Dataset(temporary, 'r+') as output, Dataset(additions) as extra:
        extra.set_auto_maskandscale(False)
        for name, dim in extra.dimensions.items():
            if name not in output.dimensions:
                output.createDimension(name, len(dim))
            elif len(dim) != len(output.dimensions[name]):
                raise ValueError(f'Dimension mismatch: {name}')
        for name in new_names:
            variable = extra[name]
            options = {'zlib': True, 'complevel': 2}
            if '_FillValue' in variable.ncattrs():
                options['fill_value'] = variable.getncattr('_FillValue')
            else:
                options['fill_value'] = False
            created = output.createVariable(name, variable.dtype, variable.dimensions, **options)
            created.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs() if attr != '_FillValue'})
            created.set_auto_maskandscale(False)
            created[:] = variable[:]
        output.setncatts({attr: extra.getncattr(attr) for attr in extra.ncattrs()})
        output.setncatts({'isopycnal_additions_sha256': addition_digest,
                         'isopycnal_original_binned_sha256': original_digest})
    original_variables = verify_preserved(target, temporary)
    with Dataset(temporary) as updated, Dataset(additions) as extra:
        updated.set_auto_maskandscale(False)
        extra.set_auto_maskandscale(False)
        for name in new_names:
            assert np.asarray(updated[name][:]).tobytes() == np.asarray(extra[name][:]).tobytes(), name
            assert updated[name].dimensions == extra[name].dimensions, name
            for attr in extra[name].ncattrs():
                np.testing.assert_equal(updated[name].getncattr(attr), extra[name].getncattr(attr))
    current_stat = target.stat()
    if (original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns) != (
            current_stat.st_ino, current_stat.st_size, current_stat.st_mtime_ns):
        raise RuntimeError('Original NetCDF changed during validation; refusing replacement')
    report = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    report.update(status='complete', output_file=str(target), file_bytes=temporary.stat().st_size,
                  sha256=digest(temporary), original_product_sha256=original_digest,
                  original_product_backup=str(backup), isopycnal_analysis=provenance,
                  added_variables=new_names,
                  verification=f'{original_variables} original variables, dimensions and attributes preserved; '
                               f'{len(new_names)} new variables verified bitwise against the computed addition file')
    shutil.copy2(target, backup)
    assert digest(backup) == original_digest
    if metadata_path.exists():
        shutil.copy2(metadata_path, metadata_backup)
    os.replace(temporary, target)
    temporary_json = metadata_path.with_suffix('.json.partial')
    temporary_json.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temporary_json, metadata_path)
    print(json.dumps({'output': str(target), 'backup': str(backup),
                      'verification': report['verification'], 'sha256': report['sha256']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binned', required=True)
    parser.add_argument('--additions', required=True)
    arguments = parser.parse_args()
    install(arguments.binned, arguments.additions)
