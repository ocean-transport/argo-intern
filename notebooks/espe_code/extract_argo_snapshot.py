#!/usr/bin/env python3
"""Extract a verified monthly Argo DOI archive into a fresh core GDAC mirror.

The source archive and existing mirrors are read only. Extraction is resumable,
uses atomic individual-file publication, and marks the snapshot complete only
after every path in the core profile index exists as a regular file.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys
import tarfile
import tempfile
import threading
import time
import traceback


FORMAT = 'argo-core-doi-snapshot-v1'
BUFFER = 8 * 1024 * 1024
CORE_NAME = re.compile(r'([a-z][a-z0-9_]*)_core\.tar\.gz\Z')
INDEX_NAME = re.compile(r'ar_index_global[a-z_]*\.txt\.gz\Z')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.json-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_signature(path):
    item = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=item.st_size, mtime_ns=item.st_mtime_ns)


class Progress:
    def __init__(self):
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.values = dict(stage='initializing', files_written=0, files_reused=0,
                           bytes_written=0, verified_index_paths=0, active_dacs=[])
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def update(self, **values):
        with self.lock:
            self.values.update(values)

    def add(self, **values):
        with self.lock:
            for key, value in values.items():
                self.values[key] = self.values.get(key, 0) + value

    def dac(self, name, add):
        with self.lock:
            names = set(self.values['active_dacs'])
            names.add(name) if add else names.discard(name)
            self.values['active_dacs'] = sorted(names)

    def emit(self, event='progress', **extra):
        with self.lock:
            value = dict(self.values)
        print(json.dumps(dict(event=event, utc=utcnow(), elapsed_seconds=round(time.monotonic() - self.started, 1),
                              **value, **extra), allow_nan=False), flush=True)

    def _heartbeat(self):
        while not self.stop.wait(30):
            self.emit()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


_WORKER_CANCEL = None
_WORKER_QUEUE = None


def initialize_worker(cancel, progress_queue):
    global _WORKER_CANCEL, _WORKER_QUEUE
    _WORKER_CANCEL, _WORKER_QUEUE = cancel, progress_queue
    # The parent handles Ctrl-C and requests orderly per-file cancellation.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class WorkerProgress(Progress):
    def __init__(self, dac):
        super().__init__()
        self.dac_name = dac
        self.worker_status = 'running'

    def emit(self, event='progress', **extra):
        with self.lock:
            values = dict(self.values)
        _WORKER_QUEUE.put(dict(dac=self.dac_name, event=event, status=self.worker_status,
                               elapsed_seconds=round(time.monotonic() - self.started, 1),
                               values=values, extra=extra))


def process_dac_worker(args, item, staging, metadata):
    with WorkerProgress(item['dac']) as progress:
        progress.update(stage='extracting_core_dacs')
        progress.emit('worker_started')
        try:
            result = extract_dac(args, item, staging, metadata, progress, _WORKER_CANCEL)
        except BaseException:
            progress.worker_status = 'failed'
            progress.emit('worker_failed')
            raise
        progress.worker_status = 'complete'
        progress.emit('worker_finished')
        return result


class WorkerProgressCollector:
    """Aggregate worker counters in the parent; no per-file IPC is required."""
    def __init__(self, context, progress, cancel):
        self.queue = context.Queue()
        self.progress = progress
        self.cancel = cancel
        self.error = None
        self.latest = {}
        with progress.lock:
            self.baseline = dict(progress.values)
        self.thread = threading.Thread(target=self._collect, daemon=True)

    def _collect(self):
        try:
            while True:
                message = self.queue.get()
                if message is None:
                    break
                self.latest[message['dac']] = message
                counters = {name: self.baseline.get(name, 0) + sum(value['values'].get(name, 0) for value in self.latest.values())
                            for name in ['files_written', 'files_reused', 'bytes_written']}
                active = sorted(name for name, value in self.latest.items()
                                if value['status'] == 'running' and value['event'] != 'dac_complete')
                self.progress.update(**counters, active_dacs=active)
                self.progress.emit('worker_progress', worker_dac=message['dac'],
                                   worker_event=message['event'], worker_status=message['status'],
                                   worker_elapsed_seconds=message['elapsed_seconds'],
                                   worker_files_written=message['values']['files_written'],
                                   worker_files_reused=message['values']['files_reused'])
        except BaseException:
            self.error = traceback.format_exc()
            self.cancel.set()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        # Pool shutdown precedes this call, so workers have flushed their queues.
        self.queue.put(None)
        self.thread.join()
        self.queue.close()
        self.queue.join_thread()
        if self.error is not None:
            raise RuntimeError('Worker progress collection failed: ' + self.error)


def safe_parts(name):
    if not isinstance(name, str) or '\\' in name or '\x00' in name:
        raise ValueError(f'Unsafe archive/index path: {name!r}')
    value = PurePosixPath(name)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError(f'Unsafe archive/index path: {name!r}')
    return value.parts


def validate_member(member):
    parts = safe_parts(member.name)
    if not (member.isdir() or member.isfile()):
        raise ValueError(f'Refusing link or special archive member: {member.name!r} (type {member.type!r})')
    if member.size < 0:
        raise ValueError(f'Negative member size: {member.name!r}')
    return parts


def ensure_directory(path, root):
    """Create parents while refusing symlink traversal within the owned root."""
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            if not stat.S_ISDIR(current.lstat().st_mode):
                raise ValueError(f'Non-directory or symlink in destination path: {current}')


def reusable_file(path, size, mtime_ns=None):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Refusing existing non-regular destination: {path}')
    if info.st_size != size or (mtime_ns is not None and info.st_mtime_ns != mtime_ns):
        raise ValueError(f'Existing destination differs from archived size/mtime; refusing overwrite: {path}')
    return True


def write_stream(stream, target, size, mtime_ns, root, partial_dir, progress, digest=False, directory_cache=None):
    if reusable_file(target, size, mtime_ns):
        progress.add(files_reused=1)
        return dict(reused=True, bytes=size)
    for directory in (target.parent, partial_dir):
        if directory_cache is None or directory not in directory_cache:
            ensure_directory(directory, root)
            if directory_cache is not None:
                directory_cache.add(directory)
    fd, temporary = tempfile.mkstemp(prefix='.argo-extract-', dir=partial_dir)
    total = 0
    hasher = hashlib.sha256() if digest else None
    try:
        with os.fdopen(fd, 'wb') as output:
            while True:
                chunk = stream.read(BUFFER)
                if not chunk:
                    break
                output.write(chunk)
                total += len(chunk)
                progress.add(bytes_written=len(chunk))
                if hasher is not None:
                    hasher.update(chunk)
            output.flush()
            # Large staged archives/indexes are durable before their markers.
            # Individual profile files are atomically published without millions
            # of per-file fsync calls; this protects against process interruption.
            if digest:
                os.fsync(output.fileno())
        if total != size:
            raise ValueError(f'Incomplete member {target}: wrote {total}, expected {size}')
        os.chmod(temporary, 0o644)
        os.utime(temporary, ns=(mtime_ns, mtime_ns))
        # link + unlink publishes without replacing any preexisting destination.
        os.link(temporary, target)
        progress.add(files_written=1)
        result = dict(reused=False, bytes=total)
        if hasher is not None:
            result['sha256'] = hasher.hexdigest()
        return result
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def hash_file(path, progress=None):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            data = stream.read(BUFFER)
            if not data:
                break
            digest.update(data)
            if progress:
                progress.add(hash_bytes_read=len(data))
    return digest.hexdigest()


def verified_archive(args, metadata, progress):
    signature = file_signature(args.archive)
    candidates = [args.verification_report] if args.verification_report else [Path(str(args.archive) + '.verified.json')]
    candidates.append(metadata / 'archive_verification.json')
    for path in candidates:
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        if (report.get('status') == 'complete' and report.get('sha256_verified') is True and
                report.get('sha256') == args.archive_sha256 and
                all(report.get(key) == value for key, value in signature.items())):
            verified = dict(report, accepted_verification_report=str(path.resolve()),
                            verification_method='Previously verified SHA256 with matching resolved path, size, and mtime_ns')
            json_write(metadata / 'archive_verification.json', verified)
            return signature
    progress.update(stage='verifying_archive_sha256')
    progress.emit('stage_started')
    actual = hash_file(args.archive, progress)
    if actual != args.archive_sha256:
        raise ValueError(f'Archive SHA256 mismatch: expected {args.archive_sha256}, got {actual}')
    if file_signature(args.archive) != signature:
        raise ValueError('Source archive changed during checksum verification')
    json_write(metadata / 'archive_verification.json', dict(signature, status='complete', sha256_verified=True,
               sha256=actual, verified_utc=utcnow(), verification_method='SHA256 of complete archive'))
    return signature


def outer_selected(relative):
    if len(relative) == 2 and relative[0] == 'dac' and CORE_NAME.fullmatch(relative[1]):
        return 'core'
    if ((len(relative) == 1 or (len(relative) == 2 and relative[0] == 'dac'))
            and INDEX_NAME.fullmatch(relative[-1])):
        return 'index'
    if len(relative) == 1 and relative[0].lower().startswith(('readme', 'read_me', 'license', 'citation')):
        return 'documentation'
    if relative and relative[0].lower() in ('doc', 'docs', 'documentation'):
        return 'documentation'
    return None


def stage_outer(args, metadata, staging, progress):
    manifest_path = metadata / 'outer_members.json'
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if prior.get('archive_sha256') not in (None, args.archive_sha256):
        raise ValueError('Outer-stage manifest refers to another archive')
    if prior.get('status') == 'complete':
        for member in prior['members']:
            if not reusable_file(staging / member['relative_path'], member['bytes'], member['mtime_ns']):
                raise ValueError('Completed staged member is missing; explicit repair required')
        return prior
    progress.update(stage='staging_outer_members')
    progress.emit('stage_started')
    previous = {item['relative_path']: item for item in prior.get('members', [])}
    result = dict(status='incomplete', archive_sha256=args.archive_sha256, members=[])
    seen = set()
    index_names = set()
    outer_root = None
    # Streaming avoids building an archive-wide list or extracting geo/BGC data.
    with tarfile.open(args.archive, mode='r|gz') as archive:
        for member in archive:
            # TarFile caches TarInfo objects even in streaming mode on older
            # Python releases; retain no archive-wide member inventory in RAM.
            archive.members.clear()
            parts = validate_member(member)
            if not parts:
                continue
            if outer_root is None:
                outer_root = parts[0]
                if not re.fullmatch(r'\d{6}-ArgoData', outer_root):
                    raise ValueError(f'Unexpected monthly archive root: {outer_root!r}')
                result['outer_root'] = outer_root
            if parts[0] != outer_root:
                raise ValueError(f'Multiple roots in monthly archive: {member.name!r}')
            relative = parts[1:]
            kind = outer_selected(relative)
            if member.isdir() or kind is None:
                continue
            key = '/'.join(relative)
            if key in seen:
                raise ValueError(f'Duplicate selected outer member: {key}')
            seen.add(key)
            if kind == 'index':
                if relative[-1] in index_names:
                    raise ValueError(f'Ambiguous duplicate global index basename: {relative[-1]}')
                index_names.add(relative[-1])
            target = staging.joinpath(*relative)
            mtime_ns = int(round(member.mtime * 1_000_000_000))
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f'Unable to read {member.name}')
            with stream:
                written = write_stream(stream, target, member.size, mtime_ns, args.output,
                                       args.output / '.snapshot_staging/partials/outer', progress, digest=True)
            digest = written.get('sha256')
            if digest is None:
                prior_item = previous.get(key, {})
                if prior_item.get('bytes') == member.size and prior_item.get('mtime_ns') == mtime_ns:
                    digest = prior_item.get('sha256')
                if digest is None:
                    digest = hash_file(target)
            item = dict(relative_path=key, kind=kind, bytes=member.size, mtime_ns=mtime_ns, sha256=digest)
            if kind == 'core':
                item['dac'] = CORE_NAME.fullmatch(relative[-1])[1]
            result['members'].append(item)
            json_write(manifest_path, result)
            progress.emit('outer_member_staged', member=key)
    if not any(m['kind'] == 'index' and Path(m['relative_path']).name == 'ar_index_global_prof.txt.gz' for m in result['members']):
        raise ValueError('Monthly archive has no ar_index_global_prof.txt.gz at root or within dac/')
    if not any(m['kind'] == 'core' for m in result['members']):
        raise ValueError('Monthly archive has no DAC core archives')
    result.update(status='complete', completed_utc=utcnow())
    json_write(manifest_path, result)
    return result


def extract_indexes_and_docs(args, members, staging, metadata, progress):
    progress.update(stage='extracting_indexes_and_documentation')
    for item in members:
        source = staging / item['relative_path']
        if item['kind'] == 'index':
            target = args.output / Path(item['relative_path']).name.removesuffix('.gz')
            marker = metadata / 'indexes' / (target.name + '.json')
            prior = json.loads(marker.read_text()) if marker.exists() else None
            if prior and prior.get('compressed_sha256') == item['sha256'] and reusable_file(target, prior['bytes'], item['mtime_ns']):
                continue
            # Count decompressed bytes in the temporary file; gzip CRC validates
            # the complete stream. Publish only after successful decompression.
            ensure_directory(args.output / '.snapshot_staging/partials/indexes', args.output)
            fd, temporary = tempfile.mkstemp(prefix='.argo-extract-', dir=args.output / '.snapshot_staging/partials/indexes')
            try:
                digest = hashlib.sha256()
                size = 0
                with os.fdopen(fd, 'wb') as output, gzip.open(source, 'rb') as input_stream:
                    while True:
                        chunk = input_stream.read(BUFFER)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                        progress.add(bytes_written=len(chunk))
                    output.flush()
                    os.fsync(output.fileno())
                os.chmod(temporary, 0o644)
                os.utime(temporary, ns=(item['mtime_ns'], item['mtime_ns']))
                if reusable_file(target, size, item['mtime_ns']):
                    if hash_file(target) != digest.hexdigest():
                        raise ValueError(f'Existing decompressed index differs: {target}')
                else:
                    os.link(temporary, target)
                json_write(marker, dict(status='complete', bytes=size, sha256=digest.hexdigest(),
                                        compressed_sha256=item['sha256'], path=str(target)))
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        elif item['kind'] == 'documentation':
            target = metadata / 'archive_documentation' / item['relative_path']
            with source.open('rb') as stream:
                write_stream(stream, target, item['bytes'], item['mtime_ns'], args.output,
                             args.output / '.snapshot_staging/partials/docs', progress)


def extract_dac(args, item, staging, metadata, progress, stop):
    dac = item['dac']
    marker = metadata / 'dacs' / (dac + '.json')
    expected = dict(archive_sha256=args.archive_sha256, core_archive_sha256=item['sha256'], dac=dac)
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    if previous.get('status') == 'complete' and all(previous.get(k) == v for k, v in expected.items()):
        return previous
    state = dict(expected, status='incomplete', files=0, directories=0, logical_bytes=0, started_utc=utcnow())
    json_write(marker, state)
    progress.dac(dac, True)
    seen = set()
    directory_cache = set()
    try:
        with tarfile.open(staging / item['relative_path'], mode='r|gz') as archive:
            for member in archive:
                archive.members.clear()
                if stop.is_set():
                    raise RuntimeError('Extraction stopped after another DAC failed')
                parts = validate_member(member)
                if not parts:
                    continue
                if parts[0] != dac:
                    raise ValueError(f'DAC archive {dac} contains unexpected path: {member.name!r}')
                if member.isdir():
                    ensure_directory(args.output / 'dac' / Path(*parts), args.output)
                    state['directories'] += 1
                    continue
                key = '/'.join(parts)
                if key in seen:
                    raise ValueError(f'Duplicate member in {dac} archive: {key}')
                seen.add(key)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f'Unable to read {member.name}')
                with stream:
                    write_stream(stream, args.output / 'dac' / Path(*parts), member.size,
                                 int(round(member.mtime * 1_000_000_000)), args.output,
                                 args.output / '.snapshot_staging/partials' / dac, progress,
                                 directory_cache=directory_cache)
                state['files'] += 1
                state['logical_bytes'] += member.size
        state.update(status='complete', completed_utc=utcnow())
        json_write(marker, state)
        progress.emit('dac_complete', dac=dac, dac_files=state['files'], dac_bytes=state['logical_bytes'])
        return state
    except BaseException:
        stop.set()
        state.update(status='failed', failed_utc=utcnow(), error=traceback.format_exc())
        json_write(marker, state)
        raise
    finally:
        progress.dac(dac, False)


def verify_core_index(args, metadata, progress):
    progress.update(stage='verifying_index_paths')
    progress.emit('stage_started')
    index = args.output / 'ar_index_global_prof.txt'
    result = dict(status='incomplete', index=str(index), index_sha256=hash_file(index),
                  index_records=0, existing_files=0, total_logical_bytes=0,
                  missing_files=0, missing_examples=[], missing_by_dac={}, years={})
    checked_directories = set()
    with index.open() as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith('#'))
        if 'file' not in (rows.fieldnames or []):
            raise ValueError('Global profile index lacks file column')
        for row in rows:
            parts = safe_parts(row['file'])
            if len(parts) < 3:
                raise ValueError(f'Invalid indexed relative path: {row["file"]!r}')
            path = args.output / 'dac' / Path(*parts)
            for parent in reversed(path.parents):
                if parent == args.output or args.output not in parent.parents:
                    continue
                if parent not in checked_directories:
                    try:
                        if not stat.S_ISDIR(parent.lstat().st_mode):
                            raise ValueError(f'Non-directory/symlink in indexed path: {parent}')
                    except FileNotFoundError:
                        break
                    checked_directories.add(parent)
            result['index_records'] += 1
            year = row.get('date', '')[:4]
            result['years'][year] = result['years'].get(year, 0) + 1
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(f'Indexed path is not a regular file: {path}')
                result['existing_files'] += 1
                result['total_logical_bytes'] += info.st_size
            except FileNotFoundError:
                result['missing_files'] += 1
                result['missing_by_dac'][parts[0]] = result['missing_by_dac'].get(parts[0], 0) + 1
                if len(result['missing_examples']) < 50:
                    result['missing_examples'].append(row['file'])
            if result['index_records'] % 10000 == 0:
                progress.update(verified_index_paths=result['index_records'])
    progress.update(verified_index_paths=result['index_records'])
    if result['index_records'] == 0:
        raise ValueError('Global profile index contains no records')
    result.update(status='complete' if result['missing_files'] == 0 else 'failed', completed_utc=utcnow())
    json_write(metadata / 'core_index_verification.json', result)
    if result['missing_files']:
        # A resumed run reopens affected DACs rather than trusting stale markers.
        for dac in result['missing_by_dac']:
            marker = metadata / 'dacs' / (dac + '.json')
            if marker.is_file():
                value = json.loads(marker.read_text())
                value.update(status='incomplete', reason='Index verification found missing paths')
                json_write(marker, value)
        raise ValueError(f'{result["missing_files"]} core-index paths are missing')
    return result


def run(args, progress):
    output_preexisted = args.output.exists()
    if output_preexisted and not args.output.is_dir():
        raise ValueError('Output is not a directory')
    if not output_preexisted:
        args.output.mkdir(parents=True)
    manifest_path = args.output / 'snapshot_manifest.json'
    entries = list(args.output.iterdir())
    only_uninitialized_lock = entries and all(path.name == '.snapshot.lock' and stat.S_ISREG(path.lstat().st_mode) for path in entries)
    if entries and not manifest_path.is_file() and not only_uninitialized_lock:
        raise ValueError('Refusing nonempty output directory without this extractor\'s snapshot manifest')
    if manifest_path.exists() and not stat.S_ISREG(manifest_path.lstat().st_mode):
        raise ValueError('Snapshot manifest must be a regular file, not a symlink')
    metadata = args.output / '.snapshot_metadata'
    staging = args.output / '.snapshot_staging/outer'
    if manifest_path.exists():
        state = json.loads(manifest_path.read_text())
        if state.get('format') != FORMAT or state.get('archive_sha256') != args.archive_sha256:
            raise ValueError('Existing output belongs to another snapshot/extractor format')
    else:
        state = dict(format=FORMAT, archive_sha256=args.archive_sha256, archive=str(args.archive),
                     output=str(args.output), created_utc=utcnow(), status='incomplete', scope='core GDAC; BGC and geo retained only in original DOI archive')
    lock_path = args.output / '.snapshot.lock'
    if lock_path.exists() and not stat.S_ISREG(lock_path.lstat().st_mode):
        raise ValueError('Snapshot lock must be a regular file, not a symlink')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state.update(status='incomplete', resumed_utc=utcnow(), extractor_sha256=hash_file(Path(__file__)), workers=args.workers)
        json_write(manifest_path, state)
        try:
            ensure_directory(metadata, args.output)
            ensure_directory(staging, args.output)
            # Only extractor-owned temporary names are removed on resume.
            # There are at most a few per interrupted worker, not millions.
            partials = args.output / '.snapshot_staging/partials'
            if partials.exists():
                if not stat.S_ISDIR(partials.lstat().st_mode):
                    raise ValueError('Extraction partials path is not a directory')
                for directory, dirs, files in os.walk(partials, followlinks=False):
                    for name in dirs:
                        if not stat.S_ISDIR((Path(directory) / name).lstat().st_mode):
                            raise ValueError('Symlink/non-directory in extraction partials')
                    for name in files:
                        path = Path(directory) / name
                        if not stat.S_ISREG(path.lstat().st_mode):
                            raise ValueError('Non-regular file in extraction partials')
                        if name.startswith('.argo-extract-'):
                            path.unlink()
            signature = verified_archive(args, metadata, progress)
            state['archive_signature'] = signature
            json_write(manifest_path, state)
            outer = stage_outer(args, metadata, staging, progress)
            extract_indexes_and_docs(args, outer['members'], staging, metadata, progress)
            cores = sorted((item for item in outer['members'] if item['kind'] == 'core'),
                           key=lambda item: item['bytes'], reverse=True)
            if len({item['dac'] for item in cores}) != len(cores):
                raise ValueError('Duplicate DAC core archives in outer archive')
            progress.update(stage='extracting_core_dacs')
            progress.emit('stage_started')
            context = multiprocessing.get_context('spawn')
            stop = context.Event()
            dacs = []
            with WorkerProgressCollector(context, progress, stop) as collector:
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                         initializer=initialize_worker,
                                         initargs=(stop, collector.queue)) as pool:
                    futures = []
                    try:
                        for item in cores:
                            futures.append(pool.submit(process_dac_worker, args, item, staging, metadata))
                        for future in as_completed(futures):
                            dacs.append(future.result())
                    except BaseException:
                        stop.set()
                        for future in futures:
                            future.cancel()
                        raise
            verification = verify_core_index(args, metadata, progress)
            if file_signature(args.archive) != signature:
                raise ValueError('Source DOI archive changed during extraction')
            state.update(status='complete', completed_utc=utcnow(), dacs=sorted(dacs, key=lambda item: item['dac']),
                         index_records=verification['index_records'], indexed_logical_bytes=verification['total_logical_bytes'],
                         verification_report=str(metadata / 'core_index_verification.json'),
                         elapsed_seconds=time.monotonic() - progress.started)
            json_write(manifest_path, state)
            progress.update(stage='complete')
            progress.emit('snapshot_complete', index_records=state['index_records'])
            return state
        except BaseException:
            state.update(status='failed', failed_utc=utcnow(), error=traceback.format_exc(),
                         elapsed_seconds=time.monotonic() - progress.started)
            json_write(manifest_path, state)
            progress.update(stage='failed')
            progress.emit('snapshot_failed')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--verification-report', type=Path, help='Optional trusted completed download verification JSON; defaults to <archive>.verified.json')
    parser.add_argument('--workers', type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    args.archive = args.archive.resolve(strict=True)
    args.output = args.output.resolve()
    args.archive_sha256 = args.archive_sha256.lower()
    if not re.fullmatch(r'[0-9a-f]{64}', args.archive_sha256):
        parser.error('--archive-sha256 must be a complete SHA256 hexadecimal digest')
    allowed = Path('/swot/SUM05/amf2288').resolve()
    if args.output == allowed or not args.output.is_relative_to(allowed):
        parser.error('Snapshot output must be a new subdirectory under /swot/SUM05/amf2288')
    if args.archive.is_relative_to(args.output):
        parser.error('Input archive must remain outside the extracted snapshot directory')
    try:
        with Progress() as progress:
            run(args, progress)
    except Exception as error:
        print(f'Extraction failed: {error}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
