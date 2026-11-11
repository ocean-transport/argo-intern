#!/usr/bin/env python3
"""Download an immutable Argo archive with bounded, resumable HTTP ranges.

The final filename appears only after a full SHA256 check. A JSON manifest and
exclusive advisory lock accompany the partial file; completed ranges are hashed
again before they are reused. Scientific processing and extraction are separate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
import urllib.request


ALLOWED_ROOT = Path('/swot/SUM05/amf2288')
BLOCK_BYTES = 1024 * 1024
PROGRESS_SECONDS = 30
MAX_ATTEMPTS = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verification_report(output: Path, digest: str) -> None:
    stat = output.stat()
    atomic_json(Path(str(output) + '.verified.json'), {
        'status': 'complete', 'sha256_verified': True, 'sha256': digest,
        'path': str(output.resolve()), 'bytes': stat.st_size,
        'mtime_ns': stat.st_mtime_ns, 'verified_utc': utc_now(),
    })


def hash_range(fd: int, start: int, length: int) -> str:
    digest = hashlib.sha256()
    offset = start
    while offset < start + length:
        block = os.pread(fd, min(BLOCK_BYTES, start + length - offset), offset)
        if not block:
            raise OSError(f'Unexpected EOF while hashing offset {offset}')
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def hash_file(path: Path, log: logging.Logger) -> str:
    digest = hashlib.sha256()
    count = 0
    last_report = time.monotonic()
    with path.open('rb') as stream:
        while block := stream.read(8 * BLOCK_BYTES):
            digest.update(block)
            count += len(block)
            if time.monotonic() - last_report >= PROGRESS_SECONDS:
                log.info('Full SHA256 verification: %.2f GiB read', count / 1024**3)
                last_report = time.monotonic()
    return digest.hexdigest()


def write_at(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    while view:
        count = os.pwrite(fd, view, offset)
        if count <= 0:
            raise OSError('os.pwrite made no progress')
        view = view[count:]
        offset += count


def fetch_chunk(url: str, total: int, fd: int, index: int, chunk_bytes: int,
                stop: threading.Event, log: logging.Logger) -> dict:
    start = index * chunk_bytes
    end = min(total, start + chunk_bytes) - 1
    length = end - start + 1
    expected_range = f'bytes {start}-{end}/{total}'
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if stop.is_set():
            raise InterruptedError('Download interrupted')
        try:
            request = urllib.request.Request(url, headers={
                'Range': f'bytes={start}-{end}',
                'Accept-Encoding': 'identity',
                'User-Agent': 'ArgoSnapshotDownloader/1.0',
            })
            digest = hashlib.sha256()
            received = 0
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 206:
                    raise ValueError(f'Expected HTTP 206, received {response.status}')
                if response.headers.get('Content-Range') != expected_range:
                    raise ValueError(f'Unexpected Content-Range: {response.headers.get("Content-Range")!r}')
                encoding = response.headers.get('Content-Encoding', 'identity')
                if encoding.lower() != 'identity':
                    raise ValueError(f'Unexpected Content-Encoding: {encoding!r}')
                content_length = response.headers.get('Content-Length')
                if content_length is not None and int(content_length) != length:
                    raise ValueError(f'Wrong Content-Length: {content_length}, expected {length}')
                while received < length:
                    if stop.is_set():
                        raise InterruptedError('Download interrupted')
                    block = response.read(min(BLOCK_BYTES, length - received))
                    if not block:
                        raise OSError(f'Short response: {received}/{length} bytes')
                    write_at(fd, block, start + received)
                    digest.update(block)
                    received += len(block)
                if response.read(1):
                    raise ValueError('Range response contains extra bytes')
            return {'start': start, 'bytes': length, 'sha256': digest.hexdigest()}
        except Exception as error:
            if stop.is_set() or attempt == MAX_ATTEMPTS:
                raise
            log.warning('Chunk %d attempt %d/%d failed: %s', index, attempt, MAX_ATTEMPTS, error)
            if stop.wait(2 ** (attempt - 1)):
                raise InterruptedError('Download interrupted') from error
    raise AssertionError('Unreachable')


def guarded_output(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    root = ALLOWED_ROOT.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Output must be a file beneath {root}')
    for suffix in ['', '.partial', '.download.json', '.download.json.tmp', '.lock',
                   '.download.log', '.verified.json', '.verified.json.tmp']:
        candidate = Path(str(path) + suffix)
        if candidate.is_symlink():
            raise ValueError(f'Refusing symlink output: {candidate}')
    return path


def download(args: argparse.Namespace) -> None:
    output = guarded_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + '.partial')
    manifest_path = Path(str(output) + '.download.json')
    lock_path = Path(str(output) + '.lock')
    log = logging.getLogger('argo_snapshot')
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    log.addHandler(console)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another downloader holds {lock_path}') from error
        lock.seek(0)
        lock.truncate()
        lock.write(f'{os.getpid()}\n')
        lock.flush()
        file_handler = logging.FileHandler(str(output) + '.download.log')
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        try:
            _download_locked(args, output, partial, manifest_path, log)
        finally:
            log.removeHandler(file_handler)
            file_handler.close()


def _download_locked(args: argparse.Namespace, output: Path, partial: Path,
                     manifest_path: Path, log: logging.Logger) -> None:
    chunk_bytes = args.chunk_mib * 1024**2
    identity = {'url': args.url, 'bytes': args.bytes, 'sha256': args.sha256.lower(),
                'chunk_bytes': chunk_bytes}
    chunks = (args.bytes + chunk_bytes - 1) // chunk_bytes
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema_version') != 1 or manifest.get('source') != identity:
            raise ValueError('Existing manifest does not match URL, bytes, SHA256, and chunk size')
    else:
        if partial.exists():
            raise ValueError('Partial file exists without its manifest; refusing to reuse it')
        manifest = {'schema_version': 1, 'source': identity, 'created_utc': utc_now(),
                    'completed_chunks': {}, 'status': 'initializing'}

    def save(status: str) -> None:
        manifest['status'] = status
        manifest['updated_utc'] = utc_now()
        manifest['completed_bytes'] = sum(x['bytes'] for x in manifest['completed_chunks'].values())
        atomic_json(manifest_path, manifest)

    if output.exists():
        if output.stat().st_size != args.bytes or hash_file(output, log) != identity['sha256']:
            raise ValueError('Existing final output fails expected size/SHA256; refusing to overwrite it')
        manifest['verified_sha256'] = identity['sha256']
        manifest['verified_utc'] = utc_now()
        manifest['final_bytes'] = args.bytes
        save('complete')
        verification_report(output, identity['sha256'])
        log.info('Existing final file verified: %s', output)
        return
    if manifest.get('status') == 'complete':
        raise ValueError('Manifest says complete but final output is missing')

    if partial.exists():
        fd = os.open(partial, os.O_RDWR)
        if os.fstat(fd).st_size != args.bytes:
            os.close(fd)
            raise ValueError('Partial file has an unexpected size')
    else:
        if manifest_path.exists():
            raise ValueError('Manifest exists but partial output is missing')
        fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
        try:
            os.ftruncate(fd, args.bytes)
        except BaseException:
            os.close(fd)
            raise
    stop = threading.Event()
    executor = None
    try:
        save('checking_resume')
        last_report = time.monotonic()
        for key, record in list(manifest['completed_chunks'].items()):
            index = int(key)
            start = index * chunk_bytes
            length = min(chunk_bytes, args.bytes - start)
            if str(index) != key or not 0 <= index < chunks or record.get('start') != start or record.get('bytes') != length:
                raise ValueError(f'Invalid completed-chunk record: {key}')
            actual = hash_range(fd, start, length)
            if actual != record.get('sha256'):
                log.warning('Chunk %d failed resume verification; scheduling replacement', index)
                del manifest['completed_chunks'][key]
            if time.monotonic() - last_report >= PROGRESS_SECONDS:
                log.info('Resume verification: checked through chunk %d/%d', index + 1, chunks)
                last_report = time.monotonic()
        save('downloading')
        baseline = manifest['completed_bytes']
        started = time.monotonic()
        log.info('Starting/resuming %s: %d/%d chunks verified, %.2f/%.2f GiB, %d workers',
                 output.name, len(manifest['completed_chunks']), chunks,
                 baseline / 1024**3, args.bytes / 1024**3, args.workers)
        remaining = iter(i for i in range(chunks) if str(i) not in manifest['completed_chunks'])
        executor = ThreadPoolExecutor(max_workers=args.workers)
        pending = {}

        def submit_one() -> None:
            index = next(remaining, None)
            if index is not None:
                future = executor.submit(fetch_chunk, args.url, args.bytes, fd,
                                         index, chunk_bytes, stop, log)
                pending[future] = index

        for _ in range(args.workers):
            submit_one()
        last_report = time.monotonic()
        while pending:
            done, _ = wait(pending, timeout=PROGRESS_SECONDS, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                record = future.result()
                os.fsync(fd)
                manifest['completed_chunks'][str(index)] = record
                save('downloading')
                submit_one()
            now = time.monotonic()
            if now - last_report >= PROGRESS_SECONDS or not pending:
                count = manifest['completed_bytes']
                rate = (count - baseline) / max(now - started, 0.001) / 1024**2
                log.info('Progress: %d/%d chunks, %.2f/%.2f GiB (%.2f%%), %.2f MiB/s of completed chunks',
                         len(manifest['completed_chunks']), chunks, count / 1024**3,
                         args.bytes / 1024**3, 100 * count / args.bytes, rate)
                last_report = now
        executor.shutdown(wait=True)
        executor = None
        os.fsync(fd)
        if len(manifest['completed_chunks']) != chunks or manifest['completed_bytes'] != args.bytes:
            raise RuntimeError('Incomplete chunk accounting; refusing to finalize')
        save('verifying')
        log.info('All chunks present; checking full-file SHA256')
        actual = hash_file(partial, log)
        if actual != identity['sha256']:
            manifest['actual_sha256'] = actual
            raise ValueError(f'Full SHA256 mismatch: expected {identity["sha256"]}, received {actual}')
        manifest['verified_sha256'] = actual
        manifest['verified_utc'] = utc_now()
        manifest['final_bytes'] = args.bytes
        os.replace(partial, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        save('complete')
        verification_report(output, actual)
        log.info('Complete and SHA256 verified: %s', output)
    except BaseException as error:
        stop.set()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        manifest['last_error'] = f'{type(error).__name__}: {error}'
        save('interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed')
        log.error('%s; partial file and manifest retained for resume', manifest['last_error'])
        raise
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--bytes', required=True, type=int)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--chunk-mib', type=int, default=256)
    args = parser.parse_args()
    if not args.url.startswith(('https://', 'http://')):
        parser.error('--url must use HTTP or HTTPS')
    if args.bytes <= 0:
        parser.error('--bytes must be positive')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', args.sha256):
        parser.error('--sha256 must contain exactly 64 hexadecimal characters')
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    if not 1 <= args.chunk_mib <= 1024:
        parser.error('--chunk-mib must be between 1 and 1024')
    download(args)


if __name__ == '__main__':
    main()
