"""Ordered worker pool for independent, bounded audio windows.

Studio stages read overlapping windows, transform each one on its own and
write the cropped results in order. The transforms never share state, so the
windows can be computed in separate processes without changing a sample: every
window still sees the same input, the same function and the same arithmetic.

Only a few windows are in flight at once, so memory stays bounded for chapter-
and book-length takes. With one worker everything runs inline, exactly as before.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import signal

_pool = None
_workers = 1

# BLAS threads only add overhead to the tiny per-bin solves used here, and they
# would compete with the window workers. Keep the numeric libraries serial.
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


_pinned = []


def serial_numeric_libraries():
    """Call before NumPy is imported; an explicit user setting is respected."""
    for name in THREAD_VARIABLES:
        if name not in os.environ:
            os.environ[name] = '1'
            _pinned.append(name)


def model_environment():
    """Environment for a model process, without the limits pinned above.

    The audio stages want serial numeric libraries; model inference wants its
    own threads. Variables the user set are passed through unchanged.
    """
    return {name: value for name, value in os.environ.items() if name not in _pinned}


def available_cpus():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def default_workers():
    """Leave one logical CPU free on larger machines; at most 8 windows in flight."""
    cpus = available_cpus()
    return min(8, cpus if cpus <= 2 else cpus-1)


def _quiet_worker():
    # The parent owns interruption and cleanup; workers just stop.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for name in ('SIGTERM', 'SIGHUP'):
        signal.signal(getattr(signal, name), signal.SIG_DFL)


def configure(workers=None):
    """Set the number of window workers; ``None`` selects the default."""
    global _workers
    shutdown()
    workers = default_workers() if workers is None else int(workers)
    if workers < 1:
        raise ValueError('Workers must be at least 1')
    _workers = workers
    return _workers


def workers():
    return _workers


def _executor():
    global _pool
    if _pool is None:
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context('fork' if 'fork' in methods else 'spawn')
        _pool = ProcessPoolExecutor(_workers, mp_context=context, initializer=_quiet_worker)
    return _pool


def shutdown():
    global _pool
    if _pool is not None:
        pool, _pool = _pool, None
        for process in list(getattr(pool, '_processes', {}).values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)


def submit(function, *arguments):
    return _executor().submit(function, *arguments)


def imap(function, tasks, enabled=True):
    """Yield ``function(*task)`` for each task, in order.

    ``function`` and its arguments must be picklable when more than one worker
    is configured. At most ``workers + 1`` windows are held at any time.
    ``enabled=False`` keeps a stage inline, for transforms that are closures.
    """
    if _workers <= 1 or not enabled:
        for task in tasks:
            yield function(*task)
        return
    pending = deque()
    pool = _executor()
    for task in tasks:
        pending.append(pool.submit(function, *task))
        if len(pending) > _workers:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()
