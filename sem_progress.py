"""Console percentage progress for SEM workflows, using the standard library.

Percentages count completed work items, not estimated algorithm iterations or
elapsed-time fractions. A terminal refreshes elapsed time during blocking calls.
Piped output/notebooks receive throttled plain-text updates. stderr is used so
CLI JSON on stdout remains usable. Set show_progress=False to disable display.
Progress never changes scientific parameters, cache recipes or returned data.
"""
from contextvars import ContextVar
from functools import wraps
import inspect
import shutil
import sys
import threading
import time

_CURRENT = ContextVar('sem_current_progress', default=None)


class Progress:
    """One batch display; nested instrumented functions update its stage."""
    def __init__(self, total, description, *, enabled=True, unit='images', stream=None):
        if not isinstance(enabled, bool):
            raise ValueError('show_progress must be True or False.')
        self.total = int(total)
        if self.total < 0:
            raise ValueError('Progress total cannot be negative.')
        self.description = description
        self.enabled = enabled
        self.unit = unit
        self.stream = sys.stderr if stream is None else stream
        self.completed = 0
        self.item = ''
        self.phase = 'starting'
        self._token = None
        self._started = False
        self._closed = False
        self._guard = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._last_render = float('-inf')
        self._start_time = None
        self._tty = bool(getattr(self.stream, 'isatty', lambda: False)())

    def start(self):
        if self._started:
            return self
        self._started = True
        self._start_time = time.monotonic()
        self._token = _CURRENT.set(self)
        self._render(force=True)
        if self.enabled and self._tty:
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()
        return self

    def _heartbeat(self):
        while not self._stop.wait(0.25):
            self._render()

    def _render(self, force=False):
        if not self.enabled or self._closed:
            return
        with self._guard:
            now = time.monotonic()
            if not force and now - self._last_render < 0.25:
                return
            self._last_render = now
            fraction = min(1., self.completed / self.total) if self.total else 1.
            filled = int(16 * fraction)
            elapsed = int(now - self._start_time)
            clock = f'{elapsed // 60:02d}:{elapsed % 60:02d}'
            line = (f'{self.description}: {fraction * 100:5.1f}% '
                    f'[{"#" * filled}{"-" * (16 - filled)}] '
                    f'{self.completed}/{self.total} {self.unit} | {clock} | {self.phase}')
            if self.item:
                line += ' | ' + self.item[-45:]
            if self._tty:
                width = max(40, shutil.get_terminal_size((120, 24)).columns - 1)
                line = line[:width].ljust(width)
            try:
                self.stream.write(('\r' if self._tty else '') + line + ('' if self._tty else '\n'))
                self.stream.flush()
            except (OSError, UnicodeError, ValueError):
                # A broken output stream must not stop processing or alter data.
                self.enabled = False

    def status(self, phase, *, item=None):
        with self._guard:
            self.phase = str(phase)
            if item is not None:
                self.item = str(item)
        self._render()

    def advance(self):
        with self._guard:
            self.completed = min(self.total, self.completed + 1)
        self._render()

    def items(self, iterable, *, name=str):
        """Count each inspected item, including cached, failed or reviewed items."""
        for item in iterable:
            self.status('checking', item=name(item))
            try:
                yield item
            finally:
                self.advance()

    def close(self, failed=None):
        if self._closed:
            return
        if failed is None:
            failed = sys.exc_info()[0] is not None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.phase = 'stopped (error)' if failed else 'done'
        self._render(force=True)
        if self.enabled and self._tty:
            try:
                self.stream.write('\n')
                self.stream.flush()
            except (OSError, UnicodeError, ValueError):
                pass
        self._closed = True
        if self._token is not None:
            _CURRENT.reset(self._token)
            self._token = None

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, value, traceback):
        self.close(failed=kind is not None)
        return False


def progress_function(function, description=None):
    """Show standalone work, or update the owning batch's stage without nesting.

    functools.wraps preserves inspect.getsource via __wrapped__, so logging
    wrappers do not invalidate the existing scientific implementation hashes.
    """
    if getattr(function, '_sem_progress_wrapped', False):
        return function
    label = description or function.__name__.replace('_', ' ').strip()

    @wraps(function)
    def wrapped(*args, **kwargs):
        show = kwargs.pop('show_progress', True)
        if not isinstance(show, bool):
            raise ValueError('show_progress must be True or False.')
        active = _CURRENT.get()
        if active is not None:
            if not active.enabled or not show:
                return function(*args, **kwargs)
            previous = active.phase
            active.status(label)
            try:
                return function(*args, **kwargs)
            finally:
                active.status(previous)
        with Progress(1, label, enabled=show, unit='calls') as progress:
            progress.status('running')
            result = function(*args, **kwargs)
            progress.advance()
            return result

    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    index = next((i for i, p in enumerate(parameters) if p.kind == inspect.Parameter.VAR_KEYWORD), len(parameters))
    parameters.insert(index, inspect.Parameter('show_progress', inspect.Parameter.KEYWORD_ONLY, default=True))
    wrapped.__signature__ = signature.replace(parameters=parameters)
    wrapped._sem_progress_wrapped = True
    wrapped.__doc__ = (function.__doc__ or '') + '\n\nshow_progress=True displays progress; False suppresses it.'
    return wrapped


def instrument_functions(namespace, *, exclude=(), include_private=()):
    """Instrument owned public functions and a small set of long helper stages."""
    for name, function in list(namespace.items()):
        if (name in exclude or (name.startswith('_') and name not in include_private)
                or not inspect.isfunction(function)
                or function.__module__ != namespace.get('__name__')):
            continue
        namespace[name] = progress_function(function)
