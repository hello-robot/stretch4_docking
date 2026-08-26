import os
import sys
import time
from pathlib import Path

__all__ = [
    "cache_dir",
    "clear_cache",
    "configure_cache",
    "default_cache_dir",
    "warm_start",
]

NUMBA_CACHE_ENV = "NUMBA_CACHE_DIR"
OVERRIDE_ENV = "STRETCH4_DOCKING_CACHE_DIR"

# Numba names its cache files "<module>.<func>-<line>.py<ver>.nbi" (an index)
# and ".nbc" (the compiled objects). Only these are ever deleted.
CACHE_SUFFIXES = (".nbi", ".nbc")

_configured_dir: Path | None = None


def default_cache_dir() -> Path:
    """Where compiled kernels are cached when nothing overrides the location."""
    override = os.environ.get(OVERRIDE_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "stretch4_docking" / "numba"


def configure_cache() -> Path | None:
    """Point numba at a writable per-user cache directory.

    Called from ``stretch4_docking/__init__.py``, i.e. before any module in this
    package imports numba, because numba reads the cache location once when a
    cached kernel is first declared. Returns the directory in use, or None if no
    writable directory could be arranged (numba then falls back to its own
    ``__pycache__`` default, and kernels may recompile every run).
    """
    global _configured_dir

    preset = os.environ.get(NUMBA_CACHE_ENV)
    if preset:
        # Someone has already told numba where to cache; don't second-guess it.
        _configured_dir = Path(preset).expanduser()
        return _configured_dir

    path = default_cache_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    if not os.access(path, os.W_OK):
        return None

    os.environ[NUMBA_CACHE_ENV] = str(path)
    if "numba" in sys.modules:
        # Numba was imported before us and has already read its configuration.
        try:
            from numba.core import config as numba_config

            numba_config.reload_config()
        except Exception:  # pragma: no cover - numba internals moved
            pass
    _configured_dir = path
    return path


def cache_dir() -> Path | None:
    """The directory numba is currently caching this package's kernels in."""
    if _configured_dir is not None:
        return _configured_dir
    preset = os.environ.get(NUMBA_CACHE_ENV)
    return Path(preset).expanduser() if preset else None


def _in_tree_cache_files() -> list[Path]:
    """Numba cache files sitting in the installed package's ``__pycache__``.

    These only exist if the package ran without a configured cache directory,
    but they would still make a "first run" look fast, so clearing sweeps them.
    """
    package_root = Path(__file__).resolve().parent
    return [
        path
        for pycache in package_root.rglob("__pycache__")
        for path in pycache.iterdir()
        if path.is_file() and path.suffix in CACHE_SUFFIXES
    ]


def clear_cache(path: Path | str | None = None) -> tuple[int, int]:
    """Delete the cached kernels so the next run recompiles from scratch.

    Removes only numba's own ``.nbi``/``.nbc`` files -- from the cache directory
    and from any ``__pycache__`` inside the installed package -- then prunes the
    directories left empty. Returns ``(files_removed, bytes_freed)``.
    """
    root = Path(path).expanduser() if path is not None else cache_dir()

    targets = list(_in_tree_cache_files())
    if root is not None and root.is_dir():
        targets += [
            f for f in root.rglob("*") if f.is_file() and f.suffix in CACHE_SUFFIXES
        ]

    removed = 0
    freed = 0
    for target in targets:
        try:
            size = target.stat().st_size
            target.unlink()
        except OSError:
            continue
        removed += 1
        freed += size

    if root is not None and root.is_dir():
        for directory in sorted(root.rglob("*"), reverse=True):
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    pass

    return removed, freed


def warm_start(verbose: bool = False) -> dict[str, float]:
    """Compile (or load from cache) every numba kernel in this package.
    """
    timings: dict[str, float] = {}

    def run(name, fn):
        start = time.perf_counter()
        fn()
        timings[name] = time.perf_counter() - start
        if verbose:
            print(f"  {name:<14} {timings[name] * 1000.0:8.1f} ms")

    if verbose:
        print(f"warming numba cache in {cache_dir() or '<numba default>'}")

    def warm_tracker():
        from stretch4_docking.trackers.dock_tracker import DockTracker

        DockTracker().warm_start()

    def warm_ring_filter():
        from stretch4_docking.costmap.ring_filter import RingFilter

        RingFilter().warm_start()

    def warm_costmap():
        from stretch4_docking.costmap.costmap import Costmap

        Costmap().warm_start()

    def warm_cloud_reader():
        from stretch4_docking.utils import cloud_reader

        cloud_reader.warm_start()

    run("tracker", warm_tracker)
    run("ring_filter", warm_ring_filter)
    run("costmap", warm_costmap)
    run("cloud_reader", warm_cloud_reader)

    if verbose:
        print(f"  {'total':<14} {sum(timings.values()) * 1000.0:8.1f} ms")
    return timings


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="stretch_autodock_cache",
        description="Warm up, inspect, or clear the on-disk numba kernel cache.",
    )
    parser.add_argument("--path", action="store_true", help="print the cache directory and exit")
    parser.add_argument("--clear",action="store_true", help="delete the cached kernels (the next run recompiles them)")
    args = parser.parse_args()

    if args.path:
        print(cache_dir() or "<numba default: __pycache__ beside the sources>")
        return 0

    if args.clear:
        location = cache_dir()
        removed, freed = clear_cache()
        where = f" from {location}" if location is not None else ""
        print(f"removed {removed} cached kernel files ({freed / 1024.0:.1f} KiB){where}")
        return 0

    warm_start(verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
