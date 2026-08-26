# Point numba at a writable cache directory before any submodule imports numba,
# so its compiled kernels survive across runs. See numba_cache.py for details.
from .numba_cache import cache_dir, clear_cache, configure_cache, warm_start

configure_cache()

__all__ = ["cache_dir", "clear_cache", "configure_cache", "warm_start"]
