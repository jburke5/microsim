import os
from unittest import skipIf


def skip_if_quick_mode(func_or_cls):
    """Skips test if `MICROSIM_TEST_QUICKMODE` env var is set and non-empty."""
    decorator = skipIf(is_quick_mode_enabled(), "Skipping slow test: quick mode enabled")
    return decorator(func_or_cls)


def is_quick_mode_enabled():
    quick_mode_env_var = os.environ.get("MICROSIM_TEST_QUICKMODE", None)
    return quick_mode_env_var is not None and quick_mode_env_var != ""
