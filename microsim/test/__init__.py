def load_tests(loader, standard_tests, pattern):
    import os

    test_dir = os.path.dirname(__file__)
    # `discover` will fail if `pattern` is `None`, so pass default in that case instead
    discover_pattern = "test_*.py" if pattern is None else pattern
    package_tests = loader.discover(test_dir, discover_pattern)
    standard_tests.addTests(package_tests)
    return standard_tests


def run_all_tests():
    import sys
    import unittest

    test_module = sys.modules[__name__]
    unittest.main(module=test_module)


__all__ = [run_all_tests]
