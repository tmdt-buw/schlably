"""
Use this module to run all tests.
"""
import unittest

# Import test modules
from src.code_tests.data_generator_tests import task_test
from src.code_tests.data_generator_tests import sp_factory_test
from src.code_tests.file_handler_tests import data_handler_test
from src.code_tests.file_handler_tests import config_handler_test
from src.code_tests.visuals_generator_tests import gantt_chart_test


def main() -> unittest.TestResult:
    """
    Runs all code tests
    :return: Test result
    """
    # Initialize test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add scheduler code_tests to the test suite
    suite.addTests(loader.loadTestsFromModule(task_test))
    suite.addTests(loader.loadTestsFromModule(sp_factory_test))
    suite.addTests(loader.loadTestsFromModule(data_handler_test))
    suite.addTests(loader.loadTestsFromModule(config_handler_test))
    suite.addTests(loader.loadTestsFromModule(gantt_chart_test))

    # Initialize runner, pass it to suite and run it
    runner = unittest.TextTestRunner(verbosity=3)
    result: unittest.TestResult = runner.run(suite)

    # Return test results
    return result


if __name__ == '__main__':
    main()
