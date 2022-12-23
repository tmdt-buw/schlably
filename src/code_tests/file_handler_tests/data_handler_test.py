"""
Data handler tests.
"""
import unittest

from src.utils.file_handler.data_handler import DataHandler


class TestDataHandler(unittest.TestCase):
    """
    Class with data handler tests.
    """
    def test_bad_config(self) -> None:
        """
        Test save data with bad config
        :return: None
        """
        with self.assertRaises(AssertionError):
            DataHandler.save_instances_data_file({}, [])


if __name__ == '__main__':
    unittest.main()
