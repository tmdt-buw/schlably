"""
Config handler tests.
"""
import unittest

from src.utils.file_handler.config_handler import SchemaHandler


class TestSchemaHandler(unittest.TestCase):
    """
    Class with schema handler tests
    """
    def test_none_behavior(self) -> None:
        """
        Test none behavior in dicts
        :return: None
        """
        self.assertFalse(SchemaHandler.check_file_dict_against_schema_dict(None, None))
        self.assertFalse(SchemaHandler.check_file_dict_against_schema_dict({}, None))
        self.assertFalse(SchemaHandler.check_file_dict_against_schema_dict(None, {}))

    def test_empty_behavior(self) -> None:
        """
        Test schema check with empty dicts
        :return: None
        """
        self.assertTrue(SchemaHandler.check_file_dict_against_schema_dict({}, {}))

    def test_unavailable_path_load_behavior(self) -> None:
        """
        Test load schema with non-existing subdirectory
        :return: None
        """
        with self.assertRaises(AssertionError):
            SchemaHandler.get_schema(sub_dir="unavailable_87a7e9105711_sp-type-that-will-never-exist")

    def test_available_path_load_behavior(self) -> None:
        """
        Test load schema for existing subdirectory
        :return: None
        """
        available_sp_type = "jssp"
        self.assertIsInstance(SchemaHandler.get_schema(sub_dir=available_sp_type), dict)


if __name__ == '__main__':
    unittest.main()
