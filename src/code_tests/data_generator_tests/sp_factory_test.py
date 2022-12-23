"""
Scheduling problem factory tests.
"""
import unittest

from src.data_generator.sp_factory import SPFactory


class TestSPFactory(unittest.TestCase):
    """
    Class with sp_factory tests
    """
    def test_generate_instances_preset(self) -> None:
        """
        Test if presets lead to valid instances
        :return: None
        """
        # Should return list with preset size - not empty!
        instances = SPFactory.generate_instances()
        # Test is list
        self.assertIsInstance(instances, list)
        # Test list is not empty
        self.assertTrue(len(instances) > 0)

    def test_generate_instances_edge_case(self) -> None:
        """
        Test instance generation for edge case num_instances = 0
        :return: None
        """
        # Should return list with preset size - not empty!
        instances = SPFactory.generate_instances(num_instances=0)
        # Test is list
        self.assertIsInstance(instances, list)
        # Test list is empty
        self.assertFalse(len(instances))

    def test_implemented_sp_types(self) -> None:
        """
        Test implemented scheduling problem types
        :return: None
        """
        # Should return list with preset size - not empty!
        instances = SPFactory.generate_instances(sp_type='jssp')
        # Test is list
        self.assertIsInstance(instances, list)

        with self.assertRaises(AssertionError):
            SPFactory.generate_instances(sp_type='')
        with self.assertRaises(AssertionError):
            SPFactory.generate_instances(sp_type='thiswillneverbeimplementedsodonteventhinkaboutit21462#')


if __name__ == '__main__':
    unittest.main()
