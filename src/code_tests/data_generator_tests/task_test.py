"""
Task class tests.
"""
import unittest

from src.data_generator.task import Task


class TestTask(unittest.TestCase):
    """
    Class with tests for the task class.
    """
    def test_init_required_args(self) -> None:
        """
        Check error handling for missing args
        :return: None
        """
        # Test Task class instance without required arguments
        with self.assertRaises(TypeError):
            _ = Task()
        with self.assertRaises(TypeError):
            _ = Task(1)

    def test_init_wrong_type(self) -> None:
        """
        Check error handling for wrong args
        :return: None
        """
        # Test Task class instance without all required arguments
        with self.assertRaises(TypeError):
            _ = Task("", "")
        with self.assertRaises(TypeError):
            _ = Task([], "")
        with self.assertRaises(TypeError):
            _ = Task([], [])
        with self.assertRaises(TypeError):
            _ = Task(None, None)
        with self.assertRaises(TypeError):
            _ = Task(1, None)
        with self.assertRaises(TypeError):
            _ = Task(None, 1)

    def test_init_minimal(self) -> None:
        """
        Test minimal init
        :return: None
        """
        # Test Task class init with minimal example
        task = Task(0, 0)
        self.assertIsInstance(task, Task)
        task = Task(int(), int())
        self.assertIsInstance(task, Task)

    def test_default_init(self) -> None:
        """
        Test default init
        :return: None
        """
        # Test Task class init for default values
        # Create instance
        task = Task(0, 0)
        # Test required and set parameters
        self.assertEqual(task.job_index, 0)
        self.assertEqual(task.task_index, 0)
        # Test all other default parameters are None
        instance_variables = vars(task)
        default_none_parameters = {k: instance_variables[k] for k in
                                   instance_variables.keys() - ['job_index', 'task_index']}
        self.assertTrue(all(v is None for v in list(default_none_parameters.values())),
                        msg="All other parameters other than the given need to be initialized with None.")


if __name__ == '__main__':
    unittest.main()
