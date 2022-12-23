"""
Gantt chart tests.
"""
import unittest

import copy
from pathlib import Path
from typing import List
import PIL.Image

from src.agents.heuristic.heuristic_agent import HeuristicSelectionAgent
from src.data_generator.task import Task
from src.environments.env_tetris_scheduling import Env
from src.utils.file_handler.config_handler import ConfigHandler
from src.utils.file_handler.data_handler import DataHandler
from src.visuals_generator.gantt_chart import GanttChartPlotter


class TestGanttChart(unittest.TestCase):
    """
    Class with gantt chart tests.
    """
    _test_tasks: List[Task]

    @classmethod
    def setUpClass(cls) -> None:
        """

        :return: None
        """

        env_config = ConfigHandler.get_config(config_file_path='training/dqn/config_job3_task4_tools0.yaml')
        data = DataHandler.load_instances_data_file(config=env_config)
        cls._test_tasks = copy.deepcopy(data[0])

        done = False
        env = Env(env_config, [data[0]])
        heuristic_agent = HeuristicSelectionAgent()
        while not done:
            # obs = env.state_obs
            mask = env.get_action_mask()

            cls._test_tasks = env.tasks
            task_mask = mask
            action = heuristic_agent(cls._test_tasks, task_mask, 'rand')

            res = env.step(action)
            done = res[2]

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Tear down class
        :return: None
        """
        del cls._test_tasks
        cls._trap = None

    def test_get_gantt_chart_image(self) -> None:
        """
        Test gantt chart image
        :return: None
        """
        test_image = GanttChartPlotter.get_gantt_chart_image(self._test_tasks)
        self.assertIsInstance(test_image, PIL.Image.Image)

    def test_get_gantt_chart_image_and_save(self) -> None:
        """
        Test gantt chart image and save
        :return: None
        """
        test_image_path: Path = \
            GanttChartPlotter.get_gantt_chart_image_and_save(self._test_tasks,
                                                             filename="automated_test_random_name_dc55c0e399428u7e",
                                                             file_type="png")
        test_image_path.unlink()

    def test_get_gantt_chart_gif_and_save(self) -> None:
        """
        Test gantt chart gif and save
        :return: None
        """
        # TODO - prevent print output - redirect_stdout does not work
        test_gif_path: Path = \
            GanttChartPlotter.get_gantt_chart_gif_and_save(self._test_tasks,
                                                           filename="automated_test_random_name_dc55c0e399428e3e",
                                                           save_intermediate_images=False,
                                                           quality_dpi=55)
        test_gif_path.unlink()


if __name__ == '__main__':
    unittest.main()
