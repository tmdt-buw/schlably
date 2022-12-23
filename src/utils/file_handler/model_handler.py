"""
This file provides functions to handle path determine correct model paths.

Constants:
MODEL_DATA_DIRECTORY: Parent directory where all experiments will be located
EXPERIMENT_SAVE_PATH: Default experiment path
MODEL_SAVE_FILE: Default model file name
"""
# OS imports
from pathlib import Path

# constants
MODEL_DATA_DIRECTORY: Path = Path(__file__).parent.parent.parent.parent / 'data'
EXPERIMENT_SAVE_PATH: str = 'models'
MODEL_SAVE_FILE: str = "example_agent"


class ModelHandler:
    """
    Handles the determination of correct model paths.
    """
    @staticmethod
    def get_best_model_path(config: dict):
        """
        Determines the best model path

        :param config

        :return: Path to best model according to config parameters

        """

        path = config.get('experiment_save_path', EXPERIMENT_SAVE_PATH)
        file = config.get('saved_model_name', None)

        if file is None:
            file = MODEL_SAVE_FILE

        return MODEL_DATA_DIRECTORY / path / file

    @staticmethod
    def get_compare_path(config: dict):
        """
        Determines the compare model path

        :param config:

        :return: Path to compare model according to config parameters

        """

        path = config.get('experiment_save_path', EXPERIMENT_SAVE_PATH)

        return MODEL_DATA_DIRECTORY / path / 'compare'
