"""
This file provides functions to handle the import and export of datasets.

Constants:
DATA_DIRECTORY: Path: Parent directory for the data import
FILE_PATH_VARIABLES: list[str]: Necessary variables to load data by config
"""
# OS imports
import json
from pathlib import Path

# Functional imports
import pickle
from typing import List, Dict
from src.data_generator.task import Task

# Constants
DATA_DIRECTORY: Path = Path(__file__).parent.parent.parent.parent / 'data' / 'instances'
SOLVER_DATA_DIRECTORY: Path = Path(__file__).parent.parent.parent.parent / 'data' / 'models' / 'solver_solution'
FILE_PATH_VARIABLES: List[str] = ['sp_type', 'num_jobs', 'num_tasks', 'num_tools']
SOLVER_DATA_DICT: str = 'instance_dict.pkl'
INSTANCES_FILE_CONFIG_KEY: str = 'instances_file'


class DataHandler:
    """
    Handles the import and export of datasets.
    """
    @classmethod
    def _data_file_path_from_variables(cls, sp_type: str, num_jobs: int, num_tasks: int, num_tools: int,
                                       **_kwargs) -> Path:
        """
        Creates a name fpr the datafile from config variables

        :param sp_type: Scheduling problem type (e.g. "jssp")
        :param num_jobs: number of jobs generated in an instance
        :param num_tasks: number of tasks per job generated in an instance
        :param num_tools: number of tools available
        :param kwargs: Additional kwargs

        :return: Path to datafile

        """
        return DATA_DIRECTORY / sp_type / f"config_job{num_jobs}_task{num_tasks}_tools{num_tools}.pkl"

    @classmethod
    def save_instances_data_file(cls, config: dict, data: List[List[Task]]) -> None:
        """
        Saves instances as file

        :param config: Config with at least FILE_PATH_VARIABLES
        :param data: List of instances

        :return: None

        """
        # get path
        if INSTANCES_FILE_CONFIG_KEY in config:
            data_file_path = DATA_DIRECTORY / config.get('sp_type') / config.get(INSTANCES_FILE_CONFIG_KEY)
        elif all(keyword in config for keyword in FILE_PATH_VARIABLES):
            data_file_path = cls._data_file_path_from_variables(**config)
        else:
            assert False, f"You either have to specify the {INSTANCES_FILE_CONFIG_KEY} or all following " \
                          f"parameters {FILE_PATH_VARIABLES} in your config, to write your generated data to a file"
        # create path and save
        if not data_file_path.exists():
            data_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_file_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_instances_data_file(cls, config: dict = None,
                                 instances_data_file_path: Path = None) -> List[List[Task]]:
        """
        Load instances by path, config or filedialog

        :param config: Config with at least TASK_FILE_CONFIG_VARIABLE or FILE_PATH_VARIABLES parameters
        :param instances_data_file_path: Relative path to the instance_data_file you want to load
        (e.g. jssp/config_job3_task4_tools0.pkl)

        :return: List of instances

        """

        assert instances_data_file_path or config, 'You either have to specify a path or a config' \
                                                   'to load an instances data file'

        data: List[List[Task]]

        if instances_data_file_path:
            data_file_path = DATA_DIRECTORY / instances_data_file_path
        elif INSTANCES_FILE_CONFIG_KEY in config:
            data_file_path = DATA_DIRECTORY / config[INSTANCES_FILE_CONFIG_KEY]
        else:
            assert False, f"Missing {INSTANCES_FILE_CONFIG_KEY} variable"
        if data_file_path.exists():
            with open(data_file_path, 'rb') as handle:
                data = pickle.load(handle)
        else:
            assert False, f"Missing file at path: {data_file_path}"

        return data

    @classmethod
    def write_solved_data_to_file(cls, instances_file_path: dict, data: List[List[Task]]) -> None:
        """
        Saves the solutions computed by the solver

        :param instances_file_path:
        :param data: List of solved instances to be saved

        :return: None

        """
        # complete path
        solved_data_path = SOLVER_DATA_DIRECTORY / f"{instances_file_path}"
        if not solved_data_path.exists():
            solved_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(solved_data_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load_solved_data(cls, load_config) -> List[List[Task]]:
        """
        Loads the solutions computed by the solver

        :param load_config: Config with at least sp_type, num_jobs, num_tasks, num_tools parameters

        :return: List of solved instances, if file specified by config parameters exists. Else None

        """
        solved_data_file = SOLVER_DATA_DIRECTORY / f"{load_config[INSTANCES_FILE_CONFIG_KEY]}"
        data = None
        if solved_data_file.exists():
            with open(solved_data_file, 'rb') as handle:
                data = pickle.load(handle)
        return data

    @classmethod
    def load_solved_instance_by_hash(cls, instance_hash: int) -> List[Task]:
        """
        Searches and loads the solved instance according to the hash if exists

        :param instance_hash: Individual hash of the instance to be loaded

        :return: Instance from the solver parse_to_plottable_format function if exists in file, else None

        """
        solved_instance = None

        # check if ile exist
        if (SOLVER_DATA_DIRECTORY / SOLVER_DATA_DICT).is_file():
            with open(file=SOLVER_DATA_DIRECTORY / SOLVER_DATA_DICT, mode='rb') as handle:
                data = pickle.load(handle)
                # check if hash exists in dict
            if instance_hash in data.keys():
                # assign solved_instance from dict to output variable
                solved_instance = data[instance_hash]

        # return instance value of directory according to the hash key
        return solved_instance

    @classmethod
    def write_solved_instance_by_hash(cls, solved_instance: List[Task], instance_hash: int) -> None:
        """
        Writes the solved_instance as value to the SOLVER_DATA_DICT file. Instance_hash is used as key

        :param solved_instance: Instance from the solver parse_to_plottable_format function to be saved
        :param instance_hash: Individual hash of the instance to be saved

        :return: None

        """
        # create item with has as key and solution as value
        new_instance = {instance_hash: solved_instance}

        # check if file exist
        if (SOLVER_DATA_DIRECTORY / SOLVER_DATA_DICT).is_file():
            with open(file=SOLVER_DATA_DIRECTORY / SOLVER_DATA_DICT, mode='rb') as handle:
                data: Dict = pickle.load(handle)
            data.update(new_instance)
            new_data = data
        else:
            new_data = new_instance
            # create path if it doesn't exist
            if not SOLVER_DATA_DIRECTORY.exists():
                SOLVER_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

        # dump new data
        with open(file=SOLVER_DATA_DIRECTORY / SOLVER_DATA_DICT, mode='wb') as handle:
            pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
