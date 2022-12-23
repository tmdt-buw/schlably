"""
This file provides utility functions to load configs, data and agents according to the config. It is used in training
and testing.

TIMESTAMP: str: timestamp of the training run, used for the creation of a unique model name
AGENT_DICT: dict[str, str]: This dictionary is used to map algorithm identifiers (keys)
to their actual class names (values).

E.g. to use the MaskedPPO class, you can use ppo as algorithm in the config.

If you add new algorithms, you can extend this dictionary to assign your algorithm class a short identifier.

"""
import datetime
from typing import List, Dict, Any

from src.data_generator.task import Task
from src.utils.file_handler.config_handler import ConfigHandler
from src.utils.file_handler.data_handler import DataHandler

# Agent Imports (do not delete)
from src.agents.reinforcement_learning.ppo import PPO
from src.agents.reinforcement_learning.dqn import DQN
from src.agents.reinforcement_learning.ppo_masked import MaskedPPO

# Constants
TIMESTAMP: str = f"{datetime.datetime.now().strftime('%d%m%Y%H%M')}"
AGENT_DICT: Dict[str, str] = {'ppo_masked': 'MaskedPPO', 'ppo': 'PPO', 'dqn': 'DQN'}
TRAIN_ALGORITHM_PARAM: str = 'algorithm'
TEST_ALGORITHM_PARAM: str = 'test_algorithm'


def load_config(config_path, external_config) -> dict:
    """
    Uses the ConfigHandler routines to load the config according to the path

    :param config_path: Path to the config to be loaded
    :param external_config: Config dict

    :return: Config

    """
    # get config
    config: dict = ConfigHandler.get_config(config_path, external_config)

    return complete_config(config)


def load_data(config: dict) -> List[List[Task]]:
    """
    Uses the DataHandler routines to load the training config

    :param config: Config dict which specifies a dataset

    :return: Dataset (List of instances)

    """
    # Load data as given by config
    data: List[List[Task]] = DataHandler.load_instances_data_file(config)
    return data


def complete_config(config: dict) -> dict:
    """
    If optional parameters have not been defined in the configuration, this function adds default values. Also creates
    missing directories, if necessary.

    :param config: config file

    :return: completed config file

    """
    # Set model name
    if config.get('saved_model_name', None) == 'automatic':
        config.update({'saved_model_name': f'agent_{TIMESTAMP}'})

    return config


def get_agent_param_from_config(config: dict) -> str:
    """
    Check if config has TRAIN or TEST algorithm param and get corresponding class string for algorithm from config

    :param config: Config for training or testing

    :return: Agent type string (e.g. 'ppo')

    """
    # check if config has TRAIN or TEST algorithm param and get corresponding class string for algorithm from config
    if TRAIN_ALGORITHM_PARAM in config.keys():
        agent_string = config[TRAIN_ALGORITHM_PARAM]
    elif TEST_ALGORITHM_PARAM in config.keys():
        agent_string = config[TEST_ALGORITHM_PARAM]
    else:
        raise KeyError(f"For training or testing you need to specify an agent type using the {TRAIN_ALGORITHM_PARAM} or"
                       f"{TEST_ALGORITHM_PARAM} parameter")
    return agent_string


def get_agent_class_from_config(config: dict) -> Any:
    """
    Determines and loads the correct agent class type according the config

    :param config: Training config

    :return: Agent class type which can be called

    """
    agent_param = get_agent_param_from_config(config)
    # use AGENT_DICT to determine the correct Class name for an agent shortcut
    class_string = AGENT_DICT[agent_param]

    # get class by string from global path
    agent_class = globals().get(class_string)
    assert agent_class, \
        f"{config['algorithm']} does not exist as an implemented class. " \
        f"Check spelling or use a different algorithm."  # config['algorithm']

    return agent_class
