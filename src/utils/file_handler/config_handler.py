"""
This file provides functions to handle the import and export of configs (e.g. training config).

Constants:
CONFIG_DIRECTORY: Path: Parent directory for the config import
FILE_PATH_VARIABLES: list[str]: Necessary variables to load config by dict
DIRECTORY_PATH_VARIABLES: dict[str, str]: Dict to specify whether training, testing or data_generation should be loaded

SCHEMA_FILE: str: Schema file for checking the loaded configs
"""
# OS imports
import os
from pathlib import Path
from typing import List, Dict
import warnings
import operator

# Data handling imports
import yaml
import json
import jsonschema

# Constants
CONFIG_DIRECTORY: Path = Path(__file__).parent.parent.parent.parent / 'config'
DIRECTORY_PATH_VARIABLES: Dict[str, str] = {'data_generation': 'sp_type', 'training': 'algorithm',
                                            'testing': 'test_algorithm'}
CONFIG_REQUIREMENTS_FILE: Path = Path(__file__).parent / 'config_requirements.json'

SCHEMA_FILE: str = "schema.json"


class ConfigHandler:
    """
    Handles the import and export of configs.
    """
    @classmethod
    def get_sub_dir_path_from_config(cls, config: dict) -> str:
        """
        Determines the subdirectory of a config (e.g. training)

        :param config: Config

        :return: Sub filepath

        """
        for parent_dir, sub_dir_from_config in DIRECTORY_PATH_VARIABLES.items():
            if sub_dir_from_config in config.keys():
                return f'{parent_dir}/{config[sub_dir_from_config]}'

        raise NotADirectoryError("The directory specified in your config does not exist in DICT_PATH_VARIABLES")

    @classmethod
    def get_sub_dir_from_path(cls, path_from_terminal: Path) -> str:
        mode_path = path_from_terminal.parts[0]
        agent_path = path_from_terminal.parts[1]

        return f'{mode_path}/{agent_path}'

    @classmethod
    def check_config_parameters(cls, config_to_check) -> bool:
        """
        Checks if config parameters match requirements

        :param config_to_check: config to be checked

        :return: True if all config parameters match the requirements

        """
        with open(CONFIG_REQUIREMENTS_FILE, 'r') as handle:
            config_requirements = json.load(handle)

        for req in config_requirements:
            param1 = config_to_check.get(req['param1'], None)
            param2 = config_to_check.get(req['param2'], None)
            # abort if at least one param is missing in config
            if not (param1 and param2):
                continue

            # get operator function and check if param fulfill requirement
            if not getattr(operator, req['op'])(param1, param2):
                warnings.warn(req['err_mess'], category=RuntimeWarning)

        return False

    @classmethod
    def check_config(cls, file_path_to_check, config_to_check) -> None:
        """
        Checks config against schema and requirements

        :param file_path_to_check:
        :param config_to_check:

        :return: None

        """
        # Check against schema
        schema_check: bool = SchemaHandler.check_file_dict_against_schema_dict(
            config_to_check,
            SchemaHandler.get_schema(sub_dir=ConfigHandler.get_sub_dir_from_path(file_path_to_check))
        )
        # Only continue if config matches schema - otherwise it will break at another point
        assert schema_check, f"Config at {file_path_to_check} failed the schema check. " \
                             f"Change the config according to schema and restart."

        # check against requirements
        cls.check_config_parameters(config_to_check)

    @classmethod
    def get_config_from_path(cls, config_file_path: str, check_against_schema: bool = True) -> Dict:
        """
        Initializes the loading of a config

        :param config_file_path: Relative path to a config file (e.g. training/dqn/config_job3_task4_tools0.yaml) which
            was entered to the terminal
        :param check_against_schema: Checking against schema is activated per default (True), can be deactivated
            by setting to False

        :return: Config dict

        """
        config_file_path = Path(config_file_path)
        # create full path
        config_path = CONFIG_DIRECTORY / config_file_path
        # Only possible if config file exists
        assert config_path.exists(), f"Path {config_path} not found. " \
                                     f"You need to point to a config in accordance to your settings in the config " \
                                     f"folder"

        # Load config
        with open(config_path, 'r') as stream:
            current_config = yaml.load(stream, Loader=yaml.Loader)

        # check
        if check_against_schema:
            cls.check_config(config_file_path, current_config)

        return current_config

    @classmethod
    def get_config(cls, config_file_path=None, external_config=None, check_against_schema: bool = True) -> Dict:
        """
        Gets a config from file or uses external config, according to input

        :param config_file_path: Path to the config file
        :param external_config: Config which was created or loaded in an external script
        :param check_against_schema: Checking against schema is activated per default (True), can be deactivated
            by setting to False

        :return: config dictionary

        """
        assert bool(config_file_path) != bool(external_config), \
            'You either have to specify a path to the config you want to use for' \
            'OR provide a pass a loaded config to this function'
        if config_file_path is not None:
            config = cls.get_config_from_path(
                config_file_path=config_file_path,
                check_against_schema=check_against_schema
            )
        else:
            config = external_config

        return config


class SchemaHandler:
    """
    Handles the schema check of loaded configs.
    """
    @classmethod
    def load_json(cls, json_path: Path, encoding: str = 'utf-8') -> dict:
        """
        Loads a json file

        :param json_path: Path to json file
        :param encoding: Encoding of the json file

        :return: Json file

        """
        json_file: dict
        with open(json_path, mode='r', encoding=encoding) as open_file:
            json_file = json.load(open_file)
        return json_file

    @classmethod
    def check_file_dict_against_schema_dict(cls, file_dict: dict, schema_dict: dict) -> bool:
        """
        Checks file against schema

        :param file_dict: File (e.g. config)
        :param schema_dict: Schema

        :return: True if the file matches the schema , else False

        """
        if isinstance(file_dict, dict) and isinstance(schema_dict, dict):
            try:
                jsonschema.validate(file_dict, schema_dict)
            except jsonschema.exceptions.ValidationError as caught_ex:
                print(caught_ex)
            except jsonschema.exceptions.SchemaError as caught_ex:
                print(caught_ex)
            else:
                return True
        return False

    @classmethod
    def _schema_file_path_from_variables(cls, sub_dir: str, **kwargs) -> Path:
        """
        Determines path to schema file

        :param sub_dir: Subdirectory of the schema file
        :param kwargs: Unused

        :return: Path to schema file

        """
        # check if path exists
        if not os.path.exists(CONFIG_DIRECTORY / sub_dir / SCHEMA_FILE):

            # Check if sub_dir with Path operations
            sub_sub_dir = sub_dir
            # check if sub_dir is sub_sub_..._dir and complete path
            for directory in [x[0] for x in os.walk(CONFIG_DIRECTORY)]:
                if directory[-len(sub_sub_dir):] == sub_sub_dir:
                    sub_dir = directory
                    break

        return CONFIG_DIRECTORY / sub_dir / SCHEMA_FILE

    @classmethod
    def get_schema(cls, sub_dir: str) -> dict:
        """
        Loads schema from file

        :param sub_dir: Subdirectory of the schema file

        :return: Schema

        """
        schema_dict: dict
        # Get path
        schema_file_path = SchemaHandler._schema_file_path_from_variables(sub_dir=sub_dir)

        # Check path and load
        if schema_file_path.exists():
            schema_dict = SchemaHandler.load_json(schema_file_path)
        else:
            assert False, f"Schema at {schema_file_path} does not exist. " \
                          f"Disable the schema check or provide the requested schema."

        return schema_dict
