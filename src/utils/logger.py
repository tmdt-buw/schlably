"""
This file provides the Logger class. The logger is called during the whole training and testing process,
to track the current status or results.

Until now the logger only provides weights and biases (wandb) as logging method.
You can add other logging methods and call them in the dump function to use existing logger infrastructure.
If you want to use wandb, please follow the "Install" part of the README and refer the constants explanations below.

Constants:

| LOG_MODE: int: Wandb log mode. Choose from [0: no wandb, 1: wandb_offline, 2: wandb_online]
| WANDB_PROJECT: str: Name of your wandb project where you want to upload all wandb logs
| WANDB_ENTITY: str: Name of your wandb entity where you want to upload all wandb logs
| WANDB_TABLE_COLUMNS list[str]: The logger uploads tables with testing results. This list is used to specify the columns
  names. If you change the content of the table, you can adapt the column labels
| WANDB_FINAL_EVALUATION_TABLE_COLUMNS List[str]: The logger uploads the final evaluation results in a separate table
  with these columns.

LOG_MODE and WANDB_PROJECT can also be set in training config.

"""
import os
import wandb
from typing import Dict, Any, List
from collections import defaultdict
from pathlib import Path
import PIL.Image


# Constants
LOG_MODE_DEFAULT: int = 0
WANDB_PROJECT: str = 'project-1'
WANDB_ENTITY: str = 'scheduling-sandbox-1'
WANDB_DIRECTORY: Path = Path(__file__).parent.parent.parent / 'data'
WANDB_TABLE_COLUMNS: List[str] = ["Agent", "Reward", "Makespan", "Tardiness", "Ganttchart"]
WANDB_FINAL_EVALUATION_TABLE_COLUMNS: List[str] = ['Agent', 'Mean Reward', 'Mean Tardiness', 'Tardiness Max',
                                                   'Mean Makespan', 'Makespan STD', 'Tardiness STD', 'Gap To Solver']


class Logger:
    """
    This class can store various parameters (e.g. loss from a model update) as key value pairs.
    By calling the dump function, all stored parameters are logged according to the log_mode.
    Because the logger supports wandb there are several functions to initialize and handle the logging to wandb

    :param: config: Training config

    """
    def __init__(self, config: dict):
        self.log_mode = config.get('wandb_mode', LOG_MODE_DEFAULT)
        self.record_buffer: Dict[str, Any] = defaultdict(Any)
        self.config = config

        self.wandb_run = None
        self.wandb_table = None
        self.initialize_wandb()

    def record(self, logging_values: Dict[str, Any]) -> None:
        """
        Stores all items of the input dictionary in the record_buffer

        :param logging_values: Dictionary with items to be logged

        :return: None

        """
        # store/overwrite key-value pairs
        for log_key, log_val in logging_values.items():
            if log_key not in self.record_buffer.keys():
                self.record_buffer[log_key] = log_val
            else:
                Warning('You are overwriting a log record before it has been dumped!')
                self.record_buffer[log_key] = log_val

    def clear_buffer(self) -> None:
        """
        Empties the record_buffer

        :return: None

        """
        self.record_buffer.clear()
        self.wandb_table = None

    def dump(self) -> None:
        """
        Logs all stored parameters according to the log_mode and clears the buffer afterwards

        :return: None

        """
        # dump according to log_mode
        if self.check_wandb():
            self.dump_wandb()
        else:
            self.dump_console()

        # clear buffer
        self.clear_buffer()

    def dump_console(self):
        pass

    def dump_wandb(self) -> None:
        """
        Logs all stored parameters to wandb

        :return: None

        """
        for key, value in self.record_buffer.items():
            wandb.log({key: value})

    def check_wandb(self) -> bool:
        """
        Check if logger is in wandb mode

        :return: True if the logger instance should use wandb according to config or constants

        """
        return self.log_mode == 1 or self.log_mode == 2

    def initialize_wandb(self) -> None:
        """
        Initializes the wandb run

        :return: None

        """
        if self.check_wandb():
            # Set wandb mode offline if requested (has to be set before calling wandb.init())
            if self.log_mode == 1:
                os.environ['WANDB_MODE'] = 'offline'
            self.wandb_run = wandb.init(project=self.config.get('wandb_project', WANDB_PROJECT), entity=WANDB_ENTITY,
                                        config=self.config, reinit=True, dir=WANDB_DIRECTORY)

            # overwrite logger config with wandb config (e.g. for the case wandb config was changed by sweep)
            self.config = dict(wandb.config.items())

    def log_wandb_artifact(self, artifact_info: dict, file_path: str = None) -> None:
        """
        Logs the input artifact to wandb (e.g. a dataset or config file)

        :param file_path: Path to the artifact to be logged
        :param artifact_info: At least have to contain keys 'name' and 'type'. 'description' and 'metadata' are optional

        :return: None

        """
        # Log only if wandb has been initialized
        if self.wandb_run:
            assert 'name' in artifact_info.keys() and 'type' in artifact_info.keys(),\
                "Please specify 'name' and  'type' in artifact_info when trying to create an log an wandb artifact"
            task_file: wandb.Artifact = wandb.Artifact(**artifact_info)
            task_file.add_file(file_path) if file_path else Warning('Artifact has been logged without adding a file')
            self.wandb_run.log_artifact(task_file)

    def add_row_to_wandb_table(self, agent: str, gantt_chart: PIL.Image,  **kwargs) -> None:
        """
        Add the recent gantt_chart and kwargs as row to the table in the record buffer.
        The table will be logged to wandb when calling dump

        :param agent: Name of the agent whose results you want to log here
        :param gantt_chart: Gantt chart image
        :param kwargs: Additional results beside the gantt chart you want to log in the table (e.g. tardiness, makespan)

        :return: None

        """
        if self.wandb_run:
            # create table if not existing
            if not self.wandb_table:
                self.wandb_table = wandb.Table(columns=WANDB_TABLE_COLUMNS)

            # add data to wandb table
            log_data = [agent]
            log_data.extend(kwargs.values())
            log_data.append(wandb.Image(gantt_chart))
            self.wandb_table.add_data(*log_data)
            # overwrite the current state of table in the buffer
            self.record_buffer['test_table'] = wandb.Table(data=self.wandb_table.data, columns=self.wandb_table.columns)

    def write_to_wandb_summary(self, evaluation_results: dict):
        """
        Log results as summary to wandb

        :param evaluation_results: Dictionary with at least all evaluation result to be logged in this function

        :return: None

        """
        if self.wandb_run:
            final_evaluation_table = wandb.Table(columns=WANDB_FINAL_EVALUATION_TABLE_COLUMNS)
            # iterate overall all agent whose results are saved in evaluation_results
            for agent in evaluation_results.keys():
                log_data = []
                log_data.append(str(agent))
                log_data.append(evaluation_results[agent]['rew_mean'])
                log_data.append(evaluation_results[agent]['tardiness_mean'])
                log_data.append(evaluation_results[agent]['tardiness_max_mean'])
                log_data.append(evaluation_results[agent]['makespan_mean'])
                log_data.append(evaluation_results[agent]['rew_std'])
                log_data.append(evaluation_results[agent]['tardiness_std'])
                log_data.append(evaluation_results[agent]['gap_to_solver'])
                final_evaluation_table.add_data(*log_data)
            self.wandb_run.log({'Final Evaluation Table': final_evaluation_table})
