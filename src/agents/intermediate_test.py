"""
This file provides the IntermediateTest class which is used to run an intermediate test on the current model policy.
If the recent model has the best result it is saved as the new current optimum
"""
from typing import List

from src.agents.test import test_model
from src.utils.logger import Logger
from src.data_generator.task import Task
from src.utils.file_handler.model_handler import ModelHandler


class IntermediateTest:
    """
    This object is used to run an intermediate test on the current model policy.
    If the recent model has the best result it is saved as the new current optimum

    :param env_config: Config used to initialize the environment for training
    :param n_test_steps: Number of environment steps between intermediate tests
    :param data: Dataset with instances to be used for the intermediate test
    :param logger: Logger object

    """
    def __init__(self, env_config: dict, n_test_steps: int, data: List[List[Task]], logger: Logger):
        # callback and model
        self.n_test_steps = n_test_steps  # steps between callbacks
        self.last_time_trigger = 0
        self.optimum_rwd = 0  # best reward
        self.results = []  # mean rewards fom all tests during training
        self.file = env_config.get('saved_model_name')
        self.logger = logger
        self.env_config = env_config

        # test/Env parameter
        self.env_config = env_config
        self.data_test = data

    def on_step(self, num_timesteps: int, instances: int, model) -> None:
        """
        This function is called by the environment during each step.
        According to n_test_steps the function runs an intermediate test

        :param num_timesteps: Number of steps that have been already run by the environment
        :param instances: Number of instances that have been already run by the environment
        :param model: Model with the current policy. E.g. PPO object

        :return: None

        """
        if (num_timesteps - self.last_time_trigger) >= self.n_test_steps:
            print('#'*5)
            print('#'*5, f'INTERMEDIATE TEST after step {num_timesteps} and after {instances} instances... ')
            print('#' * 5)

            # safe recent model as comparison for the current optimum
            compare_path = ModelHandler.get_compare_path(self.env_config)
            if not compare_path.exists():
                compare_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(compare_path)

            # change model name in config to compare -> We want that test loads our recent model
            self.env_config.update({"saved_model_name": 'compare'})

            # test current model
            evaluation_results = test_model(self.env_config, self.data_test, model=model, logger=self.logger,
                                            plot=False, log_episode=False, intermediate_test_idx=num_timesteps)

            # change name back in wandb config
            self.env_config.update({"saved_model_name": self.file})

            rwd_mean = evaluation_results['rew_mean']
            tardiness_mean = evaluation_results['tardiness_mean']
            makespan_mean = evaluation_results['makespan_mean']

            # log results
            self.logger.record(
                {
                    'interm_test/mean_reward': rwd_mean,
                    'interm_test/mean_tardiness': tardiness_mean,
                    'interm_test/mean_makespan': makespan_mean
                }
            )
            self.logger.dump()

            # if first test or reward >= current optimum, reset optimum and save model
            if self.last_time_trigger == 0 or rwd_mean >= self.optimum_rwd:
                self.optimum_rwd = rwd_mean
                best_model_save_path = ModelHandler.get_best_model_path(self.env_config)
                model.save(best_model_save_path)
            self.last_time_trigger = num_timesteps
