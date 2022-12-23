import argparse
import warnings
from typing import List

from src.data_generator.task import Task
from src.utils.file_handler.config_handler import ConfigHandler
from src.utils.file_handler.data_handler import DataHandler
from src.environments.environment_loader import EnvironmentLoader


# ray (rllib) check
_rllib = False
try:
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.registry import register_env
    from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
    from ray.tune import run
    _rllib = True
except ModuleNotFoundError:
    warnings.warn('ray/rllib option not available as module not found. Check your installation.')
    pass


def train(trainer):
    result = trainer.train()
    print(result)


def test(trainer, env):
    state = env.reset()
    done = False
    cumulative_reward = 0
    while not done:
        action = trainer.compute_single_action(state)
        state, reward, done, _ = env.step(action)
        cumulative_reward += reward
    print(cumulative_reward)


def save(trainer):
    checkpoint_path = trainer.save()
    print("Saved at: "+checkpoint_path)


def load(config, env, path):
    trainer = PPOTrainer(config=config, env=env)
    trainer = trainer.restore(path)
    return trainer


def load_config_and_data_from_paths(config_path_rel=None, data_path_rel=None):
    # Load config and data
    # Note: Turn on schema check again once a schema has been found.
    config: dict = ConfigHandler.get_config(config_path_rel, check_against_schema=False)
    if data_path_rel is None:
        data_path_rel = config.get("instances_file", None)
    data: List[List[Task]] = DataHandler.load_instances_data_file(
        config=config,
        instances_data_file_path=data_path_rel
    )
    return config, data


def main(config_path_rel=None, data_path_rel=None, agent_path_rel=None):
    config, data = load_config_and_data_from_paths(config_path_rel, data_path_rel)
    # Create env (Create env from given config+data)
    env, env_name = EnvironmentLoader.load(
        config=config,
        data=data,
        check_env_agent_compatibility=False,
        register_gym_env=True)
    # Create/Load model (Create model from config or load from given path)
    if agent_path_rel is None:
        trainer = PPOTrainer(config=config.get('rllib', None))
    else:
        trainer = load(config, env, agent_path_rel)
    # if requested: Train (Given model on given env)
    train(trainer)
    # if requested: Test (Given model on given env)
    test(trainer, env)
    # if requested: Save
    save(trainer)


def get_parser_args():
    parser = argparse.ArgumentParser(description='RLlib interface to scheduling environments.')
    parser.add_argument('-conf', '--config_file_path', type=str, required=False,
                        help='Relative path to config file you want to use for rllib') # TODO set required
    parser.add_argument('-data', '--data_file_path', type=str, required=False,
                        help='Relative path to data file you want to use for rllib')
    parser.add_argument('-agent', '--agent_file_path', type=str, required=False,
                        help='Relative path to agent file you want to load')
    parser.add_argument('-train', '--training', type=bool, required=False,
                        help='Activate train modus')
    parser.add_argument('-test', '--testing', type=bool, required=False,
                        help='Activate test modus')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Get terminal inputs
    parse_args = get_parser_args()
    config_path_relative = parse_args.config_file_path
    data_path_relative = parse_args.data_file_path
    agent_path_relative = parse_args.agent_file_path

    main(
        config_path_rel=config_path_relative,
        data_path_rel=data_path_relative,
        agent_path_rel=agent_path_relative
    )
