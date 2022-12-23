"""
This file provides the test_model function to evaluate an agent or a heuristic on a set of instances.
Furthermore, test_model_and_heuristics can be used to evaluate an agent and all heuristics specified in the
TEST_HEURISTICS constant on the same set of instances.

Using this file requires a testing config. For example, it is necessary to specify the name of the model
you want to test.

Running this file will automatically call test_model_and_heuristics.
You can adapt the heuristics used for testing in the TEST_HEURISTICS constant. An empty list is admissible.

When running the file from a console you can use --plot-ganttchart to show the generated gantt_chart figures.
"""
import argparse
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict, Union
import numpy as np
from tqdm import tqdm

from src.environments.environment_loader import EnvironmentLoader
from src.agents.heuristic.heuristic_agent import HeuristicSelectionAgent
from src.utils.evaluations import EvaluationHandler
from src.utils.logger import Logger
from src.utils.file_handler.data_handler import DataHandler
from src.utils.file_handler.model_handler import ModelHandler
from src.data_generator.task import Task
from src.agents.train_test_utility_functions import get_agent_class_from_config, load_config, load_data
from src.agents.solver import OrToolSolver

# constants
TEST_HEURISTICS: List[str] = ['rand', 'EDD', 'SPT', 'MTR', 'LTR']


def get_action(env, model, heuristic_id: str, heuristic_agent: Union[HeuristicSelectionAgent, None]) -> Tuple[int, str]:
    """
    This function determines the next action according to the input model or heuristic

    :param env: Environment object
    :param model: Model object. E.g. PPO object
    :param heuristic_id: Heuristic identifier. Can be None
    :param heuristic_agent: HeuristicSelectionAgent object. Can be None

    :return: ID of the selected action

    """
    assert bool(heuristic_id) != bool(model), \
        "You have to pass an agent model XOR a heuristic id to solve the scheduling problem"
    obs = env.state_obs
    mask = env.get_action_mask()

    if heuristic_id:
        action_mode = 'heuristic'
        tasks = env.tasks
        task_mask = mask
        selected_action = heuristic_agent(tasks, task_mask, heuristic_id)
    else:
        action_mode = 'agent'
        selected_action, _ = model.predict(observation=obs, action_mask=mask)

    return selected_action, action_mode


def run_episode(env, model, heuristic_id: Union[str, None], handler: EvaluationHandler) -> None:
    """
    This function executes one testing episode

    :param env: Environment object
    :param model: Model object. E.g. PPO object
    :param heuristic_id: Heuristic identifier. Can be None
    :param handler: EvaluationHandler object

    :return: None

    """
    done = False
    total_reward = 0

    heuristic_agent = HeuristicSelectionAgent() if heuristic_id else None

    # run agent on environment and collect rewards until done
    steps = 0
    while not done:
        steps += 1
        action, action_mode = get_action(env, model, heuristic_id, heuristic_agent)

        b = env.step(action, action_mode=action_mode)
        total_reward += b[1]
        done = b[2]

    # store episode in object
    mean_reward = total_reward / steps
    handler.record_environment_episode(env, mean_reward)


def test_solver(config: Dict, data_test: List[List[Task]], logger: Logger) -> Dict:
    """
    This function uses the OR solver to schedule the instances given in data_test.

    :param config: Testing config
    :param data_test: Data containing problem instances used for testing

    :return: Evaluation metrics

    """
    eval_handler = EvaluationHandler()

    # for each test instance
    for instance in tqdm(data_test, desc='Computing solver solution if necessary'):

        # get instance_hash from first task object
        instance_hash = instance[0].instance_hash
        # try to load solved_instance by hash
        solved_instance = DataHandler.load_solved_instance_by_hash(instance_hash)
        # if no solution exists, compute the solution and write it to file for futures use
        if solved_instance is None:
            assigned_jobs, _ = OrToolSolver.optimize(instance, objective='makespan')
            solved_instance = OrToolSolver.parse_to_plottable_format(instance, assigned_jobs)
            # write solution to file
            DataHandler.write_solved_instance_by_hash(solved_instance, instance_hash)

        # create environment and assign the solved_instance as tasks. Necessary to use the env for evaluation
        env, _ = EnvironmentLoader.load(config, data=data_test)
        env.tasks = solved_instance
        eval_handler.update_episode_solved_with_solver(env)
        log_results(plot_logger=logger, inter_test_idx=None, heuristic='solver', env=env, handler=eval_handler)

    return eval_handler.evaluate_test()


def log_results(plot_logger: Logger, inter_test_idx: Union[int, None], heuristic: str,
                env, handler: EvaluationHandler) -> None:
    """
    Calls the logger object to save the test results from this episode as table (e.g. makespan mean, gantt chart)

    :param plot_logger: Logger object
    :param inter_test_idx: Index of current test. Can be None
    :param heuristic: Heuristic identifier. Can be None
    :param env: Environment object
    :param handler: EvaluationHandler object

    :return: None

    """

    # get recent measures for the table
    measures = {'total_reward': handler.rewards[-1], 'makespan': handler.makespan[-1],
                'tardiness': handler.tardiness[-1]}

    gantt_chart = env.render(mode="image")
    # Log chart as table
    if heuristic:
        plot_logger.add_row_to_wandb_table(heuristic, gantt_chart, **measures)
    else:
        if inter_test_idx is None:
            plot_logger.add_row_to_wandb_table("RL-Agent", gantt_chart, **measures)
        else:
            plot_logger.add_row_to_wandb_table(f"RL-Agent {inter_test_idx}", gantt_chart, **measures)


def test_model(env_config: Dict, data: List[List[Task]], logger: Logger, plot: bool = None, log_episode: bool = None,
               model=None, heuristic_id: str = None, intermediate_test_idx=None) -> dict:
    """
    This function tests a model in the passed environment for all problem instances passed as data_test and returns an
    evaluation summary

    :param env_config: Environment config
    :param data: Data containing problem instances used for testing
    :param logger: Logger object
    :param plot: Plot a gantt chart of all tests
    :param log_episode: If true, calls the log function to log episode results as table
    :param model: {None, StableBaselines Model}
    :param heuristic_id: ID that identifies the used heuristic
    :param intermediate_test_idx: Step number after which the test is performed. Is used to annotate the log

    :return: evaluation metrics

    """

    # create evaluation handler
    evaluation_handler = EvaluationHandler()

    for test_i in range(len(data)):

        # create env
        environment, _ = EnvironmentLoader.load(env_config, data=[data[test_i]])
        environment.runs = test_i

        # run environment episode
        run_episode(environment, model, heuristic_id, evaluation_handler)

        # log results. Creating wandb table
        if log_episode:
            log_results(logger, intermediate_test_idx, heuristic_id, environment, evaluation_handler)

        # plot results
        if plot:
            environment.render()

    # return episode results, using EvaluationHandler properties and function
    return evaluation_handler.evaluate_test()


def test_model_and_heuristic(config: dict, model, data_test: List[List[Task]], logger: Logger,
                             plot_ganttchart: bool = False, log_episode: bool = False) -> dict:
    """
    Test model and agent_heuristics len(data) times and returns results

    :param config: Testing config
    :param model: Model to be tested. E.g. PPO object
    :param data_test: Dataset with instances to be used for the test
    :param logger: Logger object
    :param plot_ganttchart: Plot a gantt chart of all tests
    :param log_episode: If true, calls the log function to log episode results as table

    :return: Dict with evaluation_result dicts for the agent and all heuristics which were tested

    """
    print('Testing model, heuristics and solver... ')
    results = {}

    test_kwargs = {'env_config': config, 'data': data_test, 'logger': logger,
                   'plot': plot_ganttchart, 'log_episode': log_episode}

    # test agent
    res = test_model(model=model, **test_kwargs)
    results.update({'agent': res})

    # test heuristics
    for heuristic in config.get('test_heuristics', TEST_HEURISTICS):
        res = test_model(heuristic_id=heuristic, **test_kwargs)
        results.update({heuristic: res})

    # test solver and calculate optimality gap
    res = test_solver(config, data_test, logger)
    results.update({'solver': res})

    results = EvaluationHandler.add_solver_gap_to_results(results)

    return results


def get_perser_args():
    # Arguments for function
    parser = argparse.ArgumentParser(description='Test Agent in Production Scheduling Environment')

    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')
    parser.add_argument('-plot', '--plot-ganttchart', dest="plot_ganttchart", action="store_true",
                        help='Enable or disable model result plot.')

    args = parser.parse_args()

    return args


def main(external_config=None):

    # get config_file from terminal input
    parse_args = get_perser_args()
    config_file_path = parse_args.config_file_path

    # get config and data
    config = load_config(config_file_path, external_config)
    data = load_data(config)

    # Random seed for numpy as given by config
    np.random.seed(config['seed'])

    # get model path from config
    best_model_path = ModelHandler.get_best_model_path(config)

    # create logger
    logger = Logger(config=config)
    model = get_agent_class_from_config(config=config).load(file=best_model_path, config=config, logger=logger)
    results = test_model_and_heuristic(config=config, model=model, data_test=data,
                                       plot_ganttchart=parse_args.plot_ganttchart, logger=logger)
    print(results)
    plt.show()


if __name__ == '__main__':

    main()
