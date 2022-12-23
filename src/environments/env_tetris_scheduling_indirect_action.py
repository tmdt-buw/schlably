import numpy as np
import copy
from typing import List
from gym import spaces
from numpy import ndarray

from src.data_generator.task import Task
from src.environments.env_tetris_scheduling import Env


class IndirectActionEnv(Env):
    """
    Scheduling environment for scheduling optimization according to
    https://www.sciencedirect.com/science/article/pii/S0952197622001130.

    Main differences to the vanilla environment:

    - ACTION: Indirect action mapping
    - REWARD: m-r2 reward (which means we have to train on the same data again and again)
    - OBSERVATION: observation different ("normalization" looks like division by max to [0, 1] in paper code). Not every
      part makes sense, due to the different interaction logic
    - INTERACTION LOGIC WARNING:
    - original paper: time steps are run through, the agent can take as many actions as it wants per time-step,
      but may not schedule into the past.
    - our adaptation: we still play tetris, meaning that we schedule whole blocks of work at a time

    :param config: Dictionary with parameters to specify environment attributes
    :param data: Scheduling problem to be solved, so a list of instances

    """
    def __init__(self, config: dict, data: List[List[Task]]):

        super(IndirectActionEnv, self).__init__(config, data)

        # overwrite action space to predicting a processing time normalized to [0, 1]
        self.action_space: spaces.Discrete = spaces.Discrete(10)

        # overwrite observation space
        observation_shape = np.array(self.state_obs).shape
        self.observation_space = spaces.Box(low=-1, high=1, shape=observation_shape)

    def step(self, action: int, **kwargs):
        """
        Step Function

        :param action: Action to be performed on the current state of the environment
        :param kwargs: should include "action_mode", because the interaction pattern between heuristics and
            the agent are different and need to be processed differently

        :return: Observation, reward, done, infos

        """
        # check if action_mode was set
        action_mode = 'agent'  # set default, if the action mode is not defined assuming agent is taking it
        if 'action_mode' in kwargs.keys():
            action_mode = kwargs['action_mode']

        if action_mode == 'agent':
            # get selected action via indirect action mapping
            next_tasks = self.get_next_tasks()
            next_runtimes = copy.deepcopy([task.runtime if task is not None else np.inf for task in next_tasks])
            next_runtimes = np.array(next_runtimes) / self.max_runtime
            action = np.argmin(abs(next_runtimes - (action/9)))
        elif action_mode == 'heuristic':
            # action remains the same
            pass

        # transform and store action
        selected_job_vector = self.to_one_hot(action, self.num_jobs)
        self.action_history.append(action)

        # check if the action is valid/executable
        if self.check_valid_job_action(selected_job_vector, self.last_mask):
            # if the action is valid/executable/schedulable
            selected_task_id, selected_task = self.get_selected_task(action)
            selected_machine = self.choose_machine(selected_task)
            self.execute_action(action, selected_task, selected_machine)
        else:
            # if the action is not valid/executable/scheduable
            pass

        # update variables and track reward
        action_mask = self.get_action_mask()
        infos = {'mask': action_mask}
        observation = self.state_obs
        reward = self.compute_reward()
        self.reward_history.append(reward)

        done = self.check_done()
        if done:
            episode_reward_sum = np.sum(self.reward_history)
            makespan = self.get_makespan()
            tardiness = self.calculate_tardiness()

            self.episodes_makespans.append(self.get_makespan())
            self.episodes_rewards.append(np.mean(self.reward_history))

            self.logging_rewards.append(episode_reward_sum)
            self.logging_makespans.append(makespan)
            self.logging_tardinesses.append(tardiness)

            if self.runs % self.log_interval == 0:
                self.log_intermediate_step()
        self.num_steps += 1
        return observation, reward, done, infos

    def reset(self) -> ndarray:
        """
        - Resets the episode information trackers
        - Updates the number of runs
        - Loads new instance

        :return: First observation by calling the class function self.state_obs

        """
        # update runs (episodes passed so far)
        self.runs += 1

        # reset episode counters and infos
        self.num_steps = 0
        self.tardiness = np.zeros(self.num_all_tasks, dtype=int)
        self.makespan = 0
        self.ends_of_machine_occupancies = np.zeros(self.num_machines, dtype=int)
        self.tool_occupancies = [[] for _ in range(self.num_tools)]
        self.job_task_state = np.zeros(self.num_jobs, dtype=int)
        self.action_history = []
        self.executed_job_history = []
        self.reward_history = []

        # clear episode rewards after all training data has passed once. Stores info across runs.
        if self.data_idx == 0:
            self.episodes_makespans, self.episodes_rewards, self.episodes_tardinesses = ([], [], [])
            self.iterations_over_data += 1

        # load new instance every run
        self.data_idx = self.runs % len(self.data)
        self.tasks = copy.deepcopy(self.data[self.data_idx])
        if self.shuffle:
            np.random.shuffle(self.tasks)
        self.task_job_mapping = {(task.job_index, task.task_index): i for i, task in enumerate(self.tasks)}

        # retrieve maximum deadline of the current instance
        max_deadline = max([task.deadline for task in self.tasks])
        self.max_deadline = max_deadline if max_deadline > 0 else 1

        return self.state_obs

    @property
    def state_obs(self) -> ndarray:
        """
        Transforms state (task state and factory state) to gym obs
        Scales the values between 0-1 and transforms to onehot encoding
        Confer https://www.sciencedirect.com/science/article/pii/S0952197622001130 section 4.2.1

        :return: Observation

        """

        # (1) remaining time of operations currently being processed on each machine (not compatible with our offline
        # interaction logic
        # (2) sum of all task processing times still to be processed on each machine
        remaining_processing_times_on_machines = np.zeros(self.num_machines)
        # (3) sum of all task processing times left on each job
        remaining_processing_times_per_job = np.zeros(self.num_jobs)
        # (4) processing time of respective next task on job (-1 if job is done)
        operation_time_of_next_task_per_job = np.zeros(self.num_jobs)
        # (5) machine used for next task (altered for FJJSP compatability to one-hot encoded multibinary representation)
        machines_for_next_task_per_job = np.zeros((self.num_jobs, self.num_machines))
        # (6) time passed at any given moment. Not really applicable to the offline scheduling case.

        # feature assembly
        next_tasks = self.get_next_tasks()
        for task in self.tasks:
            if task.done:
                pass
            if not task.done:
                remaining_processing_times_on_machines[np.argwhere(task.machines)] += task.runtime
                remaining_processing_times_per_job[task.job_index] += task.runtime
                if task.task_index == next_tasks[task.job_index]:  # next task of the job
                    operation_time_of_next_task_per_job[task.job_index] += task.runtime
                    machines_for_next_task_per_job[task.job_index] = task.machines

        # normalization
        remaining_processing_times_on_machines /= (self.num_jobs * self.max_runtime)
        remaining_processing_times_per_job /= (self.num_tasks * self.max_runtime)
        operation_time_of_next_task_per_job /= self.max_runtime

        observation = np.concatenate([
            remaining_processing_times_on_machines,
            remaining_processing_times_per_job,
            operation_time_of_next_task_per_job,
            machines_for_next_task_per_job.flatten()
        ])

        self._state_obs = observation
        return self._state_obs

    def get_action_mask(self) -> np.array:
        """
        Get Action mask
        In this environment, we always treat all actions as valid, because the interaction logic accepts it. Note that
        we only allow non-masked algorithms.
        The heuristics, however, still need the job mask.
        0 -> available
        1 -> not available

        :return: Action mask

        """
        job_mask = np.where(self.job_task_state < self.num_tasks,
                            np.ones(self.num_jobs, dtype=int), np.zeros(self.num_jobs, dtype=int))

        self.last_mask = job_mask
        return job_mask

    def get_next_tasks(self):
        """returns the next tasks that can be scheduled"""
        next_tasks = []
        for job in range(self.num_jobs):
            if self.job_task_state[job] == self.num_tasks:  # means that job is done
                next_tasks.append(None)
            else:
                task_position = self.task_job_mapping[(job, self.job_task_state[job])]
                next_tasks.append(self.tasks[task_position])
        return next_tasks
