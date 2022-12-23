"""
This file provides the scheduling environment class Env,
which can be used to load and simulate scheduling-problem instances.
"""
import gym
import numpy
from gym import spaces
import numpy as np
import copy
from src.data_generator.task import Task
from src.visuals_generator.gantt_chart import GanttChartPlotter
from typing import List, Tuple, Dict, Any, Union

REWARD_BUFFER_SIZE = 250


class Env(gym.Env):
    """
    Environment for scheduling optimization.
    This class inherits from the base gym environment, so the functions step, reset, _state_obs and render
    are implemented and can be used by default.

    If you want to customize the given rewards, you can adapt the function compute_reward.

    :param config: Dictionary with parameters to specify environment attributes
    :param data: Scheduling problem to be solved, so a list of instances

    """

    def __init__(self, config: dict, data: List[List[Task]]):

        super(Env, self).__init__()

        # import data containing all instances
        self.data: List[List[Task]] = data  # is later shuffled before input into the environment

        # get number of jobs, tasks, tools, machines and runtimes from input data
        self.num_jobs, self.num_tasks, self.max_runtime, self.max_deadline = self.get_instance_info()
        self.num_machines: int = copy.copy(self.data[0][0]._n_machines)
        self.num_tools: int = copy.copy(self.data[0][0]._n_tools)
        self.num_all_tasks: int = self.num_jobs * self.num_tasks
        self.num_steps_max: int = config.get('num_steps_max', self.num_all_tasks)
        self.max_task_index: int = self.num_tasks - 1
        self.max_job_index: int = self.num_jobs - 1

        # retrieve run-dependent settings from config
        self.shuffle: bool = config.get('shuffle', False)
        self.log_interval: int = config.get('log_interval', 10)

        # initialize info which is reset by the reset-method after every episode
        self.num_steps: int = 0
        self.makespan: int = 0
        self.tardiness: numpy.ndarray = np.zeros(self.num_all_tasks, dtype=int)
        self.ends_of_machine_occupancies: numpy.ndarray = np.zeros(self.num_machines, dtype=int)
        self.tool_occupancies: List[List] = [[] for _ in range(self.num_tools)]  # stores tool use intervals
        self.job_task_state: numpy.ndarray = np.zeros(self.num_jobs, dtype=int)
        self.task_job_mapping: dict = {}

        # initialize info which is not reset
        self.runs: int = -2  # counts runs (episodes/dones).  -1 because reset is called twice before start
        self.last_mask: numpy.ndarray = np.zeros(self.num_jobs)
        self.tasks: List[Task] = []
        self.data_idx: int = 0
        self.iterations_over_data = -1

        # training info log updated after each "epoch" over all training data
        self.action_history: List = []  # stores the sequence of tasks taken
        self.executed_job_history: List = []  # stores the sequence of jobs, of which the task is scheduled
        self.reward_history: List = []  # stores the rewards
        self.episodes_rewards: List = []
        self.episodes_makespans: List = []
        self.episodes_tardinesses: List = []

        # logging info buffers. Are reset in self.log_intermediate_step
        self.logging_makespans: List = []
        self.logging_rewards: List = []
        self.logging_tardinesses: List = []

        # action_space: idx_job
        self.action_space: spaces.Discrete = spaces.Discrete(self.num_jobs)

        # initial observation
        self._state_obs: List = self.reset()

        # observation space
        observation_shape = np.array(self.state_obs).shape
        self.observation_space: spaces.Box = spaces.Box(low=0, high=1, shape=observation_shape)

        # reward parameters
        self.reward_strategy = config.get('reward_strategy', 'dense_makespan_reward')
        self.reward_scale = config.get('reward_scale', 1)
        self.mr2_reward_buffer: List[List] = [[] for _ in range(len(data))]  # needed for m2r reward only

    def reset(self) -> List[float]:
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

    def step(self, action: Union[int, float], **kwargs) -> (List[float], Any, bool, Dict):
        """
        Step Function
        :param action: Action to be performed on the current state of the environment
        :return: Observation, reward, done, infos
        """
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

    def get_instance_info(self) -> (int, int, int, int):
        """
        Retrieves info about the instance size and configuration from an instance sample
        :return: (number of jobs, number of tasks and the maximum runtime) of this datapoint
        """
        num_jobs, num_tasks, max_runtime, max_deadline = 0, 0, 0, 0
        for task in self.data[0]:
            num_jobs = task.job_index if task.job_index > num_jobs else num_jobs
            num_tasks = task.task_index if task.task_index > num_tasks else num_tasks
            max_runtime = task.runtime if task.runtime > max_runtime else max_runtime
            max_deadline = task.deadline if task.deadline > max_deadline else max_deadline
        return num_jobs + 1, num_tasks + 1, max_runtime, max_deadline

    @property
    def state_obs(self) -> List[float]:
        """
        Transforms state (task state and factory state) to gym obs
        Scales the values between 0-1 and transforms to onehot encoding

        :return: Observation

        """

        obs = []
        # assemble information on next tasks - note that the observation is ordered by job id!
        for job in np.arange(self.num_jobs):
            t_idx = self.job_task_state[job] if self.job_task_state[job] < self.max_task_index else self.max_task_index
            next_task_in_job = copy.copy(self.tasks[self.task_job_mapping[job, t_idx]])

            obs.append(next_task_in_job.runtime / self.max_runtime)
            obs.append(next_task_in_job.task_index / (self.num_tasks + 1))
            obs.append(next_task_in_job.deadline / (self.max_deadline + 1))

        # obs = np.zeros(self.observation_space.shape[0])  # overwrite with dummy

        self._state_obs = obs
        # print('obs| ', self._state_obs)
        return self._state_obs

    @staticmethod
    def to_one_hot(x: int, max_size: int) -> np.array:
        """
        Convert to One Hot encoding

        :param x: Index which value should be 1
        :param max_size: Size of the one hot encoding vector

        :return: One hot encoded vector

        """
        one_hot = np.zeros(max_size)
        one_hot[x] = 1
        return one_hot

    @staticmethod
    def check_valid_job_action(job_action: np.array, job_mask: np.array) -> bool:
        """
        Check if job action is valid

        :param job_action: Job action as one hot vector
        :param job_mask: One hot vector with ones for each valid job

        :return: True if job_action is valid, else False

        """
        return np.sum(job_action == job_mask) >= 1

    def get_selected_task(self, job_idx: int) -> Tuple[int, Task]:
        """
        Helper Function to get the selected task (next possible task) only by the job index

        :param job_idx: job index

        :return: Index of the task in the task list and the selected task

        """
        task_idx = self.task_job_mapping[(job_idx, self.job_task_state[job_idx])]
        selected_task = self.tasks[task_idx]
        return task_idx, selected_task

    def choose_machine(self, task: Task) -> int:
        """
        This function performs the logic, with which the machine is chosen (in the case of the flexible JSSP)
        Implemented at the moment: Choose the machine out of the set of possible machines with the earliest possible
        start time

        :param task: Task

        :return: Machine on which the task will be scheduled.

        """
        possible_machines = task.machines
        machine_times = np.where(possible_machines,
                                 self.ends_of_machine_occupancies,
                                 np.full(len(possible_machines), np.inf))
        return int(np.argmin(machine_times))

    def get_action_mask(self) -> np.array:
        """
        Get Action mask
        It is needed for the heuristics, the machine selection (and the agent, if it is masked).
        0 -> available
        1 -> not available

        :return: Action mask

        """
        job_mask = np.where(self.job_task_state < self.num_tasks,
                            np.ones(self.num_jobs, dtype=int), np.zeros(self.num_jobs, dtype=int))

        self.last_mask = job_mask
        return job_mask

    def execute_action(self, job_id: int, task: Task, machine_id: int) -> None:
        """
        This Function executes a valid action
        - set machine
        - update job and task

        :param job_id: job_id of the task to be executed
        :param task: Task
        :param machine_id: ID of the machine on which the task is to be executed

        :return: None

        """
        # check task preceding in the job (if it is not the first task within the job)
        if task.task_index == 0:
            start_time_of_preceding_task = 0
        else:
            preceding_task = self.tasks[self.task_job_mapping[(job_id, task.task_index - 1)]]
            start_time_of_preceding_task = preceding_task.finished

        # check earliest possible time to schedule according to preceding task and needed machine
        start_time = max(start_time_of_preceding_task, self.ends_of_machine_occupancies[machine_id])

        # if the task needs tools, these need to be taken into account, too
        if self.num_tools != 0:
            # only searches within the ends of machine occupancies
            search_min = min(self.ends_of_machine_occupancies)
            search_max = max(self.ends_of_machine_occupancies)

            # create numpy arrays of open windows with respect to tools
            occupied_matrix = np.zeros((len(self.tool_occupancies), search_max))
            for tool, tool_intervals in enumerate(self.tool_occupancies):
                for interval in tool_intervals:
                    occupied_matrix[tool, interval[0]:interval[1]] = 1

            # get a representation of where open slots are for the needed tools
            tool_occupation = np.sum(occupied_matrix[np.array(task.tools, dtype=bool), :], axis=0).astype('int')
            possible_start_times = []
            for time in np.arange(search_min, search_max):
                if all(tool_occupation[int(time):int(time + task.runtime)] == 0):
                    if time >= start_time:
                        possible_start_times.append(int(time))
            if len(possible_start_times) == 0:
                if sum(tool_occupation) > 0:
                    possible_start_times.append(np.max(np.argwhere(tool_occupation == 1)) + 1)
                else:
                    possible_start_times.append(0)
            min_possible_start_time = min(possible_start_times)
            if min_possible_start_time > start_time:
                start_time = min_possible_start_time

        end_time = start_time + task.runtime

        # update machine occupancy and job_task_state
        self.ends_of_machine_occupancies[machine_id] = end_time
        self.job_task_state[job_id] += 1
        for needed_tool in np.argwhere(task.tools):
            self.tool_occupancies[int(needed_tool[0])].append([start_time, end_time])

        # update job and task
        task.started = start_time
        task.finished = end_time
        task.selected_machine = machine_id
        task.done = True

    def compute_reward(self) -> Any:
        """
        Calculates the reward that will later be returned to the agent. Uses the self.reward_strategy string to
        discriminate between different reward strategies. Default is 'dense_reward'.

        :return: Reward

        """
        if self.reward_strategy == 'dense_makespan_reward':
            # dense reward for makespan optimization according to https://arxiv.org/pdf/2010.12367.pdf
            reward = self.makespan - self.get_makespan()
            self.makespan = self.get_makespan()
        elif self.reward_strategy == 'sparse_makespan_reward':
            reward = self.sparse_makespan_reward()
        elif self.reward_strategy == 'mr2_reward':
            reward = self.mr2_reward()
        else:
            raise NotImplementedError(f'The reward strategy {self.reward_strategy} has not been implemented.')

        reward *= self.reward_scale

        return reward

    def sparse_makespan_reward(self) -> int:
        """
        Computes the reward based on the final makespan at the end of the episode. Else 0.

        :return: (int) sparse reward

        """
        if not self.check_done():
            reward = 0
        else:
            reward = self.get_makespan()

        return reward

    def mr2_reward(self) -> Any:
        """
        Computes mr2 reward based on https://doi.org/10.1016/j.engappai.2022.104868

        :return: mr2 reward

        """
        if not self.check_done():
            reward = 0
        else:
            last_makespan = self.get_makespan()
            if len(self.mr2_reward_buffer[self.data_idx]) > 0:

                percentile_to_beat = np.percentile(np.array(self.mr2_reward_buffer[self.data_idx]), 70)

                if last_makespan > percentile_to_beat:
                    reward = -1
                elif last_makespan < percentile_to_beat:
                    reward = 1
                else:
                    if np.random.rand() < 0.1:
                        reward = 1
                    else:
                        reward = -1
            else:
                reward = 0

            self.mr2_reward_buffer[self.data_idx].append(last_makespan)
            if len(self.mr2_reward_buffer[self.data_idx]) > REWARD_BUFFER_SIZE:  # pop from left side to update buffer
                self.mr2_reward_buffer[self.data_idx].pop(0)

            if self.runs > 100:
                # TODO What is this?
                print('stop')

        return reward

    def check_done(self) -> bool:
        """
        Check if all jobs are done

        :return: True if all jobs are done, else False

        """
        sum_done = sum([task.done for task in self.tasks])
        return sum_done == self.num_all_tasks or self.num_steps == self.num_steps_max

    def calculate_tardiness(self) -> int:
        """
        Calculates the tardiness of all jobs
        (this is the previous was the calc reward function)

        :return: (int) tardiness of last solution

        """
        for i, task in enumerate(self.tasks):
            # if task has not the highest index -> continue

            if task.task_index == self.max_task_index:
                # if job=last task of the job done, punish possible completion
                # after deadline -> 0 if done before deadline
                if task.done == 1:
                    t_tardiness = int(np.maximum(0, task.finished - task.deadline))
                    self.tardiness[i] = t_tardiness
                # if job not done yet, punish with max tardiness punishment
                else:
                    t_tardiness = int(np.maximum(0, self.num_steps_max - task.deadline))
                    self.tardiness[i] = t_tardiness

        return t_tardiness

    def get_makespan(self):
        """
        Returns the current makespan (the time the latest of all scheduled tasks finishes)
        """
        return np.max(self.ends_of_machine_occupancies)

    def log_intermediate_step(self) -> None:
        """
        Log Function

        :return: None

        """
        if self.runs >= self.log_interval:
            print('-' * 110, f'\n{self.runs} instances played! Last instance seen: {self.data_idx}/{len(self.data)}')
            print(f'Average performance since last log: mean reward={np.around(np.mean(self.logging_rewards), 2)}, ' \
                     f'mean makespan={np.around(np.mean(self.logging_makespans), 2)}, ' \
                     f'mean tardiness={np.around(np.mean(self.logging_tardinesses), 2)}')
            self.logging_rewards.clear()
            self.logging_makespans.clear()
            self.logging_tardinesses.clear()

    def close(self):
        """
        This is a relict of using OpenAI Gym API. This is currently unnecessary.
        """
        pass

    def seed(self, seed=1):
        """
        This is a relict of using OpenAI Gym API.
        Currently unnecessary, because the environment is deterministic -> no seed is used.
        """
        return seed

    def render(self, mode='human'):
        """
        Visualizes the current status of the environment

        :param mode: "human": Displays the gantt chart,
                     "image": Returns an image of the gantt chart

        :return: PIL.Image.Image if mode=image, else None

        """
        if mode == 'human':
            GanttChartPlotter.get_gantt_chart_image(self.tasks, show_image=True, return_image=False)
        elif mode == 'image':
            return GanttChartPlotter.get_gantt_chart_image(self.tasks)
        else:
            raise NotImplementedError(f"The Environment on which you called render doesn't support mode: {mode}")
