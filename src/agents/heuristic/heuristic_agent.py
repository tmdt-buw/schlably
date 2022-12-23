"""
This module provides the following scheduling heuristics as function:

- EDD: earliest due date
- SPT: shortest processing time first
- MTR: most tasks remaining
- LTR: least tasks remaining
- Random: random action

You can implement additional heuristics in this file by specifying a function that takes a list of tasks and an action
mask and returns the index of the job to be scheduled next.

If you want to call your heuristic via the HeuristicSelectionAgent or edit an existing shortcut,
adapt/extend the task_selection dict attribute of the HeuristicSelectionAgent class.

:Example:

Add a heuristic that returns zeros (this is not a practical example!)
1. Define the according function

.. code-block:: python

    def return_0_heuristic(tasks: List[Task], action_mask: np.array) -> int:
        return 0

2. Add the function to the task_selection dict within the HeuristicSelectionAgent class:

.. code-block:: python

    self.task_selections = {
        'rand': random_task,
        'EDD': edd,
        'SPT': spt,
        'MTR': mtr,
        'LTR': ltr,
        'ZERO': return_0_heuristic
    }

"""
import numpy as np
from typing import List

from src.data_generator.task import Task


def get_active_task_dict(tasks: List[Task]) -> dict:
    """
    Helper function to determining the next unfinished task to be processed for each job

    :param tasks: List of task objects, so one instance

    :return: Dictionary containing the next tasks to be processed for each job

    Would be an empty dictionary if all tasks were completed

    """
    active_job_task_dict = {}
    for task_i, task in enumerate(tasks):
        if not task.done and task.job_index not in active_job_task_dict.keys():
            active_job_task_dict[task.job_index] = task_i
    return active_job_task_dict


def edd(tasks: List[Task], action_mask: np.array) -> int:
    """
    EDD: earliest due date. Determines the job with the smallest deadline

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        num_jobs = action_mask.shape[0] - 1
        num_tasks_per_job = len(tasks) / num_jobs
        deadlines = np.full(num_jobs + 1, np.inf)

        for job_i in range(num_jobs):
            idx = int(num_tasks_per_job * job_i)
            deadlines[job_i] = tasks[idx].deadline

        deadlines = np.where(action_mask == 1, deadlines, np.full(deadlines.shape, np.inf))
        chosen_job = np.argmin(deadlines)
    return chosen_job


def spt(tasks: List[Task], action_mask: np.array) -> int:
    """
    SPT: shortest processing time first. Determines the job of which the next unfinished task has the lowest runtime

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        num_jobs = action_mask.shape[0] - 1
        runtimes = np.full(num_jobs + 1, np.inf)
        active_task_dict = get_active_task_dict(tasks)

        for i in range(num_jobs):
            if i in active_task_dict.keys():
                task_idx = active_task_dict[i]
                runtimes[i] = tasks[task_idx].runtime
        runtimes = np.where(action_mask == 1, runtimes, np.full(runtimes.shape, np.inf))
        chosen_job = np.argmin(runtimes)
    return chosen_job


def mtr(tasks: List[Task], action_mask: np.array) -> int:
    """
    MTR: most tasks remaining. Determines the job with the least completed tasks

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        tasks_done = np.zeros(len(tasks) + 1)
        possible_tasks = get_active_task_dict(tasks)
        for _, task in enumerate(tasks):
            if task.done and task.job_index in possible_tasks.keys():
                tasks_done[possible_tasks[task.job_index]] += 1

        task_mask = np.zeros(len(tasks) + 1)
        for job_id, task_id in possible_tasks.items():
            if action_mask[job_id] == 1:
                task_mask[task_id] += 1
        tasks_done = np.where(task_mask == 1, tasks_done, np.full(tasks_done.shape, np.inf))
        tasks_done[-1] = np.inf
        chosen_task = np.argmin(tasks_done)
        chosen_job = tasks[chosen_task].job_index
    return chosen_job


def ltr(tasks: List[Task], action_mask: np.array) -> int:
    """
    LTR: least tasks remaining. Determines the job with the most completed tasks

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        tasks_done = np.zeros(len(tasks) + 1)
        possible_tasks = get_active_task_dict(tasks)
        for _, task in enumerate(tasks):
            if task.done and task.job_index in possible_tasks.keys():
                tasks_done[possible_tasks[task.job_index]] += 1
        task_mask = np.zeros(len(tasks) + 1)
        for job_id, task_id in possible_tasks.items():
            if action_mask[job_id] == 1:
                task_mask[task_id] += 1
        tasks_done = np.where(task_mask == 1, tasks_done, np.full(tasks_done.shape, -1))
        tasks_done[-1] = -1
        chosen_task = np.argmax(tasks_done)
        chosen_job = tasks[chosen_task].job_index
    return chosen_job


def random_task(tasks: List[Task], action_mask: np.array) -> int:
    """
    Returns a random task

    :param tasks: Not needed
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """

    chosen_job = None
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        valid_values_0 = np.where(action_mask > 0)[0]

        if len(valid_values_0) > 2:
            chosen_job = np.random.choice(valid_values_0, size=1)[0]
        elif len(valid_values_0) == 0:
            print('this is not possible')
        else:
            chosen_job = np.random.choice(valid_values_0, size=1)[0]
    return chosen_job


def choose_random_machine(chosen_task, machine_mask) -> int:
    """
    Determines a random machine which is available according to the mask and chosen task. Useful for the FJSSP.

    :param chosen_task: ID of the task that is scheduled on the selected machine
    :param machine_mask: Machine mask from the environment that is to receive the machine action chosen by this function

    :return: Index of the chosen machine

    """
    machine_mask = np.array(np.where(machine_mask > 0))
    idx_valid_machine = np.where(machine_mask[0] == chosen_task)
    valid_machines = machine_mask[1][idx_valid_machine]
    chosen_machine = np.random.choice(valid_machines, size=1)[0]
    return chosen_machine


def choose_first_machine(chosen_task, machine_mask) -> int:
    """
    Determines the first (by index) machine which is available according to the mask and chosen task. Useful for the
    FJSSP

    :param chosen_task: ID of the task that is scheduled on the selected machine
    :param machine_mask: Machine mask from the environment that is to receive the machine action chosen by this function

    :return: Index of the chosen machine

    """
    machine_mask = np.array(np.where(machine_mask > 0))
    idx_valid_machine = np.where(machine_mask[0] == chosen_task)
    valid_machines = machine_mask[1][idx_valid_machine]
    return valid_machines[0]


class HeuristicSelectionAgent:
    """
    This class can be used to get the next task according to the heuristic passed as string abbreviation (e.g. EDD).
    If you want to edit a shortcut, or add one for your custom heuristic, adapt/extend the task_selection dict.

    :Example:

    .. code-block:: python

        def my_custom_heuristic():
            ...<function body>...

    or

    .. code-block:: python

        self.task_selections = {
            'rand': random_task,
            'XYZ': my_custom_heuristic
            }

    """

    def __init__(self) -> None:

        super().__init__()
        # Map heuristic ids to corresponding function
        self.task_selections = {
            'rand': random_task,
            'EDD': edd,
            'SPT': spt,
            'MTR': mtr,
            'LTR': ltr
        }

    def __call__(self, tasks: List, action_mask: np.array, task_selection: str) -> int:
        """
        Selects the next heuristic function according to the heuristic passed as string abbreviation
        and the assignment in the task_selections dictionary

        :param tasks: List of task objects, so one instance
        :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic
        :param task_selection: Heuristic string abbreviation (e.g. EDD)

        :return: Index of the job selected according to the heuristic

        """
        choose_task = self.task_selections[task_selection]

        chosen_task = choose_task(tasks, action_mask)

        return chosen_task
