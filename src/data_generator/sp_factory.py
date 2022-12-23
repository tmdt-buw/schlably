"""
Helper function for the instance generation in instance_factory.py.
"""
# Standard package import
from enum import Enum
from typing import List, Tuple
import itertools
import random

# Library import
import numpy as np

# Functional internal import
from src.data_generator.task import Task


class SP(Enum):
    jssp = "_generate_instance_jssp"
    fjssp = "_generate_instance_fjssp"

    @classmethod
    def is_sp_type_implemented(cls, sp_type: str = "") -> bool:
        return True if sp_type in cls.str_list_of_sp_types_implemented() else False

    @classmethod
    def str_list_of_sp_types_implemented(cls) -> List[str]:
        return [name for name, _ in cls.__members__.items()]


class SPFactory:

    @classmethod
    def generate_instances(cls, num_jobs: int = 2, num_tasks: int = 2, num_machines: int = 2, num_tools: int = 2,
                        num_instances: int = 2, runtimes: List[int] = None, sp_type: str = "jssp",
                        print_info: bool = False, **kwargs) -> List[List[Task]]:
        """
        Creates a list of instances with random values in the range of the input parameters

        :param num_jobs: number of jobs generated in an instance
        :param num_tasks: number of tasks per job generated in an instance
        :param num_machines: number of machines available
        :param num_tools: number of tools available
        :param num_instances: number of instances which are to be generated
        :param runtimes: list of possible runtimes for tasks
        :param sp_type: Scheduling problem type (e.g. "jssp")
        :param print_info: if True additional info printed to console

        :return: List of list of Task instances which together form an instance

        """

        # Default initialization of mutable parameters - code safety.
        if runtimes is None:
            runtimes = [4, 6]

        # Check if implemented SP type is provided and get matching generate instance function.
        assert SP.is_sp_type_implemented(sp_type), \
            f"{sp_type} is not valid, you have to provide a valid sp type: {SP.str_list_of_sp_types_implemented()}\n"
        generate_instance_function = getattr(cls, SP[sp_type].value)

        # Get possible combinations according to given parameters
        # Machines binary mask - permutations without all 0 element
        machines: List[Tuple] = list(itertools.product([0, 1], repeat=num_machines))[1:]
        # Tools diagonal matrix
        # TODO (minor) - fix typing - preset for else is bad
        tools: np.ndarray = np.eye(num_tools, dtype=int) if num_tools > 0 else [[]]
        # Generate all possible combinations for job and task
        comp_attributes_task: list = [machines, tools, runtimes]
        task_combinations = list(itertools.product(*comp_attributes_task))

        # Collect arguments for generate instance function
        current_kwargs = locals().copy()
        # Remove class argument from collection
        current_kwargs.pop('cls', None)
        # Remove additional kwargs from collection to pass them individually
        current_kwargs.pop('kwargs', None)

        instances = []
        # Create and collect n instances
        for _ in range(num_instances):
            # Call instance function with currently collected arguments
            new_instance = generate_instance_function(**current_kwargs, **kwargs)
            instances.append(new_instance)

        # Print information
        if print_info:
            print('Number of generated task combinations:', len(task_combinations))

        return instances

    @classmethod
    def _generate_instance_jssp(cls, task_combinations: List[Tuple[int]], num_jobs: int, num_tasks: int,
                             num_machines: int, num_tools: int, **kwargs) -> List[Task]:
        """
        Generates a jssp instance

        :param task_combinations: List with all possible tasks
        :param num_jobs: number of jobs generated in an instance
        :param num_tasks: number of tasks per job generated in an instance
        :param num_machines: number of machines available
        :param num_tools: number of tools available
        :param kwargs: Unused

        :return: jssp instance (List of tasks)

        """
        # Initial data_generator sanity check
        assert num_tasks == num_machines, "Warning: You are not creating a classical JSSP instance, " \
                                          "where num_machines = num_tasks must hold."

        instance = []
        # pick n jobs for this instance
        for j in range(num_jobs):
            # Generate random shuffled list of machines for job tasks
            machines_jssp_random = random.sample(list(np.arange(num_tasks)), num_tasks)
            # pick num_tasks tasks for this job
            for t in range(num_tasks):
                task = list(task_combinations[np.random.randint(0, len(task_combinations) - 1)])

                machines_jssp = [0 for _ in np.arange(num_tasks)]
                machines_jssp[machines_jssp_random[t]] = 1
                task[0] = tuple(machines_jssp)

                task = Task(
                    job_index=j,
                    task_index=t,
                    machines=list(task[0]),
                    tools=list(task[1]),
                    deadline=0,
                    done=False,
                    runtime=task[2],
                    _n_machines=num_machines,
                    _n_tools=num_tools
                )
                instance.append(task)
        return instance

    @classmethod
    def _generate_instance_fjssp(cls, task_combinations: List[Tuple[int]], num_jobs: int, num_tasks: int,
                              num_machines: int, num_tools: int, **kwargs) -> List[Task]:
        """
        Generates a fjssp instance

        :param task_combinations: List with all possible tasks
        :param num_jobs: number of jobs generated in an instance
        :param num_tasks: number of tasks per job generated in an instance
        :param num_machines: number of machines available
        :param num_tools: number of tools available
        :param kwargs: Unused

        :return: fjssp instance (List of tasks)

        """
        instance = []
        # pick n jobs for this instance
        for j in range(num_jobs):
            # pick num_tasks tasks for this job
            for t in range(num_tasks):
                task = list(task_combinations[np.random.randint(0, len(task_combinations) - 1)])
                task = Task(
                    job_index=j,
                    task_index=t,
                    machines=list(task[0]),
                    tools=list(task[1]),
                    deadline=0,
                    done=False,
                    runtime=task[2],
                    _n_machines=num_machines,
                    _n_tools=num_tools
                )
                instance.append(task)
        return instance

    @classmethod
    def set_deadlines_to_max_deadline_per_job(cls, instances: List[List[Task]], num_jobs: int):
        """
        Equals all Task deadlines from one Job according to the one of the last task in the job

        :param instances: List of instances
        :param num_jobs: Number of jobs in an instance

        :return: List of instances with equaled job deadlines

        """
        # Argument sanity check
        assert isinstance(instances, list) and isinstance(num_jobs, int), \
            "Warning: You can only set deadlines for a list of instances with num_jobs of type integer."

        for instance in instances:
            # reset max deadlines for current instance
            max_deadline = [0] * num_jobs
            # get max deadline for every job
            for task in instance:
                if task.deadline > max_deadline[task.job_index]:
                    max_deadline[task.job_index] = task.deadline
            # assign max deadline to every task
            for task in instance:
                task.deadline = max_deadline[task.job_index]

    @classmethod
    def compute_and_set_hashes(cls, instances: List[List[Task]]):
        for instance in instances:
            instance_hash = hash(tuple(instance))
            # set hash attributes of each task
            for task in instance:
                task.instance_hash = instance_hash


if __name__ == '__main__':
    # my_instances = SPFactory.generate_instances(num_jobs=4, num_tasks=5, num_machines=4, sp_type="fjssp")
    # for i, my_instance in enumerate(my_instances):
    #     print("Setup", i)
    #     for my_task in my_instance:
    #         print(my_task.str_info())
    pass
