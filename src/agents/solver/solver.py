"""
This Solver solves certain scheduling problems optimally

It can handle:

- Classic JSSP
- Classic FJSSP
- both of the above with and without tool constraints
- optimization criteria tardiness and makespan

"""
import numpy as np
import collections
from ortools.sat.python import cp_model
from typing import List
import argparse
import copy

from src.data_generator.task import Task
from src.utils.file_handler.data_handler import DataHandler
from src.visuals_generator.gantt_chart import GanttChartPlotter


# constants
SOLVER_OBJECTIVE: str = 'makespan'


class OrToolSolver:
    """
    This class can be used to solve JSSP problems. It can handle:

    - Classic JSSP
    - Classic FJSSP
    - both of the above with and without tool constraints
    - optimization criteria tardiness and makespan

    Data needs to be passed in 'instance format' and is returned in this format, too.
    """

    @classmethod
    def optimize(cls, instance: List[Task], objective: str = 'makespan'):
        """
        Optimizes the passed instance according to the passed objective.

        :param List[Task] instance: The instance as a list of Tasks
        :param str objective: Bbjective to be minimized. May be 'makespan' or 'tardiness'.

        :return: tuple(list[Task], float) Solved instance and objective value

        """
        # parse instance to suitable format
        instance = cls.parse_instance_to_solver_format(instance)

        # count machines
        machines_count = 1 + max(machine_option for job in instance for task in job for machine_option in task[0])
        all_machines = range(machines_count)
        
        # count tools
        tools_list = []
        for job in instance:
            for task in job:
                if len(task[3]) > 0:
                    tools_list.append(max(task[3]))
        tools_count = 1 + max(tools_list) if len(tools_list) > 0 else 0
        all_tools = range(tools_count)
        
        # Computes worst-case makespan horizon dynamically as the sum of all durations.
        horizon = sum(task[1] for job in instance for task in job)

        # initiate CP Model
        model = cp_model.CpModel()

        # Named tuple to store information about created variables.
        task_type = collections.namedtuple('task_type', 'start end interval duedate tardiness tools')
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple(
            'assigned_task_type', 'start job index machines duration duedate tool')

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_usages = {}  # indexed by (job_id, task_id, machine_id)
        machine_to_intervals = collections.defaultdict(list)
        tool_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(instance):
            for task_id, task in enumerate(job):
                # assemble task information
                duration = task[1]
                due_date = task[2]
                tools = task[3]
                suffix = '_%i_%i' % (job_id, task_id)
                # set up constants and variables
                due_date_const = model.NewConstant(due_date)
                start_var = model.NewIntVar(0, horizon - duration, f'start{suffix}')
                duration = model.NewConstant(duration)
                end_var = model.NewIntVar(0, horizon, f'end{suffix}')
                interval_var = model.NewIntervalVar(start_var, duration, end_var, f'interval{suffix}')
                if due_date == 0:  # if set to 0 this is not the last task.
                    # Therefore set it to end_var, so that the tardiness of this subtask is 0 by default
                    tardiness_var = model.NewConstant(0)
                else:
                    tardiness_var = model.NewIntVar(- (horizon - due_date), 0, f'tardiness{suffix}')
                    model.Add(tardiness_var == end_var - due_date_const)

                # add to all_tasks
                all_tasks[job_id, task_id] = task_type(start=start_var,
                                                       end=end_var,
                                                       interval=interval_var,
                                                       duedate=due_date_const,
                                                       tardiness=tardiness_var,
                                                       tools=tools)

                # add intervals of blocked resources

                # tool intervals
                for tool in tools:
                    tool_to_intervals[tool].append(interval_var)

                # add conditional (alternative) machine intervals (only one of all machines is used)
                machines = task[0]
                alt_machine_usages = []
                for machine_id in machines:
                    alternative_suffix = f'{job_id}_{task_id}_{machine_id}'
                    machine_usage = model.NewBoolVar(f'presence_{alternative_suffix}')
                    alt_start = model.NewIntVar(0, horizon, f'start_{alternative_suffix}')
                    alt_end = model.NewIntVar(0, horizon, f'end_{alternative_suffix}')
                    alt_interval = model.NewOptionalIntervalVar(
                        alt_start, duration, alt_end, machine_usage, f'interval_{alternative_suffix}'
                    )
                    alt_machine_usages.append(machine_usage)

                    # Link the master variables with the local ones
                    model.Add(start_var == alt_start).OnlyEnforceIf(machine_usage)
                    model.Add(end_var == alt_end).OnlyEnforceIf(machine_usage)

                    # Add local interval to the right machine
                    machine_to_intervals[machine_id].append(alt_interval)

                    # Store booleans of the usages for the solution
                    machine_usages[(job_id, task_id, machine_id)] = machine_usage

                # select exactly one machine usage per task
                model.Add(sum(alt_machine_usages) == 1)

        # Create and add disjunctive constraints of intervals, in which a machine or tool may be used
        for machine in all_machines:
            model.AddNoOverlap(machine_to_intervals[machine])

        for tool in all_tools:
            model.AddNoOverlap(tool_to_intervals[tool])

        # Precedences inside a job.
        for job_id, job in enumerate(instance):
            for task_id in range(len(job) - 1):
                model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

        # # constraint to only schedule one task at a time TODO: This is currently not used. Add later, if necessary.
        # task_keys_combinations = itertools.combinations(all_tasks.keys(), 2)
        # for combination in task_keys_combinations:
        #     model.Add(all_tasks[combination[0][0], combination[0][1]].start !=
        #               all_tasks[combination[1][0], combination[1][1]].start)

        # Define objective
        if objective == 'makespan':
            obj_var = model.NewIntVar(0, horizon, 'makespan')
            model.AddMaxEquality(obj_var, [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(instance)])
        elif objective == 'tardiness':
            # Tardiness objective.
            obj_var = model.NewIntVar(-horizon, horizon, 'total_tardiness')
            model.Add(
                obj_var == sum([all_tasks[job_id, len(job) - 1].tardiness for job_id, job in enumerate(instance)]))
        else:
            raise ValueError(f'Unknown objective {objective}')

        # Add objective
        model.Minimize(obj_var)

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # Extract solution to more handy format
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # print('Solution:')
            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)
            for job_id, job in enumerate(instance):
                for task_id, task in enumerate(job):
                    for alt_machine in instance[job_id][task_id][0]:
                        if solver.Value(machine_usages[(job_id, task_id, alt_machine)]):
                            machine = alt_machine
                    assigned_jobs[machine].append(
                        assigned_task_type(start=solver.Value(
                            all_tasks[job_id, task_id].start),
                            job=job_id,
                            index=task_id,
                            machines=task[0],
                            duration=task[1],
                            duedate=task[2],
                            tool=task[3]))

            # Create per machine output lines.
            output = ''
            for machine in all_machines:
                # Sort by starting time.
                assigned_jobs[machine].sort()
                sol_line_tasks = 'Machine ' + str(machine) + ': '
                sol_line = '           '

                for assigned_task in assigned_jobs[machine]:
                    name = 'job_%i_task_%i' % (assigned_task.job,
                                               assigned_task.index)
                    # Add spaces to output to align columns.
                    sol_line_tasks += '%-15s' % name

                    start = assigned_task.start
                    duration = assigned_task.duration
                    tool = assigned_task.tool
                    sol_tmp = f'[{start}, {start + duration}, tool:{tool}]'
                    # Add spaces to output to align columns.
                    sol_line += '%-15s' % sol_tmp

                sol_line += '\n'
                sol_line_tasks += '\n'
                output += sol_line_tasks
                output += sol_line

            # # Finally print the solution found.
            # print(f'Minimal {objective}: {solver.ObjectiveValue()}')
            # print(output)
        else:
            print('No solution found.')

        return assigned_jobs, solver.ObjectiveValue()

    @staticmethod
    def parse_instance_to_solver_format(instance: List[Task]):
        """
        Parses the instance to a processable format.

        :param list[Task] instance: The instance as a list of Tasks
        :return: jobs lists with Tuples for every task, including all necessary information for the solver function
                    machine_id(s), processing_time, due_date(set to 0 for all but last task in job), tool

        :Example: Job2 = [([0, 2, 3], 8, 0, 4), ([0, 2, 3], 6, 0, 2), ...]

        """
        jobs_dict = {}
        for task in instance:
            task_info = [np.nonzero(task.machines)[0].tolist(), task.runtime]
            # set deadline to 0, if this task is not the terminal task
            if task.deadline != task._deadline:
                task_info.append(0)
            else:
                task_info.append(task.deadline)

            task_info.append(np.nonzero(task.tools)[0].tolist())

            job_num = task.job_index
            # if the job list in the dictionary is empty, initialize it
            if job_num not in jobs_dict:
                jobs_dict[job_num] = []

            jobs_dict[job_num].append(task_info)

        return list(jobs_dict.values())

    @staticmethod
    def parse_to_plottable_format(original_instance: List[Task], assigned_jobs_by_solver):
        """
        Reformats the solution into the original instance format to be passed to the gantt chart generator.

        :param list[Tasks] original_instance: Original instance in original format
        :param assigned_jobs_by_solver: solution passed by the optimize() function

        :return: list[Tasks] plottable solved instance

        """
        max_machine_num = len(original_instance[0].machines)

        # iterate over all machines and tasks scheduled on them
        tasks_list = []
        for machine in assigned_jobs_by_solver:
            for assigned_task in assigned_jobs_by_solver[machine]:
                job_index = assigned_task.job
                task_index = assigned_task.index
                machines = np.zeros(max_machine_num)
                machines[assigned_task.machines] = 1
                task = Task(job_index=job_index, task_index=task_index,
                            machines=machines,
                            tools=assigned_task.tool, deadline=assigned_task.duedate, done=True,
                            runtime=assigned_task.duration, started=assigned_task.start,
                            finished=assigned_task.start+assigned_task.duration, selected_machine=machine)
                tasks_list.append(task)
                pass

        return tasks_list


def get_perser_args():
    """parse arguments from command line"""
    # Arguments for function
    parser = argparse.ArgumentParser(description='Solver for computing solution of scheduling problems')

    parser.add_argument('-fp', '--instances_file_path', type=str, required=True,
                        help='Path to instances data file you want to solve')
    parser.add_argument('-write', '--write_to_file', dest='write_to_file', action='store_true',
                        help='Enable or disable result export to file')
    parser.add_argument('-plot', '--plot_ganttchart', dest="plot_gantt_chart", action="store_true",
                        help='Enable or disable model result plot.')
    parser.add_argument('-obj', '--solver_objective', type=str, required=False,
                        help='According to this objective the solver computes a solution. '
                             'Choose between makespan and tardiness')

    args = parser.parse_args()

    return args


def main(instances_data_file_path, solver_objective, write_to_file=False, plot_gantt_chart=False):

    # load and parse jobs
    data = DataHandler.load_instances_data_file(instances_data_file_path=instances_data_file_path)

    or_tool_solver = OrToolSolver()
    solved_data = []

    for sample_instance in copy.copy(data):
        # find solution
        assigned_jobs, objective_value = or_tool_solver.optimize(sample_instance, objective=solver_objective)

        # get solution into Task format
        parsed_data = or_tool_solver.parse_to_plottable_format(sample_instance, assigned_jobs)
        solved_data.append(parsed_data)
        if plot_gantt_chart:
            GanttChartPlotter.get_gantt_chart_image(parsed_data, show_image=True, return_image=False)

    # Write solved data to file
    if write_to_file:
        DataHandler.write_solved_data_to_file(instances_data_file_path, solved_data)


if __name__ == '__main__':

    # get arguments
    parse_args = get_perser_args()
    path = parse_args.instances_file_path
    write = parse_args.write_to_file
    plot = parse_args.plot_gantt_chart
    objective = parse_args.solver_objective
    # set objective to default if no terminal input
    if objective is None:
        objective = SOLVER_OBJECTIVE

    # pass args to main
    main(instances_data_file_path=path, solver_objective=objective, write_to_file=write, plot_gantt_chart=plot)
