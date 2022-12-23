"""
This file provides functions to visualize the current states of a scheduling problems as gantt charts.
Moreover, the generated gantt chart figures can be saved (e.g. as gif).
"""
from pathlib import Path
from typing import List

import matplotlib
from matplotlib import pyplot
import numpy as np
import PIL.Image

from src.data_generator.sp_factory import Task
from src.utils.ui_tools.progressbar import progressbar

# Constants
VISUALS_DIRECTORY: Path = Path(__file__).parent.parent / 'visuals'


class GanttChartPlotter:
    """
    This class provides functions to visualize the current states of a scheduling problems as gantt charts and
    save them as image or gif.
    """
    @classmethod
    def get_gantt_chart_image(cls, tasks: List[Task], show_image: bool = False, return_image: bool = True,
                              quality_dpi: int = 100, overall_makespan: int = None, overall_num_machines: int = None,
                              overall_task_position_list: List[int] = None) -> PIL.Image:
        """
        Can be used to visualize the current state of a scheduling problem as a gantt chart. Note that the figure
        becomes too large with large processing times and numbers of tasks.

        :param tasks: List of tasks (instance) to be visualized
        :param show_image: True, if the generated image is to be visualized
        :param return_image: True if the generated image is to be returned
        :param quality_dpi: dpi of the generated image
        :param overall_makespan: Makespan of the scheduling problem. Can be None
        :param overall_num_machines: Number of machines available in the scheduling problem. Can be None
        :param overall_task_position_list: Task position in original list. Can be None

        :return: Gantt chart image

        """
        if show_image:
            matplotlib.use('TkAgg')
            _, _ = cls._make_gantt_chart_image(tasks, quality_dpi, overall_makespan, overall_num_machines,
                                               overall_task_position_list)
            pyplot.show()
            pyplot.figure().clear()
            pyplot.clf()
            pyplot.close()
        if return_image:
            matplotlib.use('agg')
            fig, _ = cls._make_gantt_chart_image(tasks, quality_dpi, overall_makespan, overall_num_machines,
                                                 overall_task_position_list)
            fig.canvas.draw()
            image_gantt_chart: PIL.Image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                                                               fig.canvas.tostring_rgb())
            # Cleanup
            pyplot.figure().clear()
            pyplot.clf()
            pyplot.close(fig)
            pyplot.close()

            return image_gantt_chart

    @classmethod
    def _make_gantt_chart_image(cls, tasks: List[Task], quality_dpi: int = 100, overall_makespan: int = None,
                                overall_num_machines: int = None,
                                overall_task_position_list: List[int] = None) -> PIL.Image:
        """
        Can be used to visualize the current state of a scheduling problem as a gantt chart

        :param tasks: List of tasks (instance) to be visualized
        :param quality_dpi: dpi of the generated image
        :param overall_makespan: Makespan of the scheduling problem. Can be None
        :param overall_num_machines: Number of machines available in the scheduling problem. Can be None
        :param overall_task_position_list: Task position in original list. Can be None

        :return: Gantt chart image

        """
        # Set important variables
        # Overall makespan of the schedule (+ 1 so that the Gantt Chart x-axis is complete)
        makespan: int = max([task.finished for task in tasks]) + 1 \
            if overall_makespan is None else overall_makespan
        # Number of machines available in the schedule
        num_machines: int = len(tasks[0].machines) \
            if overall_num_machines is None else overall_num_machines
        # Task position in original list
        task_position_list: List[int] = list(range(len(tasks))) \
            if overall_task_position_list is None else overall_task_position_list

        # Plot figure settings
        fig, gnt = pyplot.subplots(figsize=(30 * (makespan / 50), 10), dpi=quality_dpi)

        # X-axis settings
        gnt.set_xticks(list(range(makespan)))
        gnt.set_xlim([-0.5, makespan + 2])

        # Y-axis settings
        y_ticks = []
        y_labels = []
        for i in range(num_machines):
            y_ticks.append(25 + i * 10)
            y_labels.append(f"M. {i}")
        gnt.set_yticks(y_ticks)
        gnt.set_yticklabels(y_labels)
        gnt.set_ylim([10, int(10 * (num_machines + 2) + 5)])

        # Label
        gnt.set_xlabel('Steps')
        gnt.set_title('Scheduling')

        # will use these 16 colors indicating jobs but loops back to green for the 16th and so on
        # colors = ['green', 'blue', 'red', 'orange', 'brown', 'grey', 'cyan', 'olive', 'brick', 'goldenrod',
        #         'ochre', 'teal', 'dark slate blue', 'light pink', 'beige']
        color = ['#15b01a', '#0343df', '#e50000', '#f97306', '#653700', '#929591', '#00ffff', '#6e750e',
                 '#a03623', '#fac205', '#bf9005', '#029386', '#214761', '#ffd1df', '#e6daa6']


        # Plot bars
        task: Task
        for i, task in zip(task_position_list, tasks):
            # Draw if task scheduled
            if task.done:
                # Get task state variables
                start_time = task.started
                finish_time = task.finished
                selected_machine = task.selected_machine
                # Get readable tool indices
                tools = list(np.where(task.tools)[0])
                machine_indices = list(np.where(task.machines)[0])  # Get readable machine indices

                # Calculate y-position
                y_axes = 10 * (selected_machine + 2)
                gnt.broken_barh([(start_time, finish_time - start_time)], (y_axes, 9),
                                facecolor=color[task.job_index % 15], edgecolor='black')

                # Annotate tools, job number, and deadline
                pyplot.annotate(f'{"T:" + str(tools) + "  " if tools else ""}M:{machine_indices}',
                                (start_time + 0.1, y_axes + 7.5))
                pyplot.annotate(f'J {task.job_index} | T {task.task_index + 1} |', (start_time + 0.1, y_axes + 4.3),
                                fontsize=15)
                pyplot.annotate(f'L: {i}  D:{task.deadline}', (start_time + 0.1, y_axes + 0.5))

        return fig, gnt

    @staticmethod
    def save_gantt_chart_image(gantt_chart: PIL.Image.Image, save_path_dir: Path = Path(VISUALS_DIRECTORY),
                               filename: str = "gantt_chart", file_type: str = "png") -> Path:
        """
        Saves the input image

        :param gantt_chart: Gantt chart image
        :param save_path_dir: Relative path where the image is to be saved
        :param filename: Name under the image is to be saved
        :param file_type: Suffix with the image is to be saved

        :return: Path of the saved image

        """
        # Check path existence - create if not exist
        if not save_path_dir.exists():
            save_path_dir.mkdir(parents=True, exist_ok=True)

        full_file_path: Path = save_path_dir / f'{filename}.{file_type}'
        gantt_chart.save(full_file_path)

        return full_file_path

    @classmethod
    def get_gantt_chart_image_and_save(cls, tasks: List[Task], show_image: bool = False, quality_dpi: int = 100,
                                       overall_makespan: int = None, overall_num_machines: int = None,
                                       overall_task_position_list: List[int] = None,
                                       save_path_dir: Path = Path(VISUALS_DIRECTORY),
                                       filename: str = "gantt_chart", file_type: str = "png") -> Path:
        """
        Initializes the creation and saving of a gantt chart image

        :param tasks: List of tasks (instance) to be visualized
        :param show_image: True, if the generated image is to be visualized
        :param quality_dpi: dpi of the generated image
        :param overall_makespan: Makespan of the scheduling problem. Can be None
        :param overall_num_machines: Number of machines available in the scheduling problem. Can be None
        :param overall_task_position_list: Task position in original list. Can be None
        :param save_path_dir: Relative path where the image is to be saved
        :param filename: Name under the image is to be saved
        :param file_type: Suffix with the image is to be saved

        :return: Path of the saved image

        """
        image: PIL.Image.Image = cls.get_gantt_chart_image(tasks=tasks, show_image=show_image, quality_dpi=quality_dpi,
                                                           overall_makespan=overall_makespan,
                                                           overall_num_machines=overall_num_machines,
                                                           overall_task_position_list=overall_task_position_list)

        image_path: Path = cls.save_gantt_chart_image(gantt_chart=image, save_path_dir=save_path_dir, filename=filename,
                                                      file_type=file_type)
        return image_path

    @classmethod
    def get_gantt_chart_gif_and_save(cls, tasks: List[Task], save_path_dir: Path = Path(VISUALS_DIRECTORY),
                                     filename: str = "gantt_chart", save_intermediate_images: bool = False,
                                     quality_dpi: int = 80) -> Path:
        """
        Can be used to generate and save a gif of a gantt chart

        :param tasks: List of tasks (instance) to be visualized
        :param save_path_dir: Relative path where the gif is to be saved
        :param filename: Name under the gif is to be saved
        :param save_intermediate_images: True if the intermediate images of the gif creation should be saved
        :param quality_dpi: dpi of the generated image

        :return: Path of the saved image

        """
        # Check path existence - create if not exist
        if not save_path_dir.exists():
            save_path_dir.mkdir(parents=True, exist_ok=True)

        # Set important variables
        # Overall makespan of schedule
        makespan = max([task.finished for task in tasks])
        # Overall number of machines available in schedule
        num_machines = len(tasks[0].machines)

        # Collect created images
        images: List[PIL.Image.Image] = []

        # For every step in makespan plus extra space at the end
        with pyplot.rc_context(rc={'figure.max_open_warning': 0}):
            _extra_steps: int = 5
            steps_to_take: int = makespan + _extra_steps
            for step in progressbar(list(range(steps_to_take)), prefix="Making gif: ", size=40):
                image: PIL.Image
                task_position_list, current_task_list = zip(
                    *[(i, task) for i, task in enumerate(tasks) if task.started <= step])
                image: PIL.Image.Image = cls.get_gantt_chart_image(tasks=current_task_list, quality_dpi=quality_dpi,
                                                                   overall_makespan=makespan,
                                                                   overall_num_machines=num_machines,
                                                                   overall_task_position_list=task_position_list)

                images.append(image.quantize(method=PIL.Image.MEDIANCUT))
                if save_intermediate_images:
                    image.save(save_path_dir / f'{filename}_gif_image_{step}.png')
            # Build and save GIF
            images[0].save((save_path_dir / f'{filename}.gif'), save_all=True, append_images=images[1:], loop=0,
                           duration=150)

            return save_path_dir / f'{filename}.gif'
