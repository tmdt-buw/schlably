import pathlib
import pickle
import yaml

from src.data_generator.instance_factory import main as data_generation_main
from src.agents.train import main as training_main
from src.utils.file_handler.config_handler import ConfigHandler

# constants
from src.utils.file_handler.data_handler import DATA_DIRECTORY
from src.utils.file_handler.config_handler import CONFIG_DIRECTORY
DEFAULT_DATA_GEN_FILE: str = 'data_generation/jssp/config_job3_task4_tools0.yaml'

DEFAULT_TRAINING_FILE: str = 'training/dqn/config_job3_task4_tools0.yaml'


if __name__ == "__main__":

    sp_types = ['jssp', 'fjssp']
    algorithms = ['dqn', 'ppo']
    num_jobs = [1, 3, 6]
    num_tasks = [1, 3, 6]
    num_tools = [0, 1, 6]

    # kwargs to overwrite defaults according to the current test setup
    data_gen_overwrite_kwargs = {
        'num_instances': 10,
        'num_processes': 10,
        'write_to_file': True
    }

    train_overwrite_kwargs = {
        'total_timesteps': 100,
        'intermediate_test_interval': 100,
        'saved_model_name': 'model_from_code_test'
    }

    files_to_delete = []
    # TODO understand why and where config are updated and if all owrk as intened
    for num_j, num_ta, num_to in zip(num_jobs, num_tasks, num_tools):
        # for each sp_type
        for sp_type in sp_types:
            # specify instances file
            instances_file = f"config_job{num_j}_task{num_ta}_tools{num_to}.pkl"
            instances_file_path = DATA_DIRECTORY / sp_type / instances_file
            # if file does not exist, create a file
            if not instances_file_path.exists():
                # get default config
                data_gen_default_config = ConfigHandler.get_config(DEFAULT_DATA_GEN_FILE)

                # update data_gent_overwrite_kwargs with current jobs, task, tool setup
                data_gen_overwrite_kwargs.update({
                    'sp_type': sp_type,
                    'num_jobs': num_j,
                    'num_tasks': num_ta,
                    'num_tools': num_to,
                    'num_machines': num_j,
                })

                # update default config with overwrite kwargs
                data_gen_default_config.update(data_gen_overwrite_kwargs)

                data_gen_config = data_gen_overwrite_kwargs

                # use (new) config to run data generation
                data_generation_main(external_config=data_gen_config)

                # add config and data path to files_to_delete
                files_to_delete.append(instances_file_path)

            for algorithm in algorithms:
                # test train function (includes intermediate_test, test)

                # load default config
                train_default_config = ConfigHandler.get_config(DEFAULT_TRAINING_FILE)

                # update train_overwrite kwargs_with current algorithm and instances file
                train_overwrite_kwargs.update({
                    'algorithm': algorithm,
                    'instances_file': f"{sp_type}/{instances_file}"
                })

                # update default config with overwrite kwargs
                train_default_config.update(train_overwrite_kwargs)

                print(f"Testing train file for job{num_j}_task{num_ta}_tools{num_to}, {sp_type}, {algorithm}")
                # start training
                training_main(external_config=train_default_config)

    # delete data files created only for this code test
    for file in files_to_delete:
        pathlib.Path.unlink(file)
