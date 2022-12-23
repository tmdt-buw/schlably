"""
Use the decorators from this module to time code execution.
"""
import time

func_runtimes = {}


def timing_decorator_factory(print_runtime: bool = False, print_runtime_average: bool = False) -> None:
    """
    Times code execution
    :param print_runtime: If true, prints the runtime
    :param print_runtime_average: If true, prints runtime_average
    :return: None
    """
    def timing_decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            original_return_val = func(*args, **kwargs)
            end = time.time()

            elapsed_time = end - start
            func_runtimes.setdefault(func.__name__, []).append(elapsed_time)

            # Factory options
            if print_runtime:
                print("time elapsed in ", func.__name__, ": ", elapsed_time, sep='')
            if print_runtime_average:
                runtimes = func_runtimes[func.__name__]
                total_runtime = sum(runtimes)
                average = total_runtime / len(runtimes)
                print("average run time in ", func.__name__, ": ", average, sep='')

            # Return - necessary!
            return original_return_val
        return wrapper
    return timing_decorator


def print_timing_stats() -> None:
    """
    Prints timing stats to console
    :return: None
    """
    for key, runtimes in func_runtimes.items():
        calls = len(runtimes)
        average = sum(runtimes) / len(runtimes)
        print(key, " calls ", calls, " average run time: ", average)
