"""
Progressbar for the gif creation.
"""
import sys


def progressbar(iterable_object, prefix: str = "", size: int = 60, out=sys.stdout):
    """
    Prints a progressbar. Used for the gif creation
    :param iterable_object: List to be progressed
    :param prefix: String prefix for the progressbar print
    :param size: Number of '.' to be added to the progressbar
    :param out: Output channel for the print
    :return: Yield values of iterable_object
    """
    count = len(iterable_object)

    def show(j):
        status = int(size * j / count)
        print("{}[{}{}] {}/{}".format(prefix, "█" * status, "." * (size - status), j, count),
              end='\r', file=out, flush=True)

    show(0)
    for _i, _item in enumerate(iterable_object):
        yield _item
        show(_i + 1)
    print("\n", flush=True, file=out)
