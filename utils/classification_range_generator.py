from typing import Callable, List, Tuple


def generate_price_intervals(
    A: int, B: int, interval_len_func: Callable[[float], float]
) -> List[Tuple[float, float]]:
    """
    Generates a list of disjoint price intervals from A to B.
    The length of the intervals grows according to a quadratic function f(x) = ax^2 + bx + c.

    :param A: starting value of the range
    :param B: ending value of the range
    :param interval_len_func: func that describes changing of intervals length
    :return: list of intervals (tuples) [(start, end), ...]
    """

    intervals = []
    start = A
    x = 0

    while start < B:
        length = int(interval_len_func(x))
        if length <= 0:
            x += 1
            continue
        end = start + length
        if end > B:
            end = B
        intervals.append((start, end))
        start = end
        x += 1

    return intervals
