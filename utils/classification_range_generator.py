from typing import List, Tuple


def generate_price_intervals(
    A: int, B: int, a: float = 0, b: float = 0, c: float = 1
) -> List[Tuple[float, float]]:
    """
    Generates a list of disjoint price intervals from A to B.
    The length of the intervals grows according to a quadratic function f(x) = ax^2 + bx + c.

    :param A: starting value of the range
    :param B: ending value of the range
    :param a: coefficient of the quadratic term (for x^2)
    :param b: coefficient of the linear term (for x)
    :param c: constant term of the quadratic function
    :return: list of intervals (tuples) [(start, end), ...]
    """

    intervals = []
    start = A
    x = 0

    while start < B:
        length = a * x**2 + b * x + c
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
