from typing import Callable, List, Tuple


class IntervalsHandler:
    def __init__(self, A: int, B: int, function: Callable[[float], float]):
        self.function = function
        self.intervals = self._generate_price_intervals(A, B)

    def _generate_price_intervals(self, A: int, B: int) -> List[Tuple[float, float]]:
        intervals = []
        start = A
        x = 0

        while start < B:
            length = int(self.function(x))
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
