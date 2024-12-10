from functools import reduce
from operator import mul


def product(*factors: float) -> float:
    return reduce(mul, factors)
    # same as: reduce(lambda x, y: x * y, factors)
