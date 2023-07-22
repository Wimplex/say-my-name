import math
from typing import Iterable, List
from pathlib import Path
from numbers import Number


CONFIGS_DIR = Path("./configs")


def argmax(values: Iterable[Number]) -> int:
    max_val = max(values)
    idx = values.index(max_val)

    return idx


def argsort(values: Iterable[Number]) -> List[int]:
    return list(sorted(range(len(values)), key=values.__getitem__))


def softmax(unnormed_distribution: Iterable[float]) -> List[float]:
    exps = [math.exp(v) for v in unnormed_distribution]
    sum_ = sum(exps)
    probs = [e / sum_ for e in exps]

    return probs