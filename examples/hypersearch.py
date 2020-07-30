from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np


class Score:
    def __init__(self, score: float, parameters: Any) -> None:
        self.score = score
        self.parameters = parameters

    def __str__(self) -> str:
        return f"Score<{self.score}, parameters: {str(self.parameters)}>"


Range = Tuple[float, float]
Item = Tuple[Range, bool]
Params = Dict[str, Range]
TrainFn = Callable[[Any], float]


class HyperSearch:
    def __init__(self) -> None:
        self.parameters: Params = {}
        self.scales: List[bool] = []
        self.best_score = Score(0, {})

    def __setitem__(self, key: str, item: Item) -> None:
        range, log = item
        self.parameters[key] = range
        self.scales.append(log)

    def _sample(self, range: Range, log: bool) -> float:
        if log:
            a, b = np.log(range[0]), np.log(range[1])
            return np.exp(np.random.uniform(a, b))
        return np.random.uniform(range[0], range[1])

    def search(
        self, train_fn: TrainFn, *args, iterations: int = 10, **kargs
    ) -> Score:
        for iteration in range(iterations):
            params = {
                k: self._sample(v, s)
                for (k, v), s in zip(self.parameters.items(), self.scales)
            }
            score = Score(train_fn(*args, **params, **kargs), params)
            if score.score > self.best_score.score:
                self.best_score.score = score.score
                self.best_score.parameters = params
            print("========================= SCORE =========================")
            print(score)
            print(self.best_score)
            print("=========================================================")
        return self.best_score