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
Params = Dict[str, Range]
TrainFn = Callable[[Any], float]


class HyperSearch:
    def __init__(self) -> None:
        self.parameters: Params = {}
        self.best_score = Score(0, {})

    def __setitem__(self, key: str, range: Range) -> None:
        self.parameters[key] = range

    def _sample(self, range: Range) -> float:
        return np.exp(np.random.uniform(np.log(range[0]), np.log(range[1])))

    def search(
        self, train_fn: TrainFn, *args, iterations: int = 10, **kargs
    ) -> Score:
        for iteration in range(iterations):
            params = { k: self._sample(v) for k, v in self.parameters.items() }
            score = train_fn(*args, **params, **kargs)
            if score > self.best_score[0]:
                self.best_score.score = score
                self.best_score.parameters = params
            print("========================= SCORE =========================")
            print(score)
            print("=========================================================")
        return self.best_score