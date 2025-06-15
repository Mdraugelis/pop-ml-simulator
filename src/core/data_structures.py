from dataclasses import dataclass, field
from random import Random
from typing import List

from src.utils.logging import log_call


@dataclass
class Patient:
    age: int


@dataclass
class Population:
    n_persons: int
    age_min: int
    age_max: int
    random_seed: int
    persons: List[Patient] = field(init=False)

    @log_call
    def __post_init__(self) -> None:
        rnd = Random(self.random_seed)
        self.persons = [
            Patient(age=rnd.randint(self.age_min, self.age_max))
            for _ in range(self.n_persons)
        ]
