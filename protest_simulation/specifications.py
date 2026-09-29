"""Named model specifications. Each fixes a baseline parameter set, how to build the population and how to draw
uncertain parameters for a Monte Carlo run. Experiments take a specification name, so the same analysis can be run
on the stylized reference model and on the grounded model."""
from dataclasses import dataclass, field

import numpy as np

from .model_parameters import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, ProtestModelParameters
from .monte_carlo import DEFAULT_AGENT_COUNT, POPULATION_SEED
from .parameter_uncertainty import draw_plausible_world
from .synthetic_population import build_synthetic_india


@dataclass
class Specification:
    name: str
    description: str
    base_parameters: ProtestModelParameters
    population_options: dict = field(default_factory=dict)
    world_sampler: object = draw_plausible_world
    population_builder: object = None

    def build_population(self, agent_count: int = DEFAULT_AGENT_COUNT):
        if self.population_builder is not None:
            return self.population_builder(agent_count, np.random.default_rng(POPULATION_SEED), **self.population_options)
        return build_synthetic_india(agent_count, np.random.default_rng(POPULATION_SEED), **self.population_options)


SPECIFICATIONS = {
    "stylized": Specification(
        name="stylized",
        description="The stylized reference model (fourth iteration): national population, fixed bandh days, Poisson deaths.",
        base_parameters=BASELINE_ABRUPT_INCOME_ONLY_SWITCH,
    ),
}


def get_specification(name: str) -> Specification:
    if name not in SPECIFICATIONS:
        # The grounded specification registers itself on import.
        from . import grounded_specification  # noqa: F401
    return SPECIFICATIONS[name]
