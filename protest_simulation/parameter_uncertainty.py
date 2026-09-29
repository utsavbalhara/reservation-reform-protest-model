import numpy as np

from .model_parameters import ProtestModelParameters

LOSS_AVERSION_RANGE = (0.7, 1.3)
SYMBOLIC_THREAT_RANGE = (0.8, 1.2)
OPPOSITION_AMPLIFIER_RANGE = (0.85, 1.15)
NEIGHBOURHOOD_INFLUENCE_RANGE = (0.8, 1.2)
NATIONAL_INFLUENCE_RANGE = (0.8, 1.2)
MEAN_THRESHOLD_STANDARD_DEVIATION = 0.10
DEATH_RATE_RANGE = (0.5, 2.0)


def draw_plausible_world(parameters: ProtestModelParameters, random_generator: np.random.Generator) -> ProtestModelParameters:
    world = parameters.copy()
    world.loss_aversion *= random_generator.uniform(*LOSS_AVERSION_RANGE)
    world.symbolic_threat_by_group = world.symbolic_threat_by_group * random_generator.uniform(*SYMBOLIC_THREAT_RANGE)
    world.opposition_party_amplifier *= random_generator.uniform(*OPPOSITION_AMPLIFIER_RANGE)
    world.max_neighbourhood_influence *= random_generator.uniform(*NEIGHBOURHOOD_INFLUENCE_RANGE)
    world.max_national_visibility_influence *= random_generator.uniform(*NATIONAL_INFLUENCE_RANGE)
    world.mean_participation_threshold += random_generator.normal(0, MEAN_THRESHOLD_STANDARD_DEVIATION)
    world.deaths_per_crore_protester_days *= random_generator.uniform(*DEATH_RATE_RANGE)
    return world
