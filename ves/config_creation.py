"""
Functions to help create initial configurations.
"""
import numpy as np

from ves.potentials import Potential2D


def singleParticle2D_init_coord(potential: Potential2D,
                                temp: float,
                                ntrials: int = 100,
                                xmin: float = 0,
                                xmax: float = 1,
                                ymin: float = 0,
                                ymax: float = 1):
    """
    Uses monte carlo trials to place a particle on a free energy surface
    defined by `potential`.

    Attempts are made within bounds (xmin, xmax) and (ymin, ymax).
    """

    init_coord = None
    for i in range(ntrials):
        trial_coord = [(xmax - xmin) * np.random.random() + xmin,
                       (ymax - ymin) * np.random.random() + ymin,
                       0]
        beta = 1 / (8.3145 / 1000 * temp)
        boltzmann_factor = np.exp(-beta * potential.potential(trial_coord[0], trial_coord[1]))
        if boltzmann_factor >= 0.5:
            init_coord = trial_coord
            print("[MC particle placement] Particle placement succeeded at iteration {}".format(i))
            break

    if init_coord is None:
        raise RuntimeError("[MC particle placement] Could not place particle on surface.")

    init_coord = np.array(init_coord).reshape((1, 3))

    return init_coord
