import random

import numpy as np
from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter
from tqdm import tqdm

from archive import ParetoArchive, Grid, numpy_dominates


def MOPSO(problem, low_bound, high_bound, max_iter, pop_size, archive_size, grid_size, w_inertia, rep_div):
    # Initialize all needed containers
    grid = Grid(grid_size)
    grid_func = lambda x: grid.calculate(x)
    archive = ParetoArchive(archive_size, crowding_function=grid_func, add_item_before_crowding=True)

    positions = np.random.uniform(low_bound, high_bound, size=(pop_size, problem.n_var))
    fitnesses = problem.evaluate(positions)
    velocities = np.zeros(shape=(pop_size, problem.n_var))
    p_bests = positions.copy()
    p_best_fits = fitnesses.copy()

    archive.add(positions, fitnesses)
    grid.calculate(archive.items)

    for step in tqdm(range(max_iter)):
        print("Step %d" % step)
        for i in range(problem.n_obj):
            print("mean: %.3f, min: %.3f, max: %.3f" % (np.average(fitnesses[:, i], axis=0),
                                                        np.min(fitnesses[:, i], axis=0),
                                                        np.max(fitnesses[:, i], axis=0)))

        # Perform velocity update
        r_1 = np.random.uniform(0.0, 1.0, size=(pop_size, 1))
        r_2 = np.random.uniform(0.0, 1.0, size=(pop_size, 1))

        # Calculate the reps as described in paper
        item_div_fit = []
        item_div_values = list(grid.item_div.values())
        for entry in item_div_values:
            item_div_fit.append(rep_div / len(entry))

        item_div_prob = list(map(lambda x: x / sum(item_div_fit), item_div_fit))
        item_div_idxs = list(range(len(item_div_prob)))
        grid_picks = np.random.choice(item_div_idxs, pop_size, True, p=item_div_prob)
        reps = []
        for entry in grid_picks:
            reps.append(random.sample(item_div_values[entry], 1)[0])

        reps = np.array(reps)

        inertia = w_inertia * velocities
        cognitive = r_1 * (p_bests - positions)
        social = r_2 * (reps - positions)

        velocities = inertia + cognitive + social

        # Update positions
        positions += velocities
        positions = positions.clip(low_bound, high_bound)

        # Calculate fitnesses
        fitnesses = problem.evaluate(positions)

        # Update archive
        archive.add(positions, fitnesses)
        grid.calculate(archive.items)

        # Update personal bests
        dominates = numpy_dominates(fitnesses, p_best_fits)
        p_best_fits[dominates] = fitnesses[dominates].copy()
        p_bests[dominates] = positions[dominates].copy()
    return archive.items


np.seterr('raise')  # Used for debugging, and it seems code was repaired.
problem = get_problem("zdt3")
# Must change the problem to -f(x) since MOPSO assumes maximization.
tmp = problem.evaluate
problem.evaluate = lambda x: -1 * tmp(x)

archive = MOPSO(problem, 0.0, 1.0, 100, 40, 200, 5, 0.4, 30)

# Visualize the results of the archive
# Note: did entry['fit'] * -1 to revert the fitnesses back to their original values.
pf = problem.pareto_front()
Scatter(legend=True).add(pf, label="Pareto-front").add(np.array([entry['fit'] * -1 for entry in archive]),
                                                       label="Result").show()
