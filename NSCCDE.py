import copy

import numpy as np
import pandas as pd
from nds import ndomsort
from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter
from tqdm import tqdm

from archive import ParetoArchive, crowdingDistance


def update_swarm_history(fitnesses, pop_history_columns, pop_history):
    data = {}
    for i in range(fitnesses.shape[1]):
        data["f_" + str(i) + "_Avg"] = np.average(fitnesses[:, i])
        data["f_" + str(i) + "_Min"] = np.min(fitnesses[:, i])
        data["f_" + str(i) + "_Max"] = np.max(fitnesses[:, i])
    df = pd.DataFrame([data], columns=pop_history_columns)
    pop_history = pd.concat((pop_history, df))
    return pop_history


def update_archive_history(archive, archive_history):
    """
        Updates the archive history by appending data to the archive_history variable.
    """
    curArchivePos = copy.deepcopy([x['pos'] for x in archive.items])
    curArchiveFit = copy.deepcopy([x['fit'] for x in archive.items])
    archive_history.append(
        {'positions': curArchivePos, 'fitnesses': curArchiveFit})
    return archive_history


def NSCCDE(problem, low_bound, high_bound, max_iter, pop_size, archive_size, decomp_num, F, CR, s_max, s_min):
    # decomp_num: number of populations to decomp into
    # Initialize histories
    archive_history = []
    pop_history_columns = []
    for i in range(problem.n_obj):
        pop_history_columns.append("f_" + str(i) + "_Avg")
        pop_history_columns.append("f_" + str(i) + "_Min")
        pop_history_columns.append("f_" + str(i) + "_Max")

    pop_history = [pd.DataFrame(columns=pop_history_columns)] * decomp_num

    # Initialize population, gather fitnesses and best solutions, and make archive
    archive = ParetoArchive(archive_size, allow_dominated=True, crowding_function=crowdingDistance,
                            track=['added_sol_idx'])
    pop = np.random.uniform(low_bound, high_bound, size=(pop_size, problem.n_var))
    fit = problem.evaluate(pop)
    archive.add(pop, fit)

    # Decompose the populations
    decision_vars = np.array([i for i in range(problem.n_var)])
    np.random.shuffle(decision_vars)
    decision_vars = np.array_split(decision_vars, decomp_num)
    pop = [pop[:, decision_vars[i]] for i in range(len(decision_vars))]
    fit = [fit.copy() for _ in range(len(decision_vars))]

    for step in tqdm(range(max_iter)):
        # Update histories
        for i in range(len(pop)):
            pop_history[i] = update_swarm_history(fit[i], pop_history_columns, pop_history[i])
            archive_history = update_archive_history(archive, archive_history)

        # Print out iteration info
        print("Step " + str(step))
        for i in range(decomp_num):
            print(f'Swarm {i}')
            for j in range(problem.n_obj):
                print("f_" + str(j) +
                      ", max: %.3f, min: %.3f, mean: %.3f" %
                      (pop_history[i]["f_" + str(j) + '_Max'].iloc[-1],
                       pop_history[i]["f_" + str(j) + '_Min'].iloc[-1],
                       pop_history[i]["f_" + str(j) + '_Avg'].iloc[-1]))

        # DE Operations
        for i in range(len(pop)):
            # DE/rand/1 mutation: V_i =  X_r1 + F * (X_r2 - X_r3)
            X_r1 = pop[i][np.random.randint(0, pop_size, size=pop_size)]
            X_r2 = pop[i][np.random.randint(0, pop_size, size=pop_size)]
            X_r3 = pop[i][np.random.randint(0, pop_size, size=pop_size)]

            V = X_r1 + F * (X_r2 - X_r3)

            # Binomial Crossover
            rand_float_vec = np.random.uniform(0, 1, size=pop[i].shape)
            rand_int_vec = np.random.randint(0, len(decision_vars[i]), size=pop[i].shape)
            indexes = np.tile([i for i in range(len(decision_vars[i]))], reps=(pop_size, 1))
            U = np.where(np.logical_or(rand_float_vec <= CR, indexes == rand_int_vec), V, pop[i])
            U = U.clip(low_bound, high_bound)

            # Select collaborator (highest crowding distance --> least crowded)
            crowdingDistance(archive.items)
            non_dominated = list(ndomsort.non_domin_sort(archive.items, get_objectives=lambda x: x['fit'])[0])
            non_dominated.sort(reverse=True, key=lambda x: x['dist'])  # sort by crowding distance in reverse
            collaborator = non_dominated[0]['pos']  # non-dominated with the least crowding

            # Assemble solutions
            new_sols = np.tile(collaborator, reps=(pop_size, 1))
            new_sols[:, decision_vars[i]] = U
            new_fit = problem.evaluate(new_sols)

            # Update Archive
            added_idxs = archive.add(new_sols, new_fit)

            # Replacement
            pop[i][added_idxs] = U[added_idxs]
            fit[i][added_idxs] = new_fit[added_idxs]

        # Spatial dispersal for archive
        nds_archive = ndomsort.non_domin_sort(archive.items, get_objectives=lambda x: x['fit'])
        archive_copy = []
        for front_num in range(len(nds_archive)):
            spatial_val = int(s_max - ((front_num * (s_max - s_min)) / len(nds_archive)))
            for entry in nds_archive[front_num]:
                for _ in range(spatial_val):  # add the archive entry in spatial_val number of times
                    archive_copy.append(entry['pos'].copy())
        archive_copy_size = len(archive_copy)
        archive_copy = np.array(archive_copy)

        # archive mutation is the same as the one for sub-populations
        # DE/rand/1 mutation: V_i =  X_r1 + F * (X_r2 - X_r3)
        X_r1 = archive_copy[np.random.randint(0, archive_copy_size, size=archive_copy_size)]
        X_r2 = archive_copy[np.random.randint(0, archive_copy_size, size=archive_copy_size)]
        X_r3 = archive_copy[np.random.randint(0, archive_copy_size, size=archive_copy_size)]

        archive_copy = X_r1 + F * (X_r2 - X_r3)
        archive_copy = archive_copy.clip(low_bound, high_bound)
        archive_copy_fit = problem.evaluate(archive_copy)
        archive.add(archive_copy, archive_copy_fit)

    return archive.items


np.seterr('raise')  # Used for debugging, and it seems code was repaired.
problem = get_problem("zdt1")
# NSCCDE assumes minimization so we're fine with zdt1.
archive = NSCCDE(problem, 0.0, 1.0, 100, 100, 100, 5, 0.5, 0.9, 6, 1)

# Visualize the results of the archive
pf = problem.pareto_front()
Scatter(legend=True).add(pf, label="Pareto-front").add(np.array([entry['fit'] for entry in archive]),
                                                       label="Result").show()