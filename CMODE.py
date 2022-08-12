import random

import numpy as np
from pymoo.visualization.scatter import Scatter
from tqdm import tqdm
from pymoo.factory import get_problem

from archive import ParetoArchive
from util import lehmer_mean, gen_random_vector_idx


def CMODE(problem, low_bound, high_bound, max_iter, pop_size, archive_size, c, F_m_init, CR_m_init, F_a_init, CR_a_init,
          tao_1, tao_2):
    """
    Implementation of CMODE:
    Wang, Jiahai, Weiwei Zhang, and Jun Zhang.
    "Cooperative differential evolution with multiple populations for multiobjective optimization."
    IEEE Transactions on Cybernetics 46.12 (2015): 2848-2861.

    Important: assumes maximization

    Typical parameter values:
        - c = 0.1
        - CR_m_init = 0.5
        - F_m_init = 0.5
        - F_a_init = 0.5
        - CR_a_init = 0.9
        - tao_1 = 0.1
        - tao_2 = 0.1

    :param CR_m_init: Initial CR value for subpopulations (not archive)
    :type CR_m_init: float
    :param F_m_init: Initial F value for subpopulations (not archive)
    :type F_m_init: float
    :param high_bound: Higher bound of the problem. Population will be clipped to this value.
    :type high_bound: float
    :param low_bound: Higher bound of the problem. Population will be clipped to this value.
    :type low_bound: float
    :param problem: The problem to be optimized.
    :param max_iter: Maximum number of iterations.
    :type max_iter: int
    :param pop_size: Population size.
    :type pop_size: int
    :param archive_size: Maximal capacity of archive.
    :type archive_size: int
    :param c: Constant used during adaptive F_m and CR_m procedure.
    :type c: float
    :param F_a_init: Initial archive F value
    :type F_a_init: float
    :param CR_a_init: Initial archive CR value
    :type CR_a_init: float
    :param tao_1: Probability of adjusting F_a
    :type tao_1: float
    :param tao_2: Probability of adjusting CR_a
    :type tao_2: float
    :return: Archive
    :rtype: list
    """
    # Initialize population, gather fitnesses and best solutions, and make archive
    pop = [np.random.uniform(low_bound, high_bound, size=(pop_size, problem.n_var)) for _ in range(problem.n_obj)]
    fitnesses = [problem.evaluate(pop[i]) for i in range(len(pop))]
    bests = [(np.argmax(fitnesses[i][:, i]), pop[i][np.argmax(fitnesses[i][:, i])].copy()) for i in
             range(len(fitnesses))]
    archive = ParetoArchive(archive_size)

    # Initiate the F and CR values for each population
    F_locs = np.full(shape=len(pop), fill_value=F_m_init)
    CR_locs = np.full(shape=len(pop), fill_value=CR_m_init)

    # initial archive update
    for i in range(len(pop)):
        archive.add(pop[i], fitnesses[i])

    # Initiate the F and CR values for archive population
    F_a = np.full(fill_value=F_a_init, shape=(archive_size, 1))
    CR_a = np.full(fill_value=CR_a_init, shape=(archive_size, 1))

    for _ in tqdm(range(max_iter)):
        # Update each subpopulation
        for j in range(len(pop)):
            # Generate F_m and CR_m
            F_m = F_locs[j] + 0.1 * np.random.standard_cauchy(size=(pop_size, 1))
            while len(F_m[F_m <= 0.0]) > 0:
                F_m = np.where(F_m <= 0.0, F_locs[0] + 0.1 * np.random.standard_cauchy(), F_m)
            F_m = F_m.clip(0.0, 1.0)

            CR_m = np.random.normal(loc=CR_locs[j], scale=0.1, size=(pop_size, 1))
            CR_m = CR_m.clip(0.0, 1.0)
            CR_m = np.tile(CR_m, reps=(1, problem.n_var))

            # Randomly select archive individuals
            tmp = np.random.choice(len(archive.items), size=pop_size, replace=True)
            archive_guides = np.array([archive.items[entry]['pos'] for entry in tmp])

            # Select random individuals from population for mutation
            rand_pop_vectors_1 = pop[j][gen_random_vector_idx(pop_size)]
            rand_pop_vectors_2 = pop[j][gen_random_vector_idx(pop_size)]

            # Generate mutant vector V_m
            best = np.tile(bests[j][1], reps=(pop_size, 1))
            V_m = pop[j] + F_m * (best - pop[j]) + \
                  F_m * (rand_pop_vectors_1 - rand_pop_vectors_2) + \
                  F_m * (archive_guides - pop[j])

            # Generate trial vector U_m
            rand_float_vec = np.random.uniform(0, 1, size=(pop_size, problem.n_var))
            rand_int_vec = np.random.randint(0, problem.n_var, size=(pop_size, problem.n_var))
            indexes = np.tile(np.array([i for i in range(problem.n_var)]), reps=(pop_size, 1))
            U_m = np.where(np.logical_or(rand_float_vec <= CR_m, indexes == rand_int_vec), V_m, pop[j])
            U_m = U_m.clip(low_bound, high_bound)

            # Evaluate U_m
            new_fitnesses = problem.evaluate(U_m)

            # Change population where fitnesses have improved
            better_idxs = np.where(new_fitnesses[:, j] > fitnesses[j][:, j])
            if better_idxs[0].size > 0:
                pop[j][better_idxs] = U_m[better_idxs]
                fitnesses[j][better_idxs] = new_fitnesses[better_idxs]

                # Gather the set of F and CR values that successfully improved fitness.
                S_F = F_m[better_idxs]
                S_CR = CR_m[better_idxs, 0]

                # Update the loc values
                F_locs[j] = (1 - c) * F_locs[j] + c * lehmer_mean(S_F)
                CR_locs[j] = (1 - c) * CR_locs[j] + c * lehmer_mean(S_CR)

                # Update global best solution
                bests[j] = (np.argmax(fitnesses[j][:, j]), pop[j][np.argmax(fitnesses[j][:, j])].copy())

        # Update archive population (if has more than 4 entries)
        archive_size = len(archive.items)
        if archive_size >= 4:
            # Gather archive positions
            A = np.array([entry['pos'] for entry in archive.items])

            # Generate F_a and CR_a
            F_rand = np.random.uniform(0.1, 1, size=(archive_size, 1))
            F_rand_cond = np.random.uniform(0, 1, size=(archive_size, 1))
            F_a_trunc = F_a[:archive_size]
            F_a[:archive_size] = np.where(F_rand_cond < tao_1, F_rand, F_a_trunc)
            F_a_trunc = F_a[:archive_size]

            CR_rand = np.random.uniform(0.1, 1, size=(archive_size, 1))
            CR_rand_cond = np.random.uniform(0, 1, size=(archive_size, 1))
            CR_a_trunc = CR_a[:archive_size]
            CR_a[:archive_size] = np.where(CR_rand_cond < tao_2, CR_rand, CR_a_trunc)
            CR_a_trunc = np.tile(CR_a[:archive_size], reps=(1, problem.n_var))

            # Generate mutant vector V_a
            rand_archive_vectors_1 = A[gen_random_vector_idx(archive_size)]
            rand_archive_vectors_2 = A[gen_random_vector_idx(archive_size)]
            rand_archive_vectors_3 = A[gen_random_vector_idx(archive_size)]

            V_a = rand_archive_vectors_1 + F_a_trunc * (rand_archive_vectors_2 - rand_archive_vectors_3)

            # Generate trial vector U_a
            rand_float_vec = np.random.uniform(0, 1, size=(archive_size, problem.n_var))
            rand_int_vec = np.random.randint(0, problem.n_var, size=(archive_size, problem.n_var))
            indexes = np.tile(np.array([i for i in range(problem.n_var)]), reps=(archive_size, 1))
            U_a = np.where(np.logical_or(rand_float_vec <= CR_a_trunc, indexes == rand_int_vec), V_a, A)
            U_a = U_a.clip(low_bound, high_bound)

            # Evaluate U_a
            A_new_fit = problem.evaluate(U_a)

            archive.add(U_a, A_new_fit)

        for i in range(len(pop)):
            archive.add(pop[i], fitnesses[i])
    return archive.items


np.seterr('raise')  # Used for debugging, and it seems code was repaired.
problem = get_problem("zdt1")
# Must change the problem to -f(x) since CMODE assumes maximization.
tmp = problem.evaluate
problem.evaluate = lambda x: -1 * tmp(x)
archive = CMODE(problem, 0.0, 1.0, 500, 20, 100, 0.1, 0.5, 0.5, 0.5, 0.9, 0.1, 0.1)

# Visualize the results of the archive
# Note: did entry['fit'] * -1 to revert the fitnesses back to their original values.
pf = problem.pareto_front()
Scatter(legend=True).add(pf, label="Pareto-front").add(np.array([entry['fit'] * -1 for entry in archive]),
                                                       label="Result").show()
