import math
import random

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from nds import ndomsort


def numpy_dominates(x, y):
    """
    Checks if x pareto-dominates y.

    :param x: a solution's fitnesses (numpy array)
    :param y: another solution's fitnesses (numpy array)
    """
    return np.logical_and((x >= y).all(axis=1), (x > y).any(axis=1))


def dominates(x, y):
    """
    Checks if x pareto-dominates y.

    :param x: a solution's fitnesses
    :type x: list[float]
    :param y: another solution's fitnesses
    :type y: list[float]
    :rtype: bool
    """
    assert len(x) == len(y)

    better = False
    for i in range(len(x)):
        if x[i] >= y[i]:
            if x[i] > y[i]:
                better = True
            continue
        else:
            return False

    return better


def density(items):
    """
    add_item_before_crowding should be False.

    Note: follows the format of items from class Archive, and
        add_item_before_crowding should be False in Archive.

    :param items:
    :return:
    """
    # First, calculate distance between points
    # Source: https://stackoverflow.com/questions/40996957/calculate-distance-between-numpy-arrays
    collected_objs = np.array([entry['fit'] for entry in items])

    euclidean_dist = cdist(collected_objs, collected_objs, metric='euclidean')

    # Sort distances in ascending order
    euclidean_dist = np.sort(euclidean_dist, axis=1)

    # k is set to sqrt of archive size
    k = int(math.sqrt(len(items)))

    tmp = 1 / (euclidean_dist[:, k] + 2)

    for i in range(len(items)):
        items[i]['dist'] = tmp[i]

    items.sort(key=lambda x: x["dist"], reverse=True)

    return items


def crowdingDistance(items):
    """
    Sorts the given list by crowding distance
    in ascending order. Thus, the first element
    will have the least distance, meaning
    it's the most crowded.

    This is an in-place sort.

    Note: follows the format of items from class Archive, and
        add_item_before_crowding should be False in Archive.

    :param items: List of items to be sorted by crowding distance.
    """
    assert len(items) > 0

    for item in items:
        item["dist"] = 0

    num_of_functions = len(items[0]['fit'])
    len_archive = len(items)
    for i in range(num_of_functions):
        # Sort archive by the fitness on objective i
        items.sort(key=lambda x: x["fit"][i])

        # Calculate distance of intermediate archive solutions
        for j in range(1, len(items) - 1):
            items[j]["dist"] += items[j + 1]["fit"][i] - items[j - 1]["fit"][i]

        items[0]["dist"] = items[len_archive - 1]["dist"] = float('inf')

    items.sort(key=lambda x: x["dist"])

    return items


def min_distance_indicator(items):
    """
    Calculates fitness of each solution in archive like explained in Section 3.1 of:

    Cui, Yingying, Xi Meng, and Junfei Qiao.
    "A multi-objective particle swarm optimization algorithm based on two-archive mechanism."
    Applied Soft Computing 119 (2022): 108532.
    """
    fitnesses = np.array([item['fit'] for item in items])
    scaler = MinMaxScaler(feature_range=(0, 1))
    fitnesses = scaler.fit_transform(fitnesses)

    for i in range(len(fitnesses)):
        shifted_fitnesses = fitnesses.copy()
        for j in range(len(fitnesses)):
            if i != j:
                shift_dimensions = np.where(shifted_fitnesses[j] > fitnesses[i])
                shifted_fitnesses[shift_dimensions] = fitnesses[shift_dimensions]


class Grid:
    """
    Creating a grid based on the max and min values in objective space (fitness).

    Note: follows the format of items from class Archive, and
    add_item_before_crowding should be True in Archive.

    Example (seen in code):
    -------
    | | | |
    -------
    | | | |
    -------
    | | | |
    -------
    where num_grid = 3. 3 represents the internal spaces each row or column.
    """

    def __init__(self, size):
        self.size = size
        self.item_div = {}

    def calculate(self, items):
        """
        :param items: List of items to be sorted by grid.
        """

        # Create the grid cutoffs
        objs = np.array([item['fit'] for item in items])
        min_ = np.min(objs, axis=0) - 0.001
        max_ = np.max(objs, axis=0) + 0.001

        cutoffs = np.linspace(max_, min_, self.size + 1)

        # Calculate the grid populations (density)
        self.item_div = {}
        item_grid_index = {}
        for i in range(len(items)):
            item_cutoff = [0] * cutoffs.shape[1]
            for row in cutoffs:
                for j in range(len(row)):
                    if items[i]['fit'][j] < row[j]:
                        item_cutoff[j] += 1
            tmp = 0
            for j in range(len(item_cutoff)):
                tmp += item_cutoff[j] * (self.size ** j)

            if str(tmp) not in item_grid_index:
                item_grid_index[str(tmp)] = [i]
            else:
                item_grid_index[str(tmp)].append(i)

            if str(tmp) not in self.item_div:
                self.item_div[str(tmp)] = [items[i]['pos']]
            else:
                self.item_div[str(tmp)].append(items[i]['pos'])

        # Get the most crowded grid index
        max_entry = []
        for k, v in item_grid_index.items():
            if len(v) > len(max_entry):
                max_entry = v

        # Randomly select index from most crowded grid
        rand_idx = random.sample(max_entry, 1)[0]

        # Swap first item in items with item in most crowded grid
        items[rand_idx], items[0] = items[0], items[rand_idx]

        return items


class Archive:
    def __init__(self,
                 capacity,
                 crowding_function=density,
                 track=None,
                 add_item_before_crowding=False,
                 allow_dominated=False):
        """
        Trackable information:
            - (default) 'pos': Position of solution
            - (default) 'fit': Fitness of solution
            - (default) 'dist': Crowding measure of solution (can be for density or crowding distance)
            - 'sum_fit': Sum of fitness
            - 'added_sol_idx': Returns the indices of solutions that were successfully added to archive (in add() function)

        Crowding functions:
            - Density
            - Crowding distance

        :param capacity: Maximum capacity of archive.
        :type capacity: int
        :param crowding_function: Crowding function used to remove archive entry when capacity is met.
        :param track: Additional information to track in archive
        :param add_item_before_crowding: True if item needs to be added to archive and then
            have crowding calculations complete. False if calculation is made first, and then solution replaced.
        :param allow_dominated: Whether to allow dominated solutions to be added to archive if not full,
            where non-dominated solutions have priority.
        """
        if track is None:
            track = []
        self.capacity = capacity
        self.items = []
        self.track = track
        self.crowding_function = crowding_function
        self.add_item_before_crowding = add_item_before_crowding
        self.allow_dominated = allow_dominated

    def add(self, swarm, fitneses):
        """
        This method is to be overridden by a super class to fit the
        wanted archive management system.
        """
        raise NotImplementedError


class ParetoArchive(Archive):
    def add(self, pop, fitnesses):
        """
        Attempt to add solutions from population to
        the archive in-place, while still only containing
        non-dominated solutions. If it's full, use the given
        crowding function to remove the one that's most crowded.

        :param pop: The population to be added.
        :param fitnesses: Fitnesses of population.
        """
        assert len(pop) == len(fitnesses)

        if 'added_sol_idx' in self.track:
            added_idxs = []

        for i in range(len(pop)):

            # Allow dominated solutions section
            if self.allow_dominated:
                newItem = {
                    "pos": pop[i].copy(),
                    "fit": fitnesses[i].copy(),
                    "dist": 0,
                }

                if 'sum_fit' in self.track:
                    newItem['sum_fit'] = np.sum(fitnesses[i])

                if 'added_sol_idx' in self.track:
                    added_idxs.append(i)

                self.items.append(newItem)

                if len(self.items) < self.capacity:
                    continue

                crowdingDistance(self.items)  # sort archive by crowding distance

                # Archive's at capacity, so first do non-dominated sort and get last front
                nds = ndomsort.non_domin_sort(self.items, get_objectives=lambda x: x['fit'])
                last_front_idx = max([int(val) for val in nds.keys()])
                last_front = list(nds[last_front_idx])

                # then, sort last front so first element is most crowded and remove
                last_front.sort(key=lambda x: x['dist'])
                for i in range(len(self.items)):
                    if (self.items[i]['pos'] == last_front[0]['pos']).all():
                        del self.items[i]
                        break

                continue

            # This code runs if dominated solutions aren't allowed in archive (default)
            dominated = []  # Archive entries dominated by particle
            for j in range(len(self.items)):
                if dominates(fitnesses[i], self.items[j]["fit"]):
                    # particle dominates archive entry
                    dominated.append(j)
                elif dominates(self.items[j]["fit"], fitnesses[i]):
                    # archive entry dominates particle
                    break
                elif (fitnesses[i] == self.items[j]["fit"]).all():
                    # the same entry is attempted to be added
                    break
            else:  # If inner loop wasn't broken, then code in here is executed
                # remove dominated archive entries
                self.items = [self.items[k] for k in range(len(self.items)) if k not in dominated]

                # the new item to be added to archive

                newItem = {
                    "pos": pop[i].copy(),
                    "fit": fitnesses[i].copy(),
                    "dist": 0,
                }

                if 'sum_fit' in self.track:
                    newItem['sum_fit'] = np.sum(fitnesses[i])

                if 'added_sols' in self.track:
                    added_idxs.append(i)

                if len(self.items) >= self.capacity:
                    if self.add_item_before_crowding:
                        self.items.append(newItem)

                    # Sort by crowding function
                    self.items = self.crowding_function(self.items)

                    if self.add_item_before_crowding:
                        # First element is most crowded, so delete it
                        del self.items[0]
                    else:
                        # First element is most crowded, so replace it
                        self.items[0] = newItem
                else:
                    self.items.append(newItem)

        if 'added_sol_idx' in self.track:
            return added_idxs
