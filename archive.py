import math

import numpy as np
from scipy.spatial.distance import cdist


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


class Archive:
    def __init__(self, capacity):
        """
        :param capacity: Maximum capacity of archive.
        :type capacity: int
        """
        self.capacity = capacity
        self.items = []

    def add(self, swarm, fitneses):
        """
        This method is to be overridden by a super class to fit the
        wanted archive management system.
        """
        raise NotImplementedError


class ParetoArchive(Archive):
    """
        A holder of non-dominated solutions.

        Non-dominated solutions are added in
        self.items as follows:

        {
            "pos": position,
            "fit": fitnesses on all scoring functions,
            "dist": density value
        }

        self.items is a list of dictionaries.
    """

    def add(self, pop, fitnesses):
        """
        Attempt to add solutions from population to
        the archive, while still only containing
        non-dominated solutions. If it's full,
        use density calculations to remove the one
        that's most crowded.

        :param pop: The population to be added.
        :param fitnesses: Fitnesses of population.
        """
        assert len(pop) == len(fitnesses)

        for i in range(len(pop)):
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

                if len(self.items) >= self.capacity:
                    # Sort by density calculation
                    self.items = density(self.items)

                    # First element is most crowded, so replace it
                    self.items[0] = newItem
                else:
                    self.items.append(newItem)
