# Metaheuristic Implementations #

Implementation of various metaheuristics. Not all may be completed, so double check this README to see.

Requirements listed in the requirements.txt file.

## CMODE ##

A cooperative multi-objective differential evolution algorithm, as explained in:

[Wang, Jiahai, Weiwei Zhang, and Jun Zhang.
"Cooperative differential evolution with multiple populations for multiobjective optimization."
IEEE Transactions on Cybernetics 46.12 (2015): 2848-2861.](https://www.doi.org/10.1109/TCYB.2015.2490669)

Note: improvements might still be made around the storage of solutions within the archive.

## CMOPSO (INCOMPLETE) ##

An attempted implementation of a competitive multi-objective particle swarm optimizer, as explained in:

[Zhang, Xingyi, et al. 
"A competitive mechanism based multi-objective particle swarm optimizer with fast convergence." 
Information Sciences 427 (2018): 63-76.](https://doi.org/10.1016/j.ins.2017.10.037)

The paper itself is vague on the implementation of the polynomial mutation operator. Several papers are listed,
each of which has a different mutation, making it difficult to determine the correct one. After testing with PyMoo
it doesn't appear like I've done this correctly. Results express the population has a similar shape of pareto front
to the known optimal, but is far from it.

![CMOPSO Pareto](images/CMOPSO-front.png)

