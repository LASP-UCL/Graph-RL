# -*- coding: utf-8 -*-
"""Contains main interface to LSPI algorithm."""

from copy import copy

import numpy as np


def learn(data, initial_policy, solver, epsilon=10**-5, max_iterations=10):
    r"""Find the optimal policy for the specified data.

    Parameters
    ----------
    data:
        Generally a list of samples, however, the type of data does not matter
        so long as the specified solver can handle it in its solve routine. For
        example when doing model based learning one might pass in a model
        instead of sample data
    initial_policy: Policy
        Starting policy. A copy of this policy will be made at the start of the
        method. This means that the provided initial policy will be preserved.
    solver: Solver
        A subclass of the Solver abstract base class. This class must implement
        the solve method. Examples of solvers might be steepest descent or
        any other linear system of equation matrix solver. This is basically
        going to be implementations of the LSTDQ algorithm.
    epsilon: float
        The threshold of the change in policy weights. Determines if the policy
        has converged. When the L2-norm of the change in weights is less than
        this value the policy is considered converged
    max_iterations: int
        The maximum number of iterations to run before giving up on
        convergence. The change in policy weights are not guaranteed to ever
        go below epsilon. To prevent an infinite loop this parameter must be
        specified.

    Return
    ------
    Policy
        The converged policy. If the policy does not converge by max_iterations
        then this will be the last iteration's policy.

    Raises
    ------
    ValueError
        If epsilon is <= 0
    ValueError
        If max_iteration <= 0

    """
    if epsilon <= 0:
        raise ValueError('epsilon must be > 0: %g' % epsilon)
    if max_iterations <= 0:
        raise ValueError('max_iterations must be > 0: %d' % max_iterations)

    # this is just to make sure that changing the weight vector doesn't
    # affect the original policy weights
    curr_policy = copy(initial_policy)

    distance = float('inf')
    distances = []
    iteration = 0
    while distance > epsilon and iteration < max_iterations:
        iteration += 1
        new_weights = solver.solve(data, curr_policy)

        distance = np.linalg.norm(new_weights - curr_policy.weights)
        distances.append(distance)
        curr_policy.weights = new_weights

    return curr_policy, distances
