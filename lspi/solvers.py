# -*- coding: utf-8 -*-
"""Contains main LSPI method and various LSTDQ solvers."""

import abc
import logging

import numpy as np

import scipy.linalg


class Solver(object):

    r"""ABC for LSPI solvers.

    Implementations of this class will implement the various LSTDQ algorithms
    with various linear algebra solving techniques. This solver will be used
    by the lspi.learn method. The instance will be called iteratively until
    the convergence parameters are satisified.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def solve(self, data, policy):
        r"""Return one-step update of the policy weights for the given data.

        Parameters
        ----------
        data:
            This is the data used by the solver. In most cases this will be
            a list of samples. But it can be anything supported by the specific
            Solver implementation's solve method.
        policy: Policy
            The current policy to find an improvement to.

        Returns
        -------
        numpy.array
            Return the new weights as determined by this method.

        """
        pass  # pragma: no cover


class LSTDQSolver(Solver):

    """LSTDQ Implementation with standard matrix solvers.

    Uses the algorithm from Figure 5 of the LSPI paper. If the A matrix
    turns out to be full rank then scipy's standard linalg solver is used. If
    the matrix turns out to be less than full rank then least squares method
    will be used.

    By default the A matrix will have its diagonal preconditioned with a small
    positive value. This will help to ensure that even with few samples the
    A matrix will be full rank. If you do not want the A matrix to be
    preconditioned then you can set this value to 0.

    Parameters
    ----------
    precondition_value: float
        Value to set A matrix diagonals to. Should be a small positive number.
        If you do not want preconditioning enabled then set it 0.
    """

    def __init__(self, precondition_value=.1):
        """Initialize LSTDQSolver."""
        self.precondition_value = precondition_value

    def solve(self, data, policy):
        """Run LSTDQ iteration.

        See Figure 5 of the LSPI paper for more information.
        """
        k = policy.basis.size()
        a_mat = np.zeros((k, k))
        np.fill_diagonal(a_mat, self.precondition_value)

        b_vec = np.zeros((k, 1))

        for sample in data:
            phi_sa = (policy.basis.evaluate(sample.state, sample.action)
                      .reshape((-1, 1)))

            if not sample.absorb:
                best_action = policy.best_action(sample.next_state)
                phi_sprime = (policy.basis
                              .evaluate(sample.next_state, best_action)
                              .reshape((-1, 1)))
            else:
                phi_sprime = np.zeros((k, 1))

            a_mat += phi_sa.dot((phi_sa - policy.discount*phi_sprime).T)
            b_vec += phi_sa*sample.reward

        a_rank = np.linalg.matrix_rank(a_mat)
        if a_rank == k:
            w = scipy.linalg.solve(a_mat, b_vec)
        else:
            logging.warning('A matrix is not full rank. %d < %d', a_rank, k)
            w = scipy.linalg.lstsq(a_mat, b_vec)[0]
        return w.reshape((-1, ))
