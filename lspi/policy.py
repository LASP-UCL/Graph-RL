# -*- coding: utf-8 -*-
"""LSPI Policy class used for learning and executing policy."""

import random

import numpy as np


class Policy(object):

    r"""Represents LSPI policy. Used for sampling, learning, and executing.

    The policy class includes an exploration value which controls the
    probability of performing a random action instead of the best action
    according to the policy. This can be useful during sample.

    It also includes the discount factor :math:`\gamma`, number of possible
    actions and the basis function used for this policy.

    Parameters
    ----------
    basis: BasisFunction
        The basis function used to compute :math:`phi` which is used to select
        the best action according to the policy
    discount: float, optional
        The discount factor :math:`\gamma`. Defaults to 1.0 which is valid
        for finite horizon problems.
    explore: float, optional
        Probability of executing a random action instead of the best action
        according to the policy. Defaults to 0 which is no exploration.
    weights: numpy.array or None
        The weight vector which is dotted with the :math:`\phi` vector from
        basis to produce the approximate Q value. When None is passed in
        the weight vector is initialized with random weights.
    tie_breaking_strategy: Policy.TieBreakingStrategy value
        The strategy to use if a tie occurs when selecting the best action.
        See the :py:class:`lspi.policy.Policy.TieBreakingStrategy`
        class description for what the different options are.

    Raises
    ------
    ValueError
        If discount is < 0 or > 1
    ValueError
        If explore is < 0 or > 1
    ValueError
        If weights are not None and the number of dimensions does not match
        the size of the basis function.
    """

    class TieBreakingStrategy(object):

        """Strategy for breaking a tie between actions in the policy.

        FirstWins:
            In the event of a tie the first action encountered with that
            value is returned.
        LastWins:
            In the event of a tie the last action encountered with that
            value is returned.
        RandomWins
            In the event of a tie a random action encountered with that
            value is returned.
        """

        FirstWins, LastWins, RandomWins = range(3)

    def __init__(self, basis, discount=1.0,
                 explore=0.0, weights=None,
                 tie_breaking_strategy=TieBreakingStrategy.RandomWins):
        """Initialize a Policy."""
        self.basis = basis

        if discount < 0.0 or discount > 1.0:
            raise ValueError('discount must be in range [0, 1]')

        self.discount = discount

        if explore < 0.0 or explore > 1.0:
            raise ValueError('explore must be in range [0, 1]')

        self.explore = explore

        if weights is None:
            self.weights = np.random.uniform(-1.0, 1.0, size=(basis.size(),))
        else:
            if weights.shape != (basis.size(), ):
                raise ValueError('weights shape must equal (basis.size(), 1)')
            self.weights = weights

        self.tie_breaking_strategy = tie_breaking_strategy

    def __copy__(self):
        """Return a copy of this class with a deep copy of the weights."""
        return Policy(self.basis,
                      self.discount,
                      self.explore,
                      self.weights.copy(),
                      self.tie_breaking_strategy)

    def calc_q_value(self, state, action):
        """Calculate the Q function for the given state action pair.

        Parameters
        ----------
        state: numpy.array
            State vector that Q value is being calculated for. This is
            the s in Q(s, a)
        action: int
            Action index that Q value is being calculated for. This is
            the a in Q(s, a)

        Return
        ------
        float
            The Q value for the state action pair

        Raises
        ------
        ValueError
            If state's dimensions do not conform to basis function expectations
        ValueError
            If action is outside of the range of valid action indexes

        """
        if action < 0 or action >= self.basis.num_actions:
            raise IndexError('action must be in range [0, num_actions)')

        return self.weights.dot(self.basis.evaluate(state, action))

    def best_action(self, state):
        """Select the best action according to the policy.

        This calculates argmax_a Q(state, a). In otherwords it returns
        the action that maximizes the Q value for this state.

        Parameters
        ----------
        state: numpy.array
            State vector.
        tie_breaking_strategy: TieBreakingStrategy value
            In the event of a tie specifies which action the policy should
            return. (Defaults to random)

        Returns
        -------
        int
            Action index

        Raises
        ------
        ValueError
            If state's dimensions do not match basis functions expectations.

        """
        q_values = [self.calc_q_value(state, action)
                    for action in range(self.basis.num_actions)]

        best_q = float('-inf')
        best_actions = []
        for action, q_value in enumerate(q_values):
            if q_value > best_q:
                best_actions = [action]
                best_q = q_value
            elif q_value == best_q:
                best_actions.append(action)

        if self.tie_breaking_strategy == Policy.TieBreakingStrategy.FirstWins:
            return best_actions[0]
        elif self.tie_breaking_strategy == Policy.TieBreakingStrategy.LastWins:
            return best_actions[-1]
        else:
            return random.choice(best_actions)

    def select_action(self, state):
        """With random probability select best action or random action.

        If the random number is below the explore value then pick a random
        value otherwise pick the best action according to the basis and
        policy weights.

        Parameters
        ----------
        state: numpy.array
            State vector

        Returns
        -------
        int
            Action index

        Raises
        ------
        ValueError
            If state's dimensions do not match basis functions expectations.

        """
        if random.random() < self.explore:
            return random.choice(range(self.basis.num_actions))
        else:
            return self.best_action(state)

    def calc_v_value(self, state):

        best_action = self.best_action(state)

        return self.calc_q_value(state, best_action)

    @property
    def num_actions(self):
        r"""Return number of possible actions.

        This number should always match the value stored in basis.num_actions.

        Return
        ------
        int
            Number of possible actions. In range [1, :math:`\infty`)

        """
        return self.basis.num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        This number should always match the value stored in basis.num_actions.

        Parameters
        ----------
        value: int
            Value to set num_actions to. Must be >= 1

        Raises
        ------
        ValueError
            If value is < 1

        """
        self.basis.num_actions = value
