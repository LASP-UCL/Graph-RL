# -*- coding: utf-8 -*-
"""Contains class representing an LSPI sample."""


class Sample(object):

    """Represents an LSPI sample tuple ``(s, a, r, s', absorb)``.

    Parameters
    ----------

    state : numpy.array
        State of the environment at the start of the sample.
        ``s`` in the sample tuple.
        (The usual type is a numpy array.)
    action : int
        Index of action that was executed.
        ``a`` in the sample tuple
    reward : float
        Reward received from the environment.
        ``r`` in the sample tuple
    next_state : numpy.array
        State of the environment after executing the sample's action.
        ``s'`` in the sample tuple
        (The type should match that of state.)
    absorb : bool, optional
        True if this sample ended the episode. False otherwise.
        ``absorb`` in the sample tuple
        (The default is False, which implies that this is a
        non-episode-ending sample)


    Assumes that this is a non-absorbing sample (as the vast majority
    of samples will be non-absorbing).

    This class is just a dumb data holder so the types of the different
    fields can be anything convenient for the problem domain.

    For states represented by vectors a numpy array works well.

    """

    def __init__(self, state, action, reward, next_state, absorb=False):
        """Initialize Sample instance."""
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.absorb = absorb

    def __repr__(self):
        """Create string representation of tuple."""
        return 'Sample(%s, %s, %s, %s, %s)' % (self.state,
                                               self.action,
                                               self.reward,
                                               self.next_state,
                                               self.absorb)
