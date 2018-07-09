import lspi

import numpy as np

NUM_BASIS = 5
DEGREE = 3
DISCOUNT = .9
EXPLORE = 0
NUM_SAMPLES = 5000
MAX_ITERATIONS = 100000
MAX_STEPS = 100


class LearningMazeDomain():

    def __init__(self, height, width, reward_location, walls_location,
                 obstacles_location, initial_state=None, obstacles_transition_probability=.2, num_sample=NUM_SAMPLES):

        self.domain = lspi.domains.GridMazeDomain(height, width, reward_location,
                                                  walls_location, obstacles_location, initial_state, obstacles_transition_probability)

        sampling_policy = lspi.Policy(lspi.basis_functions.FakeBasis(4), DISCOUNT, 1)

        self.samples = []

        for i in xrange(num_sample):
            action = sampling_policy.select_action(self.domain.current_state())
            self.samples.append(self.domain.apply_action(action))

        self.random_policy_cumulative_rewards = np.sum([sample.reward for
                                                        sample in self.samples])

        self.solver = lspi.solvers.LSTDQSolver()

    def learn_proto_values_basis(self, num_basis=NUM_BASIS, discount=DISCOUNT,
                                 explore=EXPLORE, max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES, initial_policy=None):

        if initial_policy is None:
            initial_policy = lspi.Policy(lspi.basis_functions.ProtoValueBasis(
                self.domain.graph, 4, num_basis), discount, explore)

        learned_policy, distances = lspi.learn(self.samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        while (not absorb) and (steps_to_goal < max_steps):
            action = learned_policy.select_action(self.domain.current_state())
            sample = self.domain.apply_action(action)
            absorb = sample.absorb
            if absorb:
                print('Reached the goal in %d', steps_to_goal)
            steps_to_goal += 1
            samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_polynomial_basis(self, degree=DEGREE, discount=DISCOUNT,
                               explore=EXPLORE, max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES, initial_policy=None):

        if initial_policy is None:
            initial_policy = lspi.Policy(lspi.basis_functions.OneDimensionalPolynomialBasis(degree, 4), discount, explore)

        learned_policy, distances = lspi.learn(self.samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        while (not absorb) and (steps_to_goal < max_steps):
            action = learned_policy.select_action(self.domain.current_state())
            sample = self.domain.apply_action(action)
            absorb = sample.absorb
            if absorb:
                print('Reached the goal in %d', steps_to_goal)
            steps_to_goal += 1
            samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_node2vec_basis(self, dimension=NUM_BASIS, walk_length=30, num_walks=10, window_size=10,
                             p=1, q=1, epochs=1, discount=DISCOUNT, explore=EXPLORE, max_iterations=MAX_ITERATIONS,
                             max_steps=NUM_SAMPLES, initial_policy=None, edgelist ='node2vec/graph/grid6.edgelist'):

        if initial_policy is None:
            initial_policy = lspi.Policy(lspi.basis_functions.Node2vecBasis(
                edgelist, num_actions=4, transition_probabilities=self.domain.transition_probabilities,
                dimension=dimension,walk_length=walk_length, num_walks=num_walks, window_size=window_size,
                p=p, q=q, epochs=epochs), discount, explore)

        learned_policy, distances = lspi.learn(self.samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        while (not absorb) and (steps_to_goal < max_steps):
            action = learned_policy.select_action(self.domain.current_state())
            sample = self.domain.apply_action(action)
            absorb = sample.absorb
            if absorb:
                print('Reached the goal in %d', steps_to_goal)
            steps_to_goal += 1
            samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances
