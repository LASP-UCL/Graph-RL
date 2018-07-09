import numpy as np
import scipy.optimize as optimization
from learning_maze import LearningMazeDomain
import lspi
import matplotlib.pyplot as plt


def func(params, xdata, ydata):
    return ydata - np.matmul(xdata, params)


def mse(params, xdata, ydata):
    return sum(np.square(ydata - np.matmul(xdata, params))) / len(xdata)


def least_squares(basis, values, weights):
    params, _ = optimization.leastsq(func, weights, args=(basis, values))

    error = mse(params, basis, values)

    return params, error


def example_grid_maze(plotV=True):
    height = 10
    width = 10
    reward_location = 9
    initial_state = None  # np.array([25])
    obstacles_location = [14, 13, 24, 23, 29, 28, 39, 38]  # range(height*width)
    walls_location = [50, 51, 52, 53, 54, 55, 56, 74, 75, 76, 77, 78, 79]
    obstacles_transition_probability = .2
    maze = LearningMazeDomain(height, width, reward_location, walls_location, obstacles_location, initial_state,
                              obstacles_transition_probability, num_sample=2000)

    def value_iteration(G, finish_state, obstacles, walls):
        V = [0] * G.N
        R = [0] * G.N
        R[finish_state] = 100
        gamma = 0.9
        success_prob = [1] * G.N
        for i in obstacles:
            success_prob[i] = obstacles_transition_probability
        for i in walls:
            success_prob[i] = .0
        epsilon = .0001
        diff = 100
        iterations = 0
        while diff > epsilon:
            iterations = iterations + 1
            diff = 0
            for s in xrange(G.N):
                if s == finish_state:
                    max_a = success_prob[s] * R[s]
                else:
                    max_a = float('-inf')
                    for s_prime in G.W.getcol(s).nonzero()[0]:
                        new_v = success_prob[s] * (R[s] + gamma * V[s_prime])
                        if new_v > max_a:
                            max_a = new_v
                diff = diff + abs(V[s] - max_a)
                V[s] = max_a
        print "number of iterations in Value Iteration:"
        print iterations
        return V

    V = value_iteration(maze.domain.graph, reward_location, obstacles_location, walls_location)

    if plotV:
        fig, ax = plt.subplots(1, 1)
        maze.domain.graph.plot_signal(np.array(V), vertex_size=60, ax=ax)
        plt.savefig('graphs/simpleMaze_trueV.pdf')
        plt.close()

    return maze, V


def compute_ProtoValueBasis(maze, num_basis=30, weighted_graph=False, lap_type='combinatorial'):
    if weighted_graph:
        graph = maze.domain.weighted_graph
    else:
        graph = maze.domain.graph

    basis = lspi.basis_functions.ProtoValueBasis(graph, 4, num_basis, lap_type)

    all_basis = []

    for state in range(maze.domain.graph.N):
        all_basis.append(basis.graph.U[state, 1:basis.num_laplacian_eigenvectors + 1])

    return all_basis


def compute_node2VecBasis(maze, dimension=30, walk_length=30, num_walks=10, window_size=10, p=1, q=1, epochs=1):
    basis = lspi.basis_functions.Node2vecBasis('node2vec/graph/grid10.edgelist', num_actions=4,
                                               transition_probabilities=maze.domain.transition_probabilities,
                                               dimension=dimension, walk_length=walk_length, num_walks=num_walks,
                                               window_size=window_size, p=p, q=q, epochs=epochs)

    all_basis = []

    for state in range(maze.domain.graph.N):
        all_basis.append(basis.model[str(state)])

    return all_basis


def plot_values(graph, basis, params, save=False, file_name='approx_v.pdf'):
    if graph is None:
        raise ValueError('graph cannot be None')

    if graph.N != len(basis):
        raise ValueError('graph.N and len(basis) must be equal')
    approx_values = np.matmul(basis, params)

    fig, ax = plt.subplots(1, 1)
    graph.plot_signal(approx_values, vertex_size=60, ax=ax)

    if not save:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()
