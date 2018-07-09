# Graph RL

## Prerequisites 
numpy, matplotlib, gensim, pygsp

## Code structure


* GraphRL/ 
    - [lspi/](#lspi)
        - [basis_functions.py](#basis_funciton.py)
        - [domains.py](#domains.py)
        - [lspi.py](#lspi.py)
        - [node2vec.py](#node2vec.py)
        - [policy.py](#policy.py)
        - [sample.py](#sample.py)
        - [solvers.py](#solvers.py)
    - [node2vec.py/](#node2vec.py)
    - [learning_maze.py](#learning_maze.py)
    - [optimise.py](#optimise.py)
    - [PVF_simulation.py](#pvf_simulation.py)


## lspi
The code in this folder was built on top of this [LSPI python package](https://pythonhosted.org/lspi-python/autodoc/modules.html]). 

Least Squares Policy Iteration (LSPI) implementation.

Implements the algorithms described in the paper

["Least-Squares Policy Iteration.” Lagoudakis, Michail G., and Ronald Parr. Journal of Machine Learning Research 4, 2003.](https://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf)

The implementation is based on the [Matlab implementation](http://www.cs.duke.edu/research/AI/LSPI/) provided by the authors.

#### basis_funciton.py
This file implements the abstract class BasisFunction. A basis function is a function
that takes in a state vector and an action index and returns a vector of features. 

A few specific types of basis functions are further implemented in this file:
* __Fake Basis Funtion__: it simply ignores all inputs an returns a constant basis vector. 
Can be useful for random sampling.
* __One Dimensional Polynomial Basis Funcion__: simple polynomial features for a state with one dimension.
* __Radial Basis Function__: the Gaussian multidimensional radial basis function (RBF).
* __Proto-Value Basis Function__: the PVFs as described in [Mahadevan and Maggioni's work](http://www.jmlr.org/papers/volume8/mahadevan07a/mahadevan07a.pdf).
* __Node2Vec__: automatically learnt basis function using the [node2vec algorithm](https://dl.acm.org/citation.cfm?id=2939672.2939754).

#### domains.py
Contains example domains that LSPI works on. In particular, it implements the __grid maze domain__ 
in which the state space is a set of nodes on a N1 by N2 grid. Most nodes are always accessible 
(= rooms with 1. transition probability), some
nodes might be inaccessible (= walls with 0. transition probability), and some
nodes might be difficult to access (= obstacles with p transition probability
0 < p < 1). There is one absorbing goal state that gives reward of 100;
all other states are non absorbing and do not give any reward.
#### lspi.py
Contains the main interface to LSPI algorithm.

#### node2vec.py
Implements the node2vec algorithm to learn node embeddings.

#### policy.py
Contains the LSPI policy class used for learning, executing and sampling policies.

#### sample.py
Contains the Sample class that respresents an LSPI sample tuple ``(s, a, r, s', absorb)`` : (state, action, observed reward, future state, absorbing state)

#### solvers.py
Implementation of LSTDQ solver with standard matrix solver (the algorithm from Figure 5 of the [LSPI paper]((https://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf))) 

### learning_maze.py
This class implements maze environments such as the one depicted below.

![alt text](https://github.com/Sephora-M/graph-rl/blob/master/twowalls_maze.png)

In such environment, the green states are accessible rooms, the dark purple states are strict walls and the
yellow state is the goal state. An agent can be initially placed in any accessible state and it aims
at reaching the goal state.

The class implements methods for learning the PVF basis functions as well as polynomial and
node2vec basis functions. 

### optimise.py
This code implements functions for approximating the true value function using least-square 
optimisation from the basis functions (that is, by minimising the mean squared error). 
Note that this is only possible in small environment since 
the true value function computed using value iteration algorithm.   

For the environment described in the previous subsection, the goal is to approximate the following 
value functions as closely as possible.

![alt text](https://github.com/Sephora-M/graph-rl/blob/master/twowalls_value.png)

### PVF_simulation.py
Code for running simulations. It can be called directly from the terminal as follows \
``>> python PVF_simulations.py``

Running this code will run LSPI algorithm on simple maze envrionments (no walls, no obstacles)
of different size and with different LSPI hyperparameters. All the parameters are defined as
constants at the beginning of the file and can be changed accordingly. After It will save plots 
(averages over 20 runs of the number of steps to reach the goal and the cumulative reward) in a folder `plots/` 
(make sure such a folder exists in the `GraphRL/` folder). 

