import numpy as np


def sumLogAddExp(array):
    output = array[0]
    for i in range(1, len(array)):
        output = np.logaddexp(output, array[i])
    return output


def returnInitializedMatrices(N_states, O_categ):
    states = range(N_states)
    
    pi_distrib_initial = {st: 1.0/N_states for st in states}
    O_dict = {}
    for st in states:
        rand_vec = np.random.uniform(0,1, len(O_categ))
        rand_vec = rand_vec/sum(rand_vec)
        O_dict[st] = {obs: rand_vec[i] for i,obs in enumerate(O_categ)}
    T_dict = {}
    for st in states:
        rand_vec = np.random.uniform(0,1, len(states))
        rand_vec = rand_vec/sum(rand_vec)
        T_dict[st] = {st1: rand_vec[i] for i,st1 in enumerate(states)}
        
    return pi_distrib_initial, O_dict, T_dict