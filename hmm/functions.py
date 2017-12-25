import numpy as np

from math import log
from collections import defaultdict
from side_functions import sumLogAddExp, returnInitializedMatrices

def forwardMatFunc(obs_vec, O_dict, T_dict, prior_Y_dict, evaluate=True):
    T = len(obs_vec)
    states = T_dict.keys()
    N = len(states)
    output_forw_mat = np.zeros((N,T))

    #Calculate first column
    for i, state in enumerate(states):
        output_forw_mat[i,0] = O_dict[state][obs_vec[0]] + prior_Y_dict[state]
    
    #Calculate remaining columns.
    #state is my landing state
    #t-1 is the index I have to go and look for
    for t, sing_obs in enumerate(obs_vec[1:]):
        for i, state in enumerate(states):
            sum_previous_comp = [output_forw_mat[k,t] + T_dict[states[k]][state] for k in range(N)]
            tot_log_prob_sum = sumLogAddExp(sum_previous_comp)
            output_forw_mat[i,(t+1)] = tot_log_prob_sum + O_dict[state][sing_obs]

    if evaluate == "True":
    	return sumLogAddExp(output_forw_mat[:,T-1])
    else:
    	return output_forw_mat


def backwardMatFunct(obs_vec, O_dict, T_dict, prior_Y_dict, evaluate=True):
    
    T = len(obs_vec)
    states = T_dict.keys()
    N = len(states)
    output_back_mat = np.zeros((N,T))
    
    #Set last column
    for i, state in enumerate(states):
        output_back_mat[i,T-1] = log(1)
        
    #Set all other columns
    for t in range(T-1)[::-1]:
        for i, state in enumerate(states):
            sum_forward_logprob = [output_back_mat[k,t+1] + 
                                 T_dict[state][states[k]] + 
                                 O_dict[states[k]][obs_vec[t+1]] for k in range(N)]
            sum_state_logprob = sumLogAddExp(sum_forward_logprob)
            output_back_mat[i,t] = sum_state_logprob
    
    if evaluate:
        eval_vec = [output_back_mat[k,0] + O_dict[states[k]][obs_vec[0]] + prior_Y_dict[states[k]] for k in range(N)]
        eval_logprob = sumLogAddExp(eval_vec)
        return output_back_mat, eval_logprob
    else:
        return output_back_mat


def ViterbiFunc(obs_vec, O_dict, T_dict, prior_Y_dict, evaluate=True):
    T = len(obs_vec)
    states = T_dict.keys()
    M = len(states)
    B_mat = np.zeros((T,M))
    V_dict = defaultdict(lambda:{})
    
    for i, state in enumerate(states):
        B_mat[0,i] = prior_Y_dict[state] + O_dict[state][obs_vec[0]]
        V_dict[0][i] = [state]
    
    for t, obs in enumerate(obs_vec[1:]):
        for i, state in enumerate(states):
            eval_vec = [B_mat[t,k] + T_dict[states[k]][state] + O_dict[state][obs] for k in range(M)]
            argmax = eval_vec.index(max(eval_vec))
            B_mat[t+1,i] = eval_vec[argmax]
            V_dict[t+1][i] = V_dict[t][argmax] + [state]
    
    if evaluate:
        last_row = B_mat[T-1, :].tolist()
        argmax_last_row = last_row.index(max(last_row))
        most_likely_seq = V_dict[T-1][argmax_last_row]
        return B_mat, V_dict, most_likely_seq
    else:
        return B_mat, V_dict


def baumWelch(Nstates, O_categ, obs_vec_training, it = 20000):
    (pi_distrib_initial, O_dict, T_dict) = returnInitializedMatrices(Nstates, O_categ)
    
    old_T_mat = np.zeros((Nstates, Nstates))
    for i in range(Nstates):
        for j in range(Nstates):
            old_T_mat[i,j] = T_dict[i][j]
    old_O_mat = np.zeros((Nstates, len(O_categ)))
    for i in range(Nstates):
        for j in range(len(O_categ)):
            old_O_mat[i,j] = O_dict[i][O_categ[j]]
    it_count = 0
    
    while it_count < it:
        it_count +=1
        gamma_mat = np.zeros((Nstates, len(obs_vec_training)))
        xi_mat = np.zeros((Nstates, Nstates, len(obs_vec_training)))
        forw_mat = forwardMatFunc(obs_vec_training, O_dict, T_dict, pi_distrib_initial)
        back_mat = backwardMatFunct(obs_vec_training, O_dict, T_dict, pi_distrib_initial, evaluate=False)
        
        new_pi_distr =np.zeros((Nstates, 1))
        new_T_distr =np.zeros((Nstates, Nstates))
        new_O_distr = np.zeros((Nstates, len(O_categ)))
        
        for i in T_dict.keys():
            for t in range(len(obs_vec_training)-2):
                gamma_mat[i,t] = forw_mat[i,t]*back_mat[i,t]/np.dot(forw_mat[:,t].T,back_mat[:,t])
                for j in T_dict.keys():
                    ind_obs_vec_training = O_categ.index(obs_vec_training[t+1])
                    xi_mat_num = forw_mat[i,t]*old_T_mat[i,j]*old_O_mat[j,ind_obs_vec_training]*back_mat[j,t+1]
                    xi_mat_den = 0
                    for i1 in T_dict.keys():
                        for j1 in T_dict.keys():
                            xi_mat_den += forw_mat[i1,t]*old_T_mat[i1,j1]*old_O_mat[j1,ind_obs_vec_training]*back_mat[j1,t+1]
                    xi_mat[i,j,t] = xi_mat_num/xi_mat_den
        
        new_pi_distr = gamma_mat[:,0]
        new_T_distr =(np.sum(xi_mat[:,:,range(len(obs_vec_training)-2)], axis=2)/
                      np.sum(gamma_mat[:,range(len(obs_vec_training)-2)], axis=1)).T
        for i in T_dict.keys():
            for j, cat in enumerate(O_categ):
                gamma_mat_row = [ g if obs_vec_training[ind]==cat else 0 for ind, g in enumerate(gamma_mat[i,:])]
                new_O_distr[i,j] = np.sum(gamma_mat_row) / np.sum(gamma_mat[i,:])
        
        if it_count%100==0:
            print(np.linalg.norm(new_T_distr-old_T_mat), 'iteration', it_count)
        
        old_T_mat = new_T_distr
        old_O_mat = new_O_distr
        
                    
    return(new_pi_distr,new_T_distr, new_O_distr)