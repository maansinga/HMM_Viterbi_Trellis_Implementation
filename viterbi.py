'''
@Author  Sree Teja Simha Gemaraju**
@Email   Sreetejasimha.gemaraju@utdallas.edu**
@Netid   sxg177330**
'''

# coding: utf-8

# In[2]:


import sys
import numpy as np


# In[3]:


# Observation Matrix
#              Normal         Cold         Dizzy
#---------------------------------------------------
# Healthy       0.1            0.4           0.5
# Fever         0.6            0.3           0.1
#
#(RowHeader = states)
#(ColHeader = observations)
# P(Observation|State)
B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
B


# In[10]:


#             Healthy           Fever
#-------------------------------------------------
# <s>          0.6              0.4
# Healthy      0.7              0.3
# Fever        0.5              0.5
#
#(RowHeader = Start State of transition)
#(ColHeader = End State of transition)

S = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3],[0.5, 0.5]])
A


# In[30]:


smap = {
    'H': 0,
    'F': 1,
}

ismap = {
    0: 'H',
    1: 'F',
}

omap = {
    'N': 0,
    'C': 1,
    'D': 2
}


# In[42]:


def viterbi(obs, model):
    (A, B) = model
    T = len(obs) #No. of observations
    N = A.shape[0]  #No. of states
    
    viterbi = np.zeros((N, T+1), dtype=np.float64)
    back_track = np.zeros((N, T), dtype=np.int)
    
    # Setting up base case    
    pos = 0
    viterbi[0,0] = 1
    for s in range(N):
        # transit from <s> to state[s] X P(Obs|state)
        viterbi[s,0] = S[s]*B[s, omap[obs[pos] ]]
        
    # Iterating over input character
    for j in range(T-1):
        # Iterating over probabilities of outcomes
        for s in range(N):
            max_array = []
            # Computing new information
            for k in range(N):
                val = A[k, s]*viterbi[k, j]*B[s, omap[obs[j+1]]]
                max_array.append(val)

            # Updating information into array
            max_array = np.array(max_array)
            max_val = np.max(max_array)
            max_state = np.argmax(max_array)
            viterbi[s, j+1] = max_val
            back_track[s, j] = max_state

    final_pointer = np.argmax(viterbi[:,-1])
    
    last = final_pointer
    stack = []
    # Back tracking over results
    for i in range(1, back_track.shape[1]+1):
        last = back_track[:, -i][last]
        stack += [last]

    stack.reverse()
    
    # Returning results
    return stack
    


# In[ ]:


if __name__ == "__main__":
    if(len(sys.argv)<2):
        print("USAGE: python viterbi.py <observables sequence>")
        exit(0)
    states = viterbi(sys.argv[1], (A, B))        
    print(("%c"*len(states))%tuple([ismap[state] for state in states]))

