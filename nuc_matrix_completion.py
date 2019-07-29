#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cvxpy as cp
import numpy as np

#nuclear norm minimization: minimizes nuclear norm with constraints that observed matrix r the same
class nuc_mc():
    def __init__(self, A, epsilon, max_iter=5000):
        self.A = A
        self.epsilon = epsilon
        self.max_iter = max_iter
  
    def complete(self, unobs):
        m,n = self.A.shape
        obs_entries = ~unobs
        
        X = cp.Variable((m,n))
        #make sure to replace nan with A
        error = cp.abs(cp.multiply(np.nan_to_num(self.A), obs_entries)-cp.multiply(X, obs_entries)) 
        constraints = [error <= self.epsilon]
        
        #set objective
        objective = cp.Minimize(cp.norm(X, "nuc"))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, max_iters = self.max_iter, use_indirect=False)
        
        return(prob.value, X.value)

