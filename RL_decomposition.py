import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output


class rl_mc():
    def __init__(self, A, max_iter=30, eps=0.001):
        self.A = A
        self.max_iter = max_iter
        self.eps = eps
        self.m, self.n = A.shape
        
    def set_params(self, k, unobs, l=0):
        self.k = k
        self.unobs = unobs
        self.l = l
    
    #solve using alternating least square with constraints
    def solve(self, unobs, plot=False, progress=False):
        m,n = self.m, self.n
        k = self.k 
        
        obs_entries = ~unobs
        R = np.random.normal(0,1,(k,n))
        R_b = np.random.normal(0,1, n).reshape((1,n))
        L_b = np.random.normal(0,1, m).reshape((m,1))

        err = []

        lam = cp.Parameter(nonneg=True)
        lam.value = self.l
        for i in range(self.max_iter):
            if i%2 == 0:
                L = cp.Variable((m,k))
                L_b = cp.Variable((m,1))
            else:
                R = cp.Variable((k,n))
                R_b = cp.Variable((1,n))

            penalty = cp.pnorm(R, 2)+cp.pnorm(R, 2)+cp.norm(L_b, 2)+cp.norm(R_b, 2)
            #make sure to replace nan with A

            t1 = L*R
            t2 = np.ones(m).reshape((m,1)) * R_b
            t3 = L_b * np.ones(n).reshape((1,n))
            A_est = t1 + t2 + t3
            error = cp.multiply(np.nan_to_num(self.A), obs_entries)-cp.multiply(A_est, obs_entries)

            #set objective
            objective = cp.Minimize(cp.norm(error, "fro") + lam*penalty)
            
            prob = cp.Problem(objective)
            prob.solve(solver=cp.SCS, use_indirect=False)
            
            if prob.status != cp.OPTIMAL:
                pass
            
            if i % 2 == 0:
                L = L.value
                L_b = L_b.value
            else:
                R = R.value
                R_b = R_b.value
                                    
            reconstruction_error = self.recovery_error(self.construct_matrix((L, R, R_b, L_b)))
            err.append(reconstruction_error)

            if plot:
                plt.clf()
                plt.plot(np.arange(i+1), err)
            
            if progress:
                clear_output(wait=True)
                print('Iter %d/%d - Loss: %.3f'  % (
                    i + 1, self.max_iter, reconstruction_error))
            
            if i > 0:
                if abs(reconstruction_error-temp) < self.eps:
                    break
        
            temp = reconstruction_error
                    
        return(L, R, R_b, L_b)
        
    #cross validate for best lambda values
    def cross_validate_l(self, lambdas, n_cv=4, plot=False):
        m,n = self.m, self.n
        
        cv_errors = []
        #bootstrapping n_cv cross validation sets

        cv_sets = []
        for i in range(n_cv):
            unobs = np.copy(self.unobs)
            obs_indices = np.where(~unobs)
            cv_indices = np.random.choice(np.arange(len(obs_indices[0])), size=int(len(obs_indices[0])/n_cv), replace=False)
            test_x = obs_indices[0][cv_indices]
            test_y = obs_indices[1][cv_indices]
            unobs[test_x, test_y] = True
            cv_error = []
            for l in lambdas:
                self.l = l
                try:
                    res = self.construct_matrix(self.solve(unobs))
                    cv_error.append(np.mean(np.square(self.A[test_x, test_y] - res[test_x, test_y])))
                except:
                    cv_error.append(np.nan)
            cv_errors.append(cv_error)
        
        MSE = np.nanmean(cv_errors, axis=0)
        if plot:
            plt.plot(lambdas, MSE)
        return(cv_errors)
    
    #cross validate for best rank
    def cross_validate_k(self, ks, n_cv=4, plot=False):
        m,n = self.m, self.n
        
        cv_errors = []
        #bootstrapping n_cv cross validation sets

        cv_sets = []
        for i in range(n_cv):
            unobs = np.copy(self.unobs)
            obs_indices = np.where(~unobs)
            cv_indices = np.random.choice(np.arange(len(obs_indices[0])), size=int(len(obs_indices[0])/n_cv), replace=False)
            test_x = obs_indices[0][cv_indices]
            test_y = obs_indices[1][cv_indices]
            unobs[test_x, test_y] = True
            cv_error = []
            for k in ks:
                self.k = k
                try:
                    res = self.construct_matrix(self.solve(unobs))
                    cv_error.append(np.mean(np.square(self.A[test_x, test_y] - res[test_x, test_y])))
                except:
                    cv_error.append(np.nan)
            cv_errors.append(cv_error)
        
        MSE = np.nanmean(cv_errors, axis=0)
        if plot:
            plt.plot(ks, MSE)
        return(cv_errors)
    
    #get recovery error of the approximated matrix
    def recovery_error(self, Ap):
        return(np.linalg.norm(np.nan_to_num(self.A-Ap))/np.linalg.norm(np.nan_to_num(self.A)))

    #constrcut an approximate matrix based on its factors 
    def construct_matrix(self, decomp):
        m,n = self.A.shape

        t1 = np.dot(decomp[0],decomp[1])
        t2 = np.ones((m,1)) * decomp[2]
        t3 = decomp[3] * np.ones((1,n))

        mat =  t1 + t2 + t3
        return(mat)
