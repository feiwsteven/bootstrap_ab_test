#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[2]:


class RVGenerator(object):
    def _rnds(self, dist_name: str, *parms, size: int = None) -> np.ndarray:      
        if dist_name != 'G-copula':
            dim = max([
                max(np.array(parm)[None].shape) for parm in parms
            ])
        else:
            # parms[0]: matrix_like float
            # *parms[1:]: tuples
            dim = len(parms[0])
            G = st.norm.cdf(
                st.multivariate_normal.rvs(np.zeros(dim), parms[0], size)
            )
            G = np.atleast_2d(G.T).T # ensure 2D and each column is indexed by sample

            dim1 = 0
            dim2 = 0
            samples = np.zeros([size, dim])
            for dist in parms[1:]:
                dim1 = dim2
                dim2 += max([np.size(parm) for parm in dist[1:]])

                if dist[0] == 'uniform':
                    samples[:, dim1:dim2] = st.uniform.ppf(G[:, dim1:dim2], *dist[1:])
                elif dist[0] == 'normal':
                    samples[:, dim1:dim2] = st.norm.ppf(G[:, dim1:dim2], *dist[1:])
                elif dist[0] == 't':
                    samples[:, dim1:dim2] = st.t.ppf(G[:, dim1:dim2], *dist[1:])
                    
        if dist_name == 'constant':
            # parms[0]: scalar or array_like float
            samples = parms[0] * np.ones([size, 1])
        elif dist_name == 'bernoulli':
            # parms[0] and parms[1]: scalar or array_like float
            samples = st.bernoulli.rvs(*parms, size=(size, dim))
        elif dist_name == 'categorical':
            # parms[0]: scalar or array_like float
            dist_cat = scipy.stats.rv_discrete(values=(range(len(parms[0])), parms[0]))
            samples = dist_cat.rvs(size=(size, dim))
        elif dist_name == 'uniform':
            # parms[0] and parms[1]: scalar or array_like float
            samples = st.uniform.rvs(*parms, size=(size, dim))
        elif dist_name == 'normal':
            # parms[0] and parms[1]: scalar or array_like float
            samples = st.norm.rvs(*parms, size=(size, dim))
        elif dist_name == 'multi-normal':
            # parms[0]: array_like float
            # parms[1]: matrix_like float
            samples = st.multivariate_normal.rvs(*parms, size=size)
        elif dist_name == 't':
            # parms[0]: scalar or array_like float > 0
            # parms[1] and parms[2]: scalar or array_like float
            samples = st.t.rvs(*parms, (size, dim))
        elif dist_name == 'multi-t':
            # parms[0]: array_like float
            # parms[1]: matrix_like float
            # parms[2]: float > 0
            samples = st.multivariate_t.rvs(*parms, size=size)

        # return: n-by-p np.ndarray
        return np.squeeze(samples)


# In[3]:


class ClinicalDataGenerator(RVGenerator):
    '''
    model for the ith sample in the jth treatment arm: 
        Y_ij = Y_ij(A_j) + beta.T * (X_i - E(X)) + epsilon, j = 1,...,q
        E(Y_ij|X) = alpha_j + beta.T * (X_i - E(X)), 
        Var(Y_ij|X) = sigma_j^2
    '''

    def __init__(self, **dists):
        '''      
        dists: 
        # either 'alphaX' or ('alpha', 'X') could be missed
            {'alphaX': dists_alphaX, 'alpha': dists_alpha, 'X': dist_X,
                'beta': dists_beta, 'epsilon': dist_epsilon}
                
        dists_alphaX: 
        # correlated part of treatment effects and covariates
        # R should implies independencies between all alpha
            ('G-copula', R: matrix_like float, 
                {'alphaX_alpha': dists_alphaX_alpha, 'alphaX_X': dists_alphaX_X})
        
        dists_alphaX_alpha, dists_alpha: 
        # composed of independent univariate distributions only
        # dimensions of dists_alphaX_alpha and dists_alpha must add up to q
            ((dist1_name: str, *dist1_parms), (dist2_name: str, *dist2_parms), ...)
            
        dists_alphaX_X: 
        # composed of independent univariate distributions only
        # dimensions of dists_alphaX_X and dists_X must add up to p
            ((dist1_name: str, *dist1_parms), (dist2_name: str, *dist2_parms), ...)
        
        dist_X: 
        # dimensions of dists_alphaX_X and dists_X must add up to p
            ((dist1_name: str, *dist1_parms), (dist2_name: str, *dist2_parms), ...)
            
        dists_beta:
        # dimensions must add up to p
            ((dist1_name: str, *dist1_parms), (dist2_name: str, *dist2_parms), ...)
        
        dist_epsilon: 
        # univariate distribution
            (dist_name: str, *dist_parms)
        '''
        self._dists = dists
        
        # max(np.array(parm)[None].shape: take care of 0 length type
        self._q1 = sum([
            max([max(np.array(parm)[None].shape) for parm in dist[1:]])
                for dist in dists['alphaX'][2]['alphaX_alpha']
        ]) if 'alphaX' in dists else 0
        
        self._p1 = sum([
            max([max(np.array(parm)[None].shape) for parm in dist[1:]])
                for dist in dists['alphaX'][2]['alphaX_X']
        ]) if 'alphaX' in dists else 0
        
        self._q2 = sum([
            max([max(np.array(parm)[None].shape) for parm in dist[1:]]) 
                for dist in dists['alpha']
        ]) if 'alpha' in dists else 0
        
        self._p2 = sum([
            max([max(np.array(parm)[None].shape) for parm in dist[1:]]) 
                for dist in dists['X']
        ]) if 'X' in dists else 0
        
        self._n_treatments = self._q1 + self._q2
        self._n_covariates = self._p1 + self._p2
        
    def _outcomes(self, alpha_: np.ndarray, X_: np.ndarray,
                  beta_: np.ndarray, epsilon_: np.ndarray) -> np.ndarray:
        Y_ = alpha_ + np.dot(X_-np.mean(X_, axis=0), beta_) + epsilon_
        return Y_

    
    # when q=1
    def _arm(self, size, 
             alphaX: tuple = None, alpha: tuple = None, X: tuple = None, 
             beta: tuple = None, epsilon: tuple = None) -> np.ndarray: 
        # either alphaX or alpha must be None
        if alphaX is not None:
            alphaX_ = self._rnds(alphaX[0], *alphaX[1:], size=size)
            alpha_ = alphaX_[:,0]
            alphaX_X_ = alphaX_[:,1:]
            
            ind_X_ = np.column_stack([self._rnds(dist_X[0], *dist_X[1:], size=size) 
                                      for dist_X in X]) if X is not None \
                else np.array([[]]*size)
            X_ = np.column_stack((np.atleast_2d(alphaX_X_.T).T, np.atleast_2d(ind_X_.T).T))
        elif alpha is not None:
            alpha_ = self._rnds(alpha[0], *alpha[1:], size=size)
            X_ = np.column_stack([self._rnds(dist_X[0], *dist_X[1:], size=size)
                                  for dist_X in X])
        
        beta_ = np.hstack([self._rnds(dist_beta[0], *dist_beta[1:], size=1)
                                 for dist_beta in beta])
        epsilon_ = self._rnds(epsilon[0], *epsilon[1:], size=size)
        
        Y_ = self._outcomes(alpha_, X_, beta_, epsilon_)
        return {'Y': Y_, 'alpha': alpha_, 'X': X_, 'beta': beta_}
    
    
    def control(self, n_samples) -> tuple:
        dists = self._dists.copy()
        dists['alpha'] = ('constant', 0)
        
        if 'alphaX' in self._dists:
            dists['alphaX'] = None
            dist_alphaX_X = (self._dists['alphaX'][0], 
                             np.array(self._dists['alphaX'][1])[self._q1:, self._q1:], 
                             *self._dists['alphaX'][2]['alphaX_X'])
            dists['X'] = (dist_alphaX_X, *self._dists['X'])                 if 'X' in self._dists else (dist_alphaX_X,)

        # return: (Y_c, alpha_c, X_c, beta_c)
        return self._arm(n_samples, **dists)

    
    def treatment(self, *n_samples) -> tuple:
        samples = [[]] * self._n_treatments
        
        dists1 = self._dists.copy()
        if 'alphaX' in self._dists:
            for i in range(self._q1):
                idx = [i] + list(range(self._q1, self._q1+self._p1))
                del_idx = np.delete(range(self._p1+self._q1), idx)
                R1 = np.delete(np.delete(self._dists['alphaX'][1], del_idx, 0), del_idx, 1)
                dists1['alphaX'] = (self._dists['alphaX'][0], R1, 
                                    self._dists['alphaX'][2]['alphaX_alpha'][i],
                                    *self._dists['alphaX'][2]['alphaX_X'])
                dists1['alpha'] = None
                    
                samples[i] = self._arm(n_samples[i], **dists1)
        
        # alpha should follows univariate distribution and input one by one
        dists2 = self._dists.copy()
        if 'alpha' in self._dists:
            for i in range(self._q2):
                dists2['alphaX'] = None
                dists2['alpha'] = self._dists['alpha'][i]
                
                if 'alphaX' in self._dists:
                    dist_alphaX_X = (self._dists['alphaX'][0], 
                                     np.array(self._dists['alphaX'][1])[self._q1:, self._q1:], 
                                     *self._dists['alphaX'][2]['alphaX_X'])
                    dists2['X'] = (dist_alphaX_X, *self._dists['X'])                         if 'X' in self._dists else (dist_alphaX_X,)
                    
                samples[self._q1+i] = self._arm(n_samples[self._q1+i], **dists2)
        
        # return: ((Y1, alpha1, X1, beta1), (Y2, alpha2, X2, beta2), ...)
        return tuple(samples)

    
# cdg = ClinicalDataGenerator(
#                       alphaX = ('G-copula', [[1,0,0],[0,1,0],[0,0,1]],
#                                 {'alphaX_alpha': (('normal',0,1),),
#                                  'alphaX_X': (('t',[1,2],0,1),)}),
#                       alpha = (('constant',1),),
#                       X = (('uniform',0,1), ('normal',0,1),),
#                       beta = (('constant',[1,1,1,1]),), 
#                       epsilon = ('normal',0,1)
# )
# cdg.treatment(500,500)




