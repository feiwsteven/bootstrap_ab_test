#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gp
from gurobipy import GRB
import math
from typing import List
import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.formula.api import ols
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import ticker

from loguru import logger

import warnings



# # Function Definition

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
            dist_cat = st.rv_discrete(values=(np.arange(len(parms[0])), parms[0]))
            if len(parms) == 1:
                # univariate
                samples = dist_cat.rvs(size=(size, 1))
            else:
                # multivariate
                # parms[1]: int
                samples = dist_cat.rvs(size=(size, parms[1]))
        elif dist_name == 'uniform':
            # parms[0] and parms[1]: scalar or array_like float
            samples = st.uniform.rvs(*parms, size=(size, dim))
        elif dist_name == 'normal':
            # parms[0] and parms[1]: scalar or array_like float
            samples = st.norm.rvs(*parms, size=(size, dim))
        elif dist_name == 'multi-normal':
            # parms[0]: array_like float
            # parms[1]: matrix_like float
            samples = st.multivariate_normal(*parms, allow_singular=True).rvs(size=size)
        elif dist_name == 't':
            # parms[0]: scalar or array_like float > 0
            # parms[1] and parms[2]: scalar or array_like float
            samples = st.t.rvs(*parms, size=(size, dim))
        elif dist_name == 'multi-t':
            # parms[0]: array_like float
            # parms[1]: matrix_like float
            # parms[2]: float > 0
            samples = st.multivariate_t(*parms, allow_singular=True).rvs(size=size)

        # return: n-by-p np.ndarray
        return np.squeeze(samples)


# In[3]:


class ClinicalDataGenerator(RVGenerator):
    '''
    model for the ith sample in the jth treatment arm: 
        Y_ij = alpha_j + beta.T * X_i + epsilon_j, j = 1,...,q-1
        E(Y_ij|X_i) = alpha_j + beta.T * X_i, 
        Var(Y_ij|X_i) = sigma_j^2
    '''

    def __init__(self, **dists):
        '''      
        dists: 
        # either 'alphaX' or ('alpha', 'X') could be missed
            {'alphaX': dists_alphaX, 'alpha': dists_alpha, 'X': dist_X,
                'beta': dists_beta, 'epsilon': dists_epsilon}
                
        dists_alphaX: 
        # correlated part of treatment effects and covariates
        # R should implies independencies between all alphas
            ('G-copula', R: matrix_like float, 
                {'alphaX_alpha': dists_alphaX_alpha, 'alphaX_X': dists_alphaX_X})
        
        dists_alphaX_alpha, dists_alpha: 
        # composed of independent univariate distributions only
        # dimensions of dists_alphaX_alpha and dists_alpha must add up to q-1
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
        
        dists_epsilon: 
        # dimensions must add up to q
            ((dist1_name: str, *dist1_parms), (dist2_name: str, *dist2_parms), ...)
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
        Y_ = alpha_ + X_.dot(beta_) + epsilon_
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
        return {'Y': Y_, 'alpha': alpha_, 'X': X_, 'beta': beta_, 'eps':epsilon_}
    
    
    def control(self, n_samples) -> tuple:
        dists = self._dists.copy()
        dists['alpha'] = ('constant',0)
        
        if 'alphaX' in self._dists:
            dists['alphaX'] = None
            dist_alphaX_X = (self._dists['alphaX'][0], 
                             np.array(self._dists['alphaX'][1])[self._q1:, self._q1:], 
                             *self._dists['alphaX'][2]['alphaX_X'])
            dists['X'] = (dist_alphaX_X, *self._dists['X'])                 if 'X' in self._dists else (dist_alphaX_X,)
            
        dists['epsilon'] = self._dists['epsilon'][0]
        
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
                dists1['epsilon'] = self._dists['epsilon'][1+i]
                
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

                dists2['epsilon'] = self._dists['epsilon'][1+self._q1+i]
                
                samples[self._q1+i] = self._arm(n_samples[self._q1+i], **dists2)
        
        # return: ((Y1, alpha1, X1, beta1), (Y2, alpha2, X2, beta2), ...)
        return tuple(samples)


# In[4]:


def t_test(Yc, Yt) -> np.float64:
    pvalue_t = st.ttest_ind(Yc, Yt, equal_var=False)[1]
    return pvalue_t

def F_test(X, Y, n_samples:list) -> np.float64:
    n_groups = len(n_samples)
    group_idx = np.repeat(np.arange(n_groups), n_samples)
    data_mat = np.column_stack([Y, group_idx, X])
    X_str = ['X'+str(i) for i in np.arange(1, 1+X.shape[1])]
    data_df = pd.DataFrame(data_mat, columns=['Y', 'idx'] + X_str)

    formula = 'Y ~ 1 + C(group_idx)'
    for Xi in X_str:
        formula += ' + ' + Xi
    model = ols(formula, data=data_df).fit()
    pvalue_F = sm.stats.anova_lm(model, typ=2)['PR(>F)'][0]

    return pvalue_F

def CUPED(Xc, Yc, Xt, Yt) -> np.float64:
    if len(Xc.shape) == 1:
        Xc = Xc.reshape([-1, 1])
        Xt = Xt.reshape([-1, 1])
    X = np.row_stack([Xc, Xt])
    Y = np.concatenate([Yc, Yt])

    X_ = sm.add_constant(X)
    theta = sm.OLS(Y, X_).fit().params
    Y_cv = Y - X_.dot(theta)

    pvalue_CUPED = st.ttest_ind(Y_cv[:Yc.size], Y_cv[Yc.size:], equal_var=False)[1]
    return pvalue_CUPED

def GMM(X: np.ndarray, Y: np.ndarray,
        n_samples: list, max_iter: int = 4) -> (np.float64, np.float64):
    n_groups = len(n_samples)
    sample_size = sum(n_samples)
    group_idx = np.repeat(np.arange(n_groups), n_samples)
    group_code = np.zeros([sample_size, n_groups])
    group_code[np.arange(sample_size), group_idx] = 1

    X_ = np.concatenate([np.ones([sample_size, 1]), group_code[:, 1:], X], axis=1)
    X__h0 = np.concatenate([np.ones([sample_size, 1]), X], axis=1)

    inv_variance = np.ones(sample_size)
    parameter_h0 = [0] + list(range(1+n_groups-1, 1+n_groups-1+X.shape[1]))

    i = 0
    while i < max_iter:
        A = X_.T.dot(np.multiply(inv_variance.reshape([-1, 1]), X_))
        b = X_.T.dot(inv_variance * Y)
        try:
            theta = np.linalg.solve(A,b)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(A,b, rcond=None)[0]

        residual = Y - X_.dot(theta)
        for j in range(n_groups):
            inv_variance[group_idx == j] =                 1 / residual[group_idx == j].var()
        i += 1
    
    X__weighted = X_ * inv_variance.reshape([-1, 1])
#     g = X__weighted.T.dot(Y - X_.dot(theta)) / sample_size
#     G = - X__weighted.T.dot(X_) / sample_size
# 
#     v_mat = X__weighted.T.dot(X_) / sample_size
#     try:
#         w_mat = np.linalg.inv(v_mat)
#     except np.linalg.LinAlgError:
#         w_mat = np.linalg.pinv(v_mat)
#     gmm = g.dot(w_mat).dot(g) * sample_size
    
    # under the H_0
    theta_h0 = np.zeros(X_.shape[1])
    A_h0 = X__h0.T.dot(np.multiply(inv_variance.reshape([-1, 1]), X__h0))
    b_h0 = X__h0.T.dot(inv_variance * Y)
    try:
        theta_h0[parameter_h0] = np.linalg.solve(A_h0,b_h0)
    except np.linalg.LinAlgError:
        theta_h0[parameter_h0] = np.linalg.lstsq(A_h0,b_h0, rcond=None)[0]
    
    g_h0 = X__weighted.T.dot(Y - X_.dot(theta_h0)) / sample_size
    G_h0 = - X__weighted.T.dot(X_) / sample_size

    residual_h0 = (Y - X_.dot(theta_h0)).reshape([-1, 1])
    omega_h0 = (X__weighted * residual_h0).T.dot(
        X__weighted * residual_h0) / sample_size
    try:
        omega_h0_inv = np.linalg.inv(omega_h0)
    except np.linalg.LinAlgError:
        omega_h0_inv = np.linalg.pinv(omega_h0)
    
    sigma_h0 = G_h0.dot(omega_h0_inv).dot(G_h0.T)
    try:
        sigma_h0_inv = np.linalg.inv(sigma_h0)
    except np.linalg.LinAlgError:
        sigma_h0_inv = np.linalg.pinv(sigma_h0)
    
    d_gmm_h0 = G_h0.dot(omega_h0_inv).dot(g_h0)
    score = d_gmm_h0.T.dot(sigma_h0_inv).dot(d_gmm_h0) * sample_size
    pvalue_gmm_score = 1 - st.chi2.cdf(score, df=n_groups-1)
    
#     gmm_h0 = sample_size * g_h0.dot(omega_h0_inv).dot(g_h0)
#     likelihood_ratio = gmm_h0 - gmm
#     pvalue_gmm_lr = 1 - st.chi2.cdf(likelihood_ratio, df=n_groups-1)

    return pvalue_gmm_score, 0


# In[7]:


# l0 ridge balance
##############
# arguments: #
##############
#   XAA : covariates in AA test, nAA by p numpy ndarray 
#   YAA : outcome variable in AA test, nAA by 1 numpy ndarray  
#   WAA : inverse of covariance matrix of YAA conditional on XAA in AA test, nAA by nAA numpy ndarray
#   Tau : treatment variable in AB test, n by q numpy ndarray 
#   X : covariates in AB test, n by p numpy ndarray 
#   nAA : sample size in AA test 
#   n : sample size in AB test 
#   p : number of covairates
#   q : number of treatment arms
#   m : upper bound of number of covariates associated with non-zero coefficient
#   lam : ridge penalty's tuning parameter
#   timeLimit : time limit in seconds, a termination criterion for Gurobi
#   mipGap : a number belongs to (0,1), a termination criterion for Gurobi. The smaller the mipGap, the  closer the solution to the global optimal point

#########################################
# return value: a dictionary containing #
# following items                       #
#########################################
# betaHat  : beta of the solution
# etaHat   : eta of the solution
# runTime  : run time of gurobi in seconds
# objVal   : solution's objective value
# objBound : lower bound of the global optimal objective value
# MIPGAP   : solution's MIP gap

def GurobiAATest_RidgeBalance(    XAA,    YAA,    WAA,    Tau,    X,    nAA,    n,    p,    q,    m,    lam,    timeLimit,    mipGap=0.0001   ):
    if not isinstance(XAA,np.ndarray):
        raise TypeError
    
    if not isinstance(YAA,np.ndarray):
        raise TypeError
    
    if not isinstance(WAA,np.ndarray):
        raise TypeError
    
    if not isinstance(Tau,np.ndarray):
        raise TypeError
    
    if not isinstance(X,np.ndarray):
        raise TypeError
    
    if not isinstance(nAA,int):
        raise TypeError
    
    if not isinstance(n,int):
        raise TypeError
    
    if not isinstance(p,int):
        raise TypeError
    
    if not isinstance(q,int):
        raise TypeError
    
    if not isinstance(m,int):
        raise TypeError
    
    if not isinstance(lam,float):
        raise TypeError
    
    if not X.shape[0]==n:
        raise Exception("X's row dimension should be n!")
    
    if not XAA.shape[0]==nAA:
        raise Exception("XAA's row dimension should be nAA!")
    
    if not XAA.shape[1]==p:
        raise Exception("XAA's column dimension should be p!")
    
    if not YAA.shape[0]==nAA:
        raise Exception("YAA's column dimension should be nAA!")
    
    if not YAA.shape[1]==1:
        raise Exception("YAA's column dimension should be 1!")
    
    if not WAA.shape==(nAA,nAA):
        raise Exception("WAA's row and column dimensions should be nAA!")
    
    if not Tau.shape[0]==n:
        raise Exception("Tau's row dimensions should be n!")
    
    if not Tau.shape[1]==q:
        raise Exception("Tau's column dimensions should be q!")
    
    if not m<=p:
        raise Exception("m should be less than or equal to p!")
    
    if lam<0:
        raise Exception("lam should be non-negative!")
    
    if not timeLimit>0.0:
        raise Exception("timeLimit should be positive!")
    
    if not mipGap>0.0 and mipGap<1.0:
        raise Exception("mipGap should be positive and less than 1!")
    
    # create model and disable all outputs
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model("ABTestVarianceReduction", env=env) as model:  
            # create beta variables, the associated coefficients of covariates X
            beta = model.addVars(                p,                lb=-GRB.INFINITY,                ub=GRB.INFINITY,                obj=0.0,                vtype=GRB.CONTINUOUS,                name="beta"            )

            # create eta variables, the associated binary variable of each corresponding
            # beta is non-zero
            oneMinusEta = model.addVars(                p,                lb=0,                ub=1,                obj=0.0,                vtype=GRB.BINARY,                name="oneMinusEta"            )

            # create alpha variable, the associated coefficients of treatment effect \Tau
            alpha = model.addVars(                q,                lb=-GRB.INFINITY,                ub=GRB.INFINITY,                obj=0.0,                vtype=GRB.CONTINUOUS,                name="alpha"            )

            # create rho variables, which are (Tau^T * Tau)^{-1}*(Tau^T * X * beta) 
            rho = model.addVars(                q,                lb=-GRB.INFINITY,                ub=GRB.INFINITY,                obj=0.0,                vtype=GRB.CONTINUOUS,                name="rho"            )

            # create mu variables, which are rho's ordered ascendingly
            mu = model.addVars(                q,                lb=-GRB.INFINITY,                ub=GRB.INFINITY,                obj=0.0,                vtype=GRB.CONTINUOUS,                name="mu"            )

            # create rhoMinusMu variables, which are rho[i]-mu[j]
            rhoMinusMu = model.addVars(               q*q,               lb=-GRB.INFINITY,               ub=GRB.INFINITY,               obj=0.0,               vtype=GRB.CONTINUOUS,               name="rhoMinusMu"            )

            # create zeta variables, which are binary variables to attain mu   
            zeta = model.addVars(                q*q,                lb=0.0,                ub=1.0,                obj=0.0,                vtype=GRB.BINARY,                name="zeta"            )

            # set objective function
            XAATWAAXAA = np.transpose(XAA) @ WAA @ XAA
            YAATWAAXAA = -2.0 * (np.transpose(YAA) @ WAA @ XAA)
            objective = gp.QuadExpr( gp.quicksum( [ XAATWAAXAA[i-1][j-1] * beta[i-1] * beta[j-1] for i in range(1,p+1) for j in range(1,p+1) ] ) )
            objective.add( gp.quicksum( [ YAATWAAXAA[0][i-1] * beta[i-1] for i in range(1,p+1) ] ) ) 
            objective.addConstant( (np.transpose(YAA) @ WAA @ YAA)[0][0] )
            for i in range(1,q):
                objective.addTerms(                        [ lam,   -2.0*lam, lam     ],                        [ mu[i], mu[i],    mu[i-1] ],                        [ mu[i], mu[i-1],  mu[i-1] ]                )

            model.setObjective(objective, GRB.MINIMIZE)
            model.update()
            del XAATWAAXAA 
            del YAATWAAXAA 
            del objective

            # add orthogonality constraints 
            # TauTXOvern = np.transpose(Tau) @ X
            # TauTXOvern = np.diag( 1/np.sum(Tau , axis=0) ) @ TauTXOvern
            # for i in range(1,q):
            #     model.addConstr(\
            #             gp.quicksum( [ TauTXOvern[i-1][j-1] * beta[j-1] for j in range(1,p+1) ] ) == \
            #             gp.quicksum( [ TauTXOvern[i][j-1] * beta[j-1] for j in range(1,p+1) ] ),\
            #             "orthogonal_constraint_" + str(i)\
            #     )
            # 
            #model.update()
            #del TauTXOvern

            # add constraints to make rho equal to (Tau^T * Tau)^{-1}*(Tau^T * X * beta) 
            TauTXOvern = np.transpose(Tau) @ X
            TauTXOvern = np.diag( 1/np.sum(Tau , axis=0) ) @ TauTXOvern
            for i in range(1,q+1):
                model.addConstr(                        gp.quicksum( [ TauTXOvern[i-1][j-1] * beta[j-1] for j in range(1,p+1) ] ) ==                         rho[i-1],                        "rho_" + str(i)                )

            model.update()
            del TauTXOvern

            #add constraints to let rhoMinusMu equal to rho[i]-mu[j]
            for i in range(1,q+1):
                for j in range(1,q+1):
                    model.addConstr(                            rho[i-1] - mu[j-1] ==                             rhoMinusMu[(i-1)*q+(j-1)],                            "rhoMinusMu_" + str(i) + "_" + str(j)                    )

            model.update()

            # add SOS-1 related to zeta and rhoMinusMu
            for i in range(1,q+1):
                for j in range(1,q+1):
                    model.addSOS(                            GRB.SOS_TYPE1,                            [ rhoMinusMu[(i-1)*q+(j-1)] , zeta[(i-1)*q+(j-1)] ]
                    )

            model.update()

            # add constraints related to zeta 
            for i in range(1,q+1):
                model.addConstr(                    gp.quicksum( [ zeta[(i-1)*q+(j-1)] for j in range(1,q+1) ] ) == 1.0,                    "zeta_row_" + str(i)                )

            for i in range(1,q+1):
                model.addConstr(                   gp.quicksum( [ zeta[(j-1)*q+(i-1)] for j in range(1,q+1) ] ) == 1.0,                   "zeta_col_" + str(i)                )

            model.update()

            # add SOS-1 related to eta
            for i in range(1,p+1):
                model.addSOS(GRB.SOS_TYPE1,[ oneMinusEta[i-1] , beta[i-1] ])

            model.update()

            # add sparsity constraints
            model.addConstr(                    oneMinusEta.sum() >=                     p-m,                    "sparsity_constraint"            )
            model.update()

            # set time limit
            model.setParam("TimeLimit", timeLimit)
            model.setParam("MIPGap", mipGap)

            # solve
            model.optimize()

            # extract solution
            betaHat = [ beta[i-1].X for i in range(1,p+1) ]
            etaHat = [1-oneMinusEta[i-1].X for i in range(1,p+1) ]
            rhoHat = [rho[i-1].X for i in range(1,q+1)]
            muHat = [mu[i-1].X for i in range(1,q+1)]
            zetaHat = [zeta[i-1].X for i in range(1,q*q+1)]
            runTime = model.runtime
            objVal = model.ObjVal
            objBound = model.ObjBound
            MIPGAP = model.MIPGap
    
    return {        "betaHat"  : betaHat,        "etaHat"   : etaHat,        "rhoHat"   : rhoHat,        "muHat"    : muHat,        "zetaHat"  : zetaHat,        "runTime"  : runTime,        "objVal"   : objVal,        "objBound" : objBound,        "MIPGAP"   : MIPGAP    }


# In[61]:


# TAB_CUPED l0 ridge balance
def TAB_CUPED(XAA: np.ndarray, YAA: np.ndarray,
              X: np.ndarray, Y: np.ndarray,
              n_samples: list, max_m: int,
              lam: float = 0.0, timeLimit: float = 100,
              mipGap: float = 0.0001, max_iter: int = 3) -> np.float64:
    n_groups = len(n_samples)
    nAA = YAA.size
    n = sum(n_samples)
    group_idx = np.repeat(np.arange(n_groups), n_samples)
    group_code = np.zeros([n, n_groups])
    group_code[np.arange(n), group_idx] = 1

    XAA_ = np.concatenate([np.ones([nAA, 1]), group_code[:, 1:], XAA], axis=1)
    X_ = np.concatenate([np.ones([n, 1]), group_code[:, 1:], X], axis=1)
    
    XAA__h0 = np.concatenate([np.ones([nAA, 1]), XAA], axis=1)
    X__h0 = np.concatenate([np.ones([n, 1]), X], axis=1)
    
    inv_varAA = np.ones(nAA)
    inv_varAA_l0 = np.ones(nAA)
#     inv_var_l0 = np.ones(n)

    #------------------------------------------
    # estimate inv_varAA, and use it to estimate etaHat
    AAA = XAA_.T.dot(np.multiply(inv_varAA.reshape([-1, 1]), XAA_))
    bAA = XAA_.T.dot(inv_varAA * YAA)
    try:
        thetaAA = np.linalg.solve(AAA,bAA)
    except np.linalg.LinAlgError:
        thetaAA = np.linalg.lstsq(AAA,bAA, rcond=None)[0]

    residualAA = YAA - XAA_.dot(thetaAA)
    for j in range(n_groups):
        inv_varAA[group_idx == j] =             1 / residualAA[group_idx == j].var()
    
    #------------------------------------------------------
    # find m according to BIC
    if max_m > XAA__h0.shape[1]:
        max_m = XAA__h0.shape[1]
    
    BIC_list = np.empty(max_m)
    inv_varAA_l0_list = np.empty(max_m)
    for m in range(1,max_m+1):
#         inv_varAA = np.ones(nAA)
        res_l0 = GurobiAATest_RidgeBalance(
            XAA__h0, YAA.reshape([-1,1]),
            np.diag(inv_varAA), group_code, 
            X__h0, int(nAA), int(n), X__h0.shape[1],
            n_groups, int(m), float(lam), timeLimit, mipGap
        )
        eta_l0_h0 = np.array(res_l0['etaHat'], dtype=bool)
        eta_l0 = np.concatenate([eta_l0_h0[:1], np.ones(n_groups-1, dtype=bool), eta_l0_h0[1:]])
        XAA__l0 = XAA_[:,eta_l0]

        # estimate inv_varAA_l0
        i = 1
        while i <= max_iter:
            AAA_l0 = XAA__l0.T.dot(np.multiply(inv_varAA_l0.reshape([-1, 1]), XAA__l0))
            bAA_l0 = XAA__l0.T.dot(inv_varAA_l0 * YAA)
            try:
                thetaAA_l0 = np.linalg.solve(AAA_l0,bAA_l0)
            except np.linalg.LinAlgError:
                thetaAA_l0 = np.linalg.lstsq(AAA_l0,bAA_l0, rcond=None)[0]

            residualAA_l0 = YAA - XAA__l0.dot(thetaAA_l0)
            for j in range(n_groups):
                inv_varAA_l0[group_idx == j] =                     1 / residualAA_l0[group_idx == j].var()
            i += 1
        inv_varAA_l0_list = inv_varAA_l0

        # calculate BIC
        RSS_l0 = (residualAA_l0**2).sum()
        BIC_list[m-1] = nAA*np.log(RSS_l0/nAA) + (m+1)*np.log(nAA)
        
    m = BIC_list.argmin() + 1
    inv_varAA_l0 = inv_varAA_l0_list[m-1]
    
    #-----------------------------------------
    # estimate theta_h0
#     inv_var_l0 = np.ones(n)
    inv_var_l0 = inv_varAA_l0
    X__l0 = X_[:,eta_l0]
    X__l0_h0 = X__h0[:,eta_l0_h0]
    A_l0_h0 = X__l0_h0.T.dot(np.multiply(inv_var_l0.reshape([-1, 1]), X__l0_h0))
    b_l0_h0 = X__l0_h0.T.dot(inv_var_l0 * Y)
    try:
        beta_l0_h0 = np.linalg.solve(A_l0_h0,b_l0_h0)
    except np.linalg.LinAlgError:
        beta_l0_h0 = np.linalg.lstsq(A_l0_h0,b_l0_h0, rcond=None)[0]
    
#     beta = np.array(res_l0['betaHat'])
#     print(\
#         X__h0[:n_samples[0],:].dot(beta).sum()/n_samples[0],\
#         X__h0[n_samples[0]:(n_samples[0]+n_samples[1]),:].dot(beta).sum()/n_samples[1],\
#         X__h0[(n_samples[0]+n_samples[1]):,:].dot(beta).sum()/n_samples[2],\
#     )
    
#     beta_l0_h0 = np.array(res_l0['betaHat'])[eta_l0_h0]
    theta_h0 = np.zeros(X__l0.shape[1])
    if eta_l0_h0[0] == False:
        theta_h0[n_groups-1:] = beta_l0_h0
    else:
        parameter_l0_h0 = np.array([0] + list(range(1+n_groups-1, X__l0.shape[1])))
        theta_h0[parameter_l0_h0] = beta_l0_h0
    
    X__weighted = X__l0 * inv_var_l0.reshape([-1, 1])
    g_h0 = X__weighted.T.dot(Y - X__l0.dot(theta_h0)) / n
    G_h0 = - X__weighted.T.dot(X__l0) / n
    
    residual_h0 = (Y - X__l0.dot(theta_h0)).reshape([-1, 1])
    omega_h0 = (X__weighted * residual_h0).T.dot(
        X__weighted * residual_h0) / n
    try:
        omega_h0_inv = np.linalg.inv(omega_h0)
    except np.linalg.LinAlgError:
        omega_h0_inv = np.linalg.pinv(omega_h0)
    
    sigma_h0 = G_h0.dot(omega_h0_inv).dot(G_h0.T)
    try:
        sigma_h0_inv = np.linalg.inv(sigma_h0)
    except np.linalg.LinAlgError:
        sigma_h0_inv = np.linalg.pinv(sigma_h0)
    
    d_gmm_h0 = G_h0.dot(omega_h0_inv).dot(g_h0)
    score = d_gmm_h0.T.dot(sigma_h0_inv).dot(d_gmm_h0) * n
    pvalue_gmm_l0 = 1 - st.chi2.cdf(score, df=n_groups-1)
    
    true_eta = np.zeros(X__h0.shape[1], dtype=bool)
    true_eta[1:1+n_true] = True
    flag = (sum(eta_l0_h0 * true_eta) == n_true)
    
    return pvalue_gmm_l0, flag, m


# In[62]:


# parallel
from multiprocessing import Pool
from functools import partial
from tqdm.auto import trange, tqdm
from tqdm.contrib.concurrent import process_map
# from concurrent.futures import ProcessPoolExecutor
import mkl

mkl.set_dynamic(0)
mkl.set_num_threads(2)

def parallel(fun, iterable, pool, *params) -> list:
    if __name__ == '__main__':
        fun_p = partial(fun, *params)
        
#         result = process_map(fun_p, iterable,
#                              max_workers=n_core, leave=False,
#                              chunksize = len(iterable)//1000 + 1)

        result = tqdm(pool.imap_unordered(fun_p, iterable,
                                          chunksize = len(iterable)//1000 + 1),
                      total=len(iterable), leave=False)

#     return np.array(result)
    return np.array(list(result))


# In[84]:


def test_pvalue(cdg: ClinicalDataGenerator, 
                nAA_samples_c: int, nAA_samples_t: int or List[int],
                n_samples_c: int, n_samples_t: int or List[int], 
                tests_str=['t','F','CUPED','GMM score','TAB_CUPED'], 
                test_arm=0, max_m=None, lam=0.0, dum=None) -> np.ndarray:   
    # take care of only one treatment arm
    n_samples_t = np.reshape(n_samples_t, -1)
    
    control_arm = cdg.control(n_samples_c)
    XAAc = control_arm['X']
    YAAc = control_arm['Y']
#     Xc = XAAc
    Yc = YAAc
    
    treatment_arms = cdg.treatment(*n_samples_t)
    XAAt = [arm['X'] for arm in treatment_arms]
    YAAt = [arm['Y']-arm['alpha'] for arm in treatment_arms]
#     Xt = XAAt
    Yt = [arm['Y'] for arm in treatment_arms]
    
    XAA = np.row_stack([XAAc, *XAAt])
    YAA = np.concatenate([YAAc, *YAAt])
    X = XAA
    Y = np.concatenate([Yc, *Yt])

    if 'GMM' in tests_str:
        pvalues = np.zeros(len(tests_str)+1)
    else:
        pvalues = np.zeros(len(tests_str))
    
    for i,test in enumerate(tests_str):
        if 't' == test:
            pvalues[i] = t_test(Yc, Yt[test_arm])

        if 'F' == test:
            pvalues[i] = F_test(X, Y, [n_samples_c, *n_samples_t])

        if 'CUPED' == test:
            pvalues[i] = CUPED(YAAc, Yc, YAAt[test_arm], Yt[test_arm])

        if 'GMM' == test or             'GMM score'  == test or             'GMM LR'  == test:
            
            p_score, p_lr = GMM(X, Y, [n_samples_c, *n_samples_t])
            if 'GMM' == test:
                pvalues[i:i+2] = [p_score, p_lr]
                i += 1
            elif 'GMM score' == test:
                pvalues[i] = p_score
            elif 'GMM LR' == test:
                pvalues[i] = p_lr
        
        if 'TAB_CUPED' == test:
#             pvalues[i], flag, m = TAB_CUPED(XAA, YAA, X, Y, [n_samples_c, *n_samples_t], max_m, lam)
            pvalues[i], flag, m = TAB_CUPED(np.column_stack([YAA,XAA]),
                                            YAA, np.column_stack([Y,X]), 
                                            Y, [n_samples_c, *n_samples_t], max_m, lam)

    return pvalues, flag, m


# In[64]:


def power(cdg: ClinicalDataGenerator,
          nAA_samples_c: int, nAA_samples_t: int or List[int], 
          n_samples_c: int, n_samples_t: int or List[int], 
          tests_str=['t','F','CUPED','GMM score','TAB_CUPED'],
          test_arm=0, max_m=None, lam=0.0,
          level=0.05, n_reps=10000, pool=[]) -> np.ndarray:
        
    par_res = parallel(
        test_pvalue, range(n_reps), pool,
        cdg, nAA_samples_c, nAA_samples_t,
        n_samples_c, n_samples_t, tests_str, test_arm,
        max_m, lam
    )
    
#     n_tests = len(tests_str)
#     if 'GMM' in tests_str:
#         n_tests += 1
#     pvalues = np.zeros([n_reps, n_tests])
    
#     for i in trange(n_reps, leave=False):
#         pvalues[i,:] = _test_pvalue(cdg, n_samples_c, n_samples_t, tests_str, test_arm, cov_used)

    results = np.sum(np.array(np.vstack(par_res[:,0])) < level, axis=0) / n_reps,         np.array(par_res[:,1], dtype=int).sum() / n_reps,         np.array(par_res[:,2])

    return results


# # Power Simulations

# ## High Dimension with noise

# In[90]:


points = 1
n_samples_c = 3000
n_samples_t = [3000] * 2
n_true = 5
tests_str = ['t','F','CUPED','GMM score','TAB_CUPED']
level = 0.05
n_reps = 3000
n_core = 6

n_noise_list = np.linspace(10, 10, points, dtype=int)

if 'GMM' in tests_str:
    i_ex = tests_str.index('GMM')
    tests_str_exgmm = tests_str[:i_ex] + ['GMM score', 'GMM LR'] + tests_str[i_ex+1:]
else:
    tests_str_exgmm = tests_str

pows_highdim_noise = np.empty([points, len(tests_str_exgmm)])
cover_rates = np.empty(points)
mi_list = np.empty([n_reps, points])
for i,n_noise in enumerate(tqdm(n_noise_list)):
    cdg = ClinicalDataGenerator(
        alpha = (('constant',0.05), ('constant',0)),
        X = (('normal', np.zeros(n_true+n_noise), 1), ),
        beta = (('constant', [1]*n_true), ('constant', [0]*n_noise), ), 
        epsilon = (('normal',0,1), )*3
    )
    if __name__ == '__main__':
        with Pool(processes=n_core) as pool:
            pows_highdim_noise[i,:], cover_rates[i], mi_list[:,i] = power(
                cdg, n_samples_c, n_samples_t, n_samples_c, n_samples_t,
                tests_str=tests_str, max_m=1+n_true+n_noise//2, lam=0.0,
                level=level, n_reps=n_reps, pool=pool
            )
            pool.close()
            pool.join()


# In[94]:


p_idx = points+1
# p_idx = 8
fig, axes = plt.subplots(2,1, figsize=(15,10))
for i,test in enumerate(tests_str_exgmm):
    axes[0].plot(n_noise_list[:p_idx], pows_highdim_noise[:p_idx,i], label=test)
axes[0].set_xlabel('Number of Noise Covariates')
axes[0].set_ylabel('Power')
axes[0].legend()

rate_line, = axes[1].plot(n_noise_list[:p_idx], cover_rates[:p_idx], color='r', label='Signal Cover Rate')
axes[1].set_ylabel('Signal Cover Rate')
axes[1].set_xlabel('Number of Noise Covariates')
ax2 = axes[1].twinx()
m_box = ax2.boxplot(mi_list[:,:p_idx], positions=n_noise_list[:p_idx], patch_artist=True)
ax2.set_ylabel('Chosen m')
axes[1].legend([rate_line, m_box['boxes'][0]], ['Signal Cover Rate', 'Chosen m'])
plt.show()


# # Type I Error Rate Simulations

# ## High Dimension with Noise


# In[80]:


points = 8
n_samples_c = 3000
n_samples_t = [3000] * 2
n_true = 5
tests_str = ['t','F','CUPED','GMM score','TAB_CUPED']
level = 0.05
n_reps = 5000
n_core = 6

n_noise_list = np.linspace(0, 30, points, dtype=int)

if 'GMM' in tests_str:
    i_ex = tests_str.index('GMM')
    tests_str_exgmm = tests_str[:i_ex] + ['GMM score', 'GMM LR'] + tests_str[i_ex+1:]
else:
    tests_str_exgmm = tests_str

errs_highdim_noise = np.empty([points, len(tests_str_exgmm)])
cover_rates = np.empty(points)
mi_list = np.empty([n_reps, points])
for i,n_noise in enumerate(tqdm(n_noise_list)):
    cdg = ClinicalDataGenerator(
        alpha = (('constant',0), ('constant',0)),
        X = (('normal', np.zeros(n_true+n_noise), 1), ),
        beta = (('constant', [1]*n_true), ('constant', [0]*n_noise), ), 
        epsilon = (('normal',0,1), )*3
    )
    if __name__ == '__main__':
        with Pool(processes=n_core) as pool:
            errs_highdim_noise[i,:], cover_rates[i], mi_list[:,i] = power(
                cdg, n_samples_c, n_samples_t, n_samples_c, n_samples_t,
                tests_str=tests_str, max_m=1+n_true+n_noise//2, lam=0.0,
                level=level, n_reps=n_reps, pool=pool
            )
            pool.close()
            pool.join()


# In[81]:


p_idx = points+1
p_idx = 7
fig, axes = plt.subplots(2,1, figsize=(15,10))
for i,test in enumerate(tests_str_exgmm):
    axes[0].plot(n_noise_list[:p_idx], errs_highdim_noise[:p_idx,i], label=test)
axes[0].axhline(level, color='b', linestyle='--')
axes[0].set_xlabel('Number of Noise Covariates')
axes[0].set_ylabel('Type I Error Rate')
axes[0].legend()

rate_line, = axes[1].plot(n_noise_list[:p_idx], cover_rates[:p_idx], color='r', label='Signal Cover Rate')
axes[1].set_ylabel('Signal Cover Rate')
axes[1].set_xlabel('Number of Noise Covariates')
ax2 = axes[1].twinx()
m_box = ax2.boxplot(mi_list[:,:p_idx], positions=n_noise_list[:p_idx], patch_artist=True)
ax2.set_ylabel('Chosen m')
axes[1].legend([rate_line, m_box['boxes'][0]], ['Signal Cover Rate', 'Chosen m'])
plt.show()
