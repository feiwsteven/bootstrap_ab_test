#!/usr/bin/env python
# coding: utf-8

# In[1]:


############## Package Reqired ###############
import gurobipy as gp
from gurobipy import GRB
import scipy.stats as st
from typing import List
import numpy as np


# In[14]:


############## Example ###############

import math
# set random number generator seed
rng = np.random.default_rng(74809)
# number of covariates X
p = 20
# number of treatment arms
q = 3
# sample size of each arm in A/A test
nEachArmAA = 100
# sample size of A/B test
nEachArm = 500
# sample size of A/B test
n = nEachArm * q
# sample size of A/A test
nAA = nEachArmAA * q
# variance of each covariate X
var = 1
# correlation \rho between each pair of covariates X
rho = 0.3
# sigma square of outcome variable Y conditional on X and \Tau
sigmaSquare = 1
# sparsity level m
m = 5
# time limit in seconds
timeLimit=100
# gurobi termination mipGap
mipGap=0.0001
# n by p covariate matrix X for A/B test
X = np.random.multivariate_normal(
    mean=[0 for i in range(1,p+1)],
    cov=[ [var*(rho+(1-rho)*float(i==j)) for i in range(1,p+1) ] for j in range(1,p+1)],
    size=n
)
# nAA by p covariate matrix XAA for A/A test
XAA = np.random.multivariate_normal(
    mean=[0 for i in range(1,p+1)],
    cov=[ [var*(rho+(1-rho)*float(i==j)) for i in range(1,p+1) ] for j in range(1,p+1)],
    size=nAA
)
# nAA by q treatment indicator \TauAA
TauAA = np.array([[float(i/nEachArmAA<=j and j-1<i/nEachArmAA) for j in range(1,q+1) ] for i in range(1,nAA+1)])
# n by q treatment indicator \Tau
Tau = np.array([[float(i/nEachArm<=j and j-1<i/nEachArm) for j in range(1,q+1) ] for i in range(1,n+1)])
# p by 1 beta0, the true coefficient associated with X to predict Y
beta0 = np.array([[float(i<=m)] for i in range(1,p+1)])
# q by 1 alpha0, the true coefficient associated with \Tau to predict Y
alpha0 = np.array([[i*0.1] for i in range(1,q+1)])
# n by 1 outcome variable Y of A/B test
Y = np.matmul( Tau , alpha0 ) + np.matmul( X , beta0 ) + np.transpose( [np.random.standard_normal(size=n) * math.sqrt(sigmaSquare)] )
# nAA by 1 outcome variable YAA of A/A test
YAA = np.matmul( TauAA , np.zeros([q,1]) ) + np.matmul( XAA , beta0 ) + np.transpose( [np.random.standard_normal(size=nAA) * math.sqrt(sigmaSquare)] )

output = TAB_CUPED(XAA, YAA.squeeze(), X, Y.squeeze(), [nEachArmAA]*q, [nEachArm]*q, p//2)
print(output)


# In[13]:


# TAB_CUPED: l0 no ridge balance

##############
# arguments: #
##############
#   XAA : covariates in AA test, nAA by p numpy ndarray 
#   YAA : outcome variable in AA test, length nAA numpy ndarray  
#   X   : covariates in AB test, n by p numpy ndarray 
#   Y   : outcome variable in AB test, length n numpy ndarray
#   nAA_samples : treatment variable in AA test, length q list, (must add up to nAA)
#   n_samples   : treatment variable in AB test, length q list, (must add up to n)
#   max_m : m will search within range (1,max_m), (must be less than 1+p)
#   timeLimit : time limit in seconds, a termination criterion for Gurobi
#   mipGap : a number belongs to (0,1), a termination criterion for Gurobi.
#            The smaller the mipGap, the  closer the solution to the global optimal point
#   max_iter: maximum interation number to estimate the inverse of covariance matrix of residual
#########################################
# return value: a dictionary containing #
# following items                       #
#########################################
# pvalue_gmm_l0: the pvalue
# eta_l0_h0: the final eta chosen indiating which covariates have been chosen,
#            length 1+X.shape[1] boolean numpy array
#            (intercept included and took the first place)
# m: the final m chosen
def TAB_CUPED(XAA: np.ndarray, YAA: np.ndarray,
              X: np.ndarray, Y: np.ndarray,
              nAA_samples: list, n_samples: list,
              max_m: int,
              timeLimit: float = 100,
              mipGap: float = 0.0001,
              max_iter: int = 3):
    n_groups = len(n_samples)
    
    nAA = sum(nAA_samples)
    groupAA_idx = np.repeat(np.arange(n_groups), nAA_samples)
    groupAA_code = np.zeros([nAA, n_groups])
    groupAA_code[np.arange(nAA), groupAA_idx] = 1
    
    n = sum(n_samples)
    group_idx = np.repeat(np.arange(n_groups), n_samples)
    group_code = np.zeros([n, n_groups])
    group_code[np.arange(n), group_idx] = 1

    XAA_ = np.concatenate([np.ones([nAA, 1]), groupAA_code[:, 1:], XAA], axis=1)
    X_ = np.concatenate([np.ones([n, 1]), group_code[:, 1:], X], axis=1)
    
    XAA__h0 = np.concatenate([np.ones([nAA, 1]), XAA], axis=1)
    X__h0 = np.concatenate([np.ones([n, 1]), X], axis=1)
    
    #------------------------------------------------------
    # find m according to BIC
    if max_m > XAA__h0.shape[1]:
        max_m = XAA__h0.shape[1]
    
    BIC_list = np.empty(max_m)
    eta_l0_h0_list = [[]] * max_m
    residualAA_l0_list = [[]] * max_m
    for m in range(1,max_m+1):
        # estimate inv_varAA
        AAA = XAA_.T.dot(XAA_)
        bAA = XAA_.T.dot(YAA)
        try:
            thetaAA = np.linalg.solve(AAA,bAA)
        except np.linalg.LinAlgError:
            thetaAA = np.linalg.lstsq(AAA,bAA, rcond=None)[0]

        residualAA = YAA - XAA_.dot(thetaAA)
        inv_varAA = np.ones(nAA)inv_varAA = np.ones(nAA)
        for j in range(n_groups):
            inv_varAA[groupAA_idx == j] =                 1 / residualAA[groupAA_idx == j].var()
        
        # use inv_varAA to estimate etaHat
        res_l0 = GurobiAATest_NoBalance(
            XAA__h0, YAA.reshape([-1,1]),
            np.diag(inv_varAA), group_code, 
            X__h0, int(nAA), int(n), X__h0.shape[1],
            n_groups, int(m), timeLimit, mipGap
        )
        eta_l0_h0 = np.array(res_l0['etaHat'], dtype=bool)
        eta_l0_h0_list[m-1] = eta_l0_h0
        eta_l0 = np.concatenate([eta_l0_h0[:1], np.ones(n_groups-1, dtype=bool), eta_l0_h0[1:]])

        # use eta_l0 to estimate inv_varAA_l0
        XAA__l0 = XAA_[:,eta_l0]
        inv_varAA_l0 = np.ones(nAA)
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
                inv_varAA_l0[groupAA_idx == j] =                     1 / residualAA_l0[groupAA_idx == j].var()
            i += 1
        residualAA_l0_list[m-1] = residualAA_l0

        # calculate BIC
        RSS_l0 = (residualAA_l0**2).sum()
        BIC_list[m-1] = nAA*np.log(RSS_l0/nAA) + (m+1)*np.log(nAA)
        
    m = BIC_list.argmin() + 1
    eta_l0_h0 = eta_l0_h0_list[m-1]
    eta_l0 = np.concatenate([eta_l0_h0[:1], np.ones(n_groups-1, dtype=bool), eta_l0_h0[1:]])
    
    
    #-----------------------------------------
    # use the inv_varAA_l0 in the chosen model as inv_var_l0
    inv_var_l0 = np.ones(n)
    for j in range(n_groups):
        inv_var_l0[group_idx == j] =             1 / residualAA_l0_list[m-1][groupAA_idx == j].var()
    
    # estimate theta_h0
    X__l0 = X_[:,eta_l0]
    X__l0_h0 = X__h0[:,eta_l0_h0]
    A_l0_h0 = X__l0_h0.T.dot(np.multiply(inv_var_l0.reshape([-1, 1]), X__l0_h0))
    b_l0_h0 = X__l0_h0.T.dot(inv_var_l0 * Y)
    try:
        beta_l0_h0 = np.linalg.solve(A_l0_h0,b_l0_h0)
    except np.linalg.LinAlgError:
        beta_l0_h0 = np.linalg.lstsq(A_l0_h0,b_l0_h0, rcond=None)[0]

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

    return pvalue_gmm_l0, eta_l0_h0, m


# In[3]:


# l0 no ridge balance

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
#   timeLimit : time limit in seconds, a termination criterion for Gurobi
#   mipGap : a number belongs to (0,1), a termination criterion for Gurobi. The smaller the mipGap, the  closer the solution to the global optimal point
#   logName : a string indicating file name of the gurobi's log file. Default value is empty string "", which means no log file is created. 
#   printToConsole: a boolean variable indicating whether gurobi print progress to console. Default value is "False".   
#   threadCount: a positive integer indicating maximum number of threads utilized by Gurobi. Default value is 4. 
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

def GurobiAATest_NoBalance(    XAA,    YAA,    WAA,    Tau,    X,    nAA,    n,    p,    q,    m,    timeLimit,    mipGap=0.0001,    logName="",    printToConsole=False,    threadCount=4   ):
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
        raise Exception("m should not be greater than q!")
    
    if not timeLimit>0.0:
        raise Exception("timeLimit should be positive!")
    
    if not mipGap>0.0 and mipGap<1.0:
        raise Exception("mipGap should be positive and less than 1!")
    
    if not isinstance(logName,str):
        raise Exception("logName should be a string!")
    
    if not isinstance(printToConsole,bool):
        raise Exception("printToConsole should be a bool!")
    
    if not isinstance(threadCount,int):
        raise Exception("threadCount should be an integer!")
    
    if not threadCount>0:
        raise Exception("threadCount should be positive!")
    
    # create local environment to allow parallelism
    if printToConsole:
        localEnv = gp.Env(logName)
    else:
        localEnv = gp.Env(logName, empty=True)
        localEnv.setParam('OutputFlag',0)
        localEnv.start()
    
    # create model
    model = gp.Model("ABTestVarianceReduction",localEnv)

    # create beta variables, the associated coefficients of covariates X
    beta = model.addVars(        p,        lb=-GRB.INFINITY,        ub=GRB.INFINITY,        obj=0.0,        vtype=GRB.CONTINUOUS,        name="beta"    )
    
    # create eta variables, the associated binary variable of each corresponding
    # beta is non-zero
    oneMinusEta = model.addVars(        p,        lb=0,        ub=1,        obj=0.0,        vtype=GRB.BINARY,        name="oneMinusEta"    )
    
    # create alpha variable, the associated coefficients of treatment effect \Tau
    alpha = model.addVars(        q,        lb=-GRB.INFINITY,        ub=GRB.INFINITY,        obj=0.0,        vtype=GRB.CONTINUOUS,        name="alpha"    )
    
    # set objective function
    XAATWAAXAA = np.transpose(XAA) @ WAA @ XAA
    YAATWAAXAA = -2 * (np.transpose(YAA) @ WAA @ XAA)
    objective = gp.QuadExpr( gp.quicksum( [ XAATWAAXAA[i-1][j-1] * beta[i-1] * beta[j-1] for i in range(1,p+1) for j in range(1,p+1) ] ) )
    objective.add( gp.quicksum( [ YAATWAAXAA[0][i-1] * beta[i-1] for i in range(1,p+1) ] ) ) 
    objective.addConstant( (np.transpose(YAA) @ WAA @ YAA)[0][0] )
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
    # model.update()
    # del TauTXOvern
    
    # add SOS-1 
    for i in range(1,p+1):
        model.addSOS(GRB.SOS_TYPE1,[ oneMinusEta[i-1] , beta[i-1] ])
    
    model.update()
    
    # add sparsity constraints
    model.addConstr(            oneMinusEta.sum() >=             p-m,            "sparsity_constraint"    )
    model.update()
    
    # set time limit
    model.setParam("TimeLimit", timeLimit)
    model.setParam("MIPGap", mipGap)
#     model.setParam("LogToConsole", int(printToConsole))
    model.setParam("Threads", threadCount)
    
    # solve
    model.optimize()
    
    # extract solution
    betaHat = [ beta[i-1].X for i in range(1,p+1) ]
    etaHat = [1-oneMinusEta[i-1].X for i in range(1,p+1) ]
    runTime = model.runtime
    objVal = model.ObjVal
    objBound = model.ObjBound
    MIPGAP = model.MIPGap
    
    #dispose model
    model.dispose()
    
    #dispose environment
    localEnv.dispose()
    
    return {        "betaHat"  : betaHat,        "etaHat"   : etaHat,        "runTime"  : runTime,        "objVal"   : objVal,        "objBound" : objBound,        "MIPGAP"   : MIPGAP    }


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


# In[ ]:




