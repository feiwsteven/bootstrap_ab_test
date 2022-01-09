#!/usr/bin/env python3
# run "pip3 install gurobipy" in command line first

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math

# set random number generator seed
rng = np.random.default_rng(74809)
# number of covariates X
p = 20
# number of treatment arms
q = 3
# sample size of each arm in A/B test
nEachArm = 500
# sample size of A/B test
n = nEachArm * q
# sample size of A/A test
nAA = 5000
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
# n by q treatment indicator \Tau
Tau = np.array([[float(i/nEachArm<=j and j-1<i/nEachArm) for j in range(1,q+1) ] for i in range(1,n+1)])
# p by 1 beta0, the true coefficient associated with X to predict Y
beta0 = np.array([[float(i<=m)] for i in range(1,p+1)])
# q by 1 alpha0, the true coefficient associated with \Tau to predict Y
alpha0 = np.array([[i*0.1] for i in range(1,q+1)])
# inverse of covariance matrix of residual in A/A test, which is YAA-XAA*beta0
WAA=np.array([ [float(i==j) for j in range(1,nAA+1)] for i in range(1,nAA+1)])
# n by 1 outcome variable Y of A/B test
Y = np.matmul( X , beta0 ) + np.matmul( Tau , alpha0 ) + np.transpose( [np.random.standard_normal(size=n) * math.sqrt(sigmaSquare)] )
# nAA by 1 outcome variable YAA of A/A test
YAA = np.matmul( XAA , beta0 ) + np.transpose( [np.random.standard_normal(size=nAA) * math.sqrt(sigmaSquare)] )
#lam is ridge penalty's tuning parameter
lam=1.0





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

def GurobiAATest_RidgeBalance(\
    XAA,\
    YAA,\
    WAA,\
    Tau,\
    X,\
    nAA,\
    n,\
    p,\
    q,\
    m,\
    lam,\
    timeLimit,\
    mipGap=0.0001\
   ):
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
        raise Exception("Tau's column dimensions should be q!")
    
    if lam<0:
        raise Exception("lam should be non-negative!")
    
    if not timeLimit>0.0:
        raise Exception("timeLimit should be positive!")
    
    if not mipGap>0.0 and mipGap<1.0:
        raise Exception("mipGap should be positive and less than 1!")
    
    # create model
    model = gp.Model("ABTestVarianceReduction")
    
    # create beta variables, the associated coefficients of covariates X
    beta = model.addVars(\
        p,\
        lb=-GRB.INFINITY,\
        ub=GRB.INFINITY,\
        obj=0.0,\
        vtype=GRB.CONTINUOUS,\
        name="beta"\
    )
    
    # create eta variables, the associated binary variable of each corresponding
    # beta is non-zero
    oneMinusEta = model.addVars(\
        p,\
        lb=0,\
        ub=1,\
        obj=0.0,\
        vtype=GRB.BINARY,\
        name="oneMinusEta"\
    )
    
    # create alpha variable, the associated coefficients of treatment effect \Tau
    alpha = model.addVars(\
        q,\
        lb=-GRB.INFINITY,\
        ub=GRB.INFINITY,\
        obj=0.0,\
        vtype=GRB.CONTINUOUS,\
        name="alpha"\
    )
    
    # create rho variables, which are (Tau^T * Tau)^{-1}*(Tau^T * X * beta) 
    rho = model.addVars(\
        q,\
        lb=-GRB.INFINITY,\
        ub=GRB.INFINITY,\
        obj=0.0,\
        vtype=GRB.CONTINUOUS,\
        name="rho"\
    )
    
    # create mu variables, which are rho's ordered ascendingly
    mu = model.addVars(\
        q,\
        lb=-GRB.INFINITY,\
        ub=GRB.INFINITY,\
        obj=0.0,\
        vtype=GRB.CONTINUOUS,\
        name="mu"\
    )
    
    # create rhoMinusMu variables, which are rho[i]-mu[j]
    rhoMinusMu = model.addVars(\
       q*q,\
       lb=-GRB.INFINITY,\
       ub=GRB.INFINITY,\
       obj=0.0,\
       vtype=GRB.CONTINUOUS,\
       name="rhoMinusMu"\
    )
    
    # create zeta variables, which are binary variables to attain mu   
    zeta = model.addVars(\
        q*q,\
        lb=0.0,\
        ub=1.0,\
        obj=0.0,\
        vtype=GRB.BINARY,\
        name="zeta"\
    )
    
    # set objective function
    XAATWAAXAA = np.transpose(XAA) @ WAA @ XAA
    YAATWAAXAA = -2.0 * (np.transpose(YAA) @ WAA @ XAA)
    objective = gp.QuadExpr( gp.quicksum( [ XAATWAAXAA[i-1][j-1] * beta[i-1] * beta[j-1] for i in range(1,p+1) for j in range(1,p+1) ] ) )
    objective.add( gp.quicksum( [ YAATWAAXAA[0][i-1] * beta[i-1] for i in range(1,p+1) ] ) ) 
    objective.addConstant( (np.transpose(YAA) @ WAA @ YAA)[0][0] )
    for i in range(1,q):
        objective.addTerms(\
                [ lam,   -2.0*lam, lam     ],\
                [ mu[i], mu[i],    mu[i-1] ],\
                [ mu[i], mu[i-1],  mu[i-1] ]\
        )
    
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
        model.addConstr(\
                gp.quicksum( [ TauTXOvern[i-1][j-1] * beta[j-1] for j in range(1,p+1) ] ) == \
                rho[i-1],\
                "rho_" + str(i)\
        )
    
    model.update()
    del TauTXOvern
    
    #add constraints to let rhoMinusMu equal to rho[i]-mu[j]
    for i in range(1,q+1):
        for j in range(1,q+1):
            model.addConstr(\
                    rho[i-1] - mu[j-1] == \
                    rhoMinusMu[(i-1)*q+(j-1)],\
                    "rhoMinusMu_" + str(i) + "_" + str(j)\
            )
    
    model.update()
    
    # add SOS-1 related to zeta and rhoMinusMu
    for i in range(1,q+1):
        for j in range(1,q+1):
            model.addSOS(\
                    GRB.SOS_TYPE1,\
                    [ rhoMinusMu[(i-1)*q+(j-1)] , zeta[(i-1)*q+(j-1)] ]
            )
    
    model.update()
    
    # add constraints related to zeta 
    for i in range(1,q+1):
        model.addConstr(\
            gp.quicksum( [ zeta[(i-1)*q+(j-1)] for j in range(1,q+1) ] ) == 1.0,\
            "zeta_row_" + str(i)\
        )
    
    for i in range(1,q+1):
        model.addConstr(\
           gp.quicksum( [ zeta[(j-1)*q+(i-1)] for j in range(1,q+1) ] ) == 1.0,\
           "zeta_col_" + str(i)\
        )
    
    model.update()
    
    # add SOS-1 related to eta
    for i in range(1,p+1):
        model.addSOS(GRB.SOS_TYPE1,[ oneMinusEta[i-1] , beta[i-1] ])
    
    model.update()
    
    # add sparsity constraints
    model.addConstr(\
            oneMinusEta.sum() >= \
            p-m,\
            "sparsity_constraint"\
    )
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
    return {\
        "betaHat"  : betaHat,\
        "etaHat"   : etaHat,\
        "rhoHat"   : rhoHat,\
        "muHat"    : muHat,\
        "zetaHat"  : zetaHat,\
        "runTime"  : runTime,\
        "objVal"   : objVal,\
        "objBound" : objBound,\
        "MIPGAP"   : MIPGAP\
    }


output=GurobiAATest_RidgeBalance(\
    XAA,\
    YAA,\
    WAA,\
    Tau,\
    X,\
    nAA,\
    n,\
    p,\
    q,\
    m,\
    0.0,\
    timeLimit,\
    mipGap\
)

output2=GurobiAATest_RidgeBalance(\
    XAA,\
    YAA,\
    WAA,\
    Tau,\
    X,\
    nAA,\
    n,\
    p,\
    q,\
    m,\
    0000.0,\
    timeLimit,\
    mipGap\
)
