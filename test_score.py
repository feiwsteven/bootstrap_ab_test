#!/usr/bin/env python
# coding: utf-8


# In[4]:


def t_test(Yc, Yt):
    pvalue_t = st.ttest_ind(Yc, Yt, equal_var=False)[1]
    return pvalue_t

def F_test(*Y):
    pvalue_F = st.f_oneway(*Y)[1]
    return pvalue_F

def CUPED(Xc, Yc, Xt, Yt):
    Xc_ = sm.add_constant(Xc)
    theta_c = sm.OLS(Yc, sm.add_constant(Xc_)).fit().params
    Yc_cv = Yc - np.dot(Xc_-np.mean(Xc_, axis=0), theta_c)
    
    Xt_ = sm.add_constant(Xt)
    theta_t = sm.OLS(Yt, Xt_).fit().params
    Yt_cv = Yt - np.dot(Xt_-np.mean(Xt_, axis=0), theta_t)
    
    pvalue_CUPED = st.ttest_ind(Yc_cv, Yt_cv, equal_var=False)[1]
    return pvalue_CUPED



def gmm(X: np.ndarray, Y: np.ndarray, n_samples: list, max_iter: int):
    n_treatments = len(n_samples)
    sample_size = sum(n_samples)
    treatment_idx = np.repeat(np.arange(n_treatments), n_samples)
    treatment_code = np.zeros([sample_size, n_treatments])
    treatment_code[np.arange(sample_size), treatment_idx] = 1

    X_ = np.concatenate([np.ones([sample_size, 1]),
                             treatment_code[:, 1:],
                             X], axis=1)
    X__h0 = np.concatenate([np.ones([sample_size, 1]), X], axis=1)

    inv_variance = np.ones(sample_size)
    inv_variance_h0 = np.ones(sample_size)

    i = 0
    while i < max_iter:
        theta = np.linalg.solve(
            X_.T.dot(
                np.multiply(inv_variance.reshape([-1, 1]), X_)
            ),
            X_.T.dot(inv_variance * Y)
        )

        residual = Y - X_.dot(theta)
        for j in range(n_treatments):
            inv_variance[treatment_idx == j] =                                 1 / residual[treatment_idx == j].var()

        i += 1

    X__weighted = X_ * inv_variance.reshape([-1, 1])
    g = X__weighted.T.dot(Y - X_.dot(theta)) / sample_size
    G = - X__weighted.T.dot(X_) / sample_size

    v_mat = X__weighted.T.dot(X_) / sample_size

    # variance of GMM estimator is
    # \sqrt n (\hat \theta - \theta_0) -> N(0, (G W G^T)^{-1})
    # G = \partial \bar g(theta)^T / \partial \theta

    w_mat = np.linalg.inv(v_mat)
    variance = np.linalg.inv(G.dot(w_mat).dot(G.T)).diagonal() / sample_size
    
    gmm = g.dot(w_mat).dot(g) * sample_size
                                
    
    # under the H_0
    i = 0
    while i < max_iter:
        theta_h0 = np.linalg.solve(
            X__h0.T.dot(
                np.multiply(inv_variance_h0.reshape([-1, 1]), X__h0)
            ),
            X__h0.T.dot(inv_variance_h0 * Y)
        )
                                
        residual_h0 = Y - X__h0.dot(theta_h0)
        for j in range(n_treatments):
            inv_variance_h0[treatment_idx == j] =                                 1 / residual_h0[treatment_idx == j].var()

        i += 1
    
    X__weighted_h0 = X__h0 * inv_variance_h0.reshape([-1, 1])
    g_h0 = X__weighted_h0.T.dot(Y - X__h0.dot(theta_h0)) / sample_size
    G_h0 = - X__weighted_h0.T.dot(X__h0) / sample_size

    residual_h0 = (Y - X__h0.dot(theta_h0)).reshape([-1, 1])
    omega_h0 = (X__weighted_h0 * residual_h0).T.dot(
        X__weighted_h0 * residual_h0) / sample_size
    
    sigma_h0 = G_h0.dot(np.linalg.inv(omega_h0)).dot(G_h0.T)
    d_gmm_h0 = G_h0.dot(np.linalg.inv(omega_h0)).dot(g_h0)
    score = d_gmm_h0.T.dot(np.linalg.inv(sigma_h0)).dot(d_gmm_h0) * sample_size
    
    gmm_h0 = sample_size * g_h0.dot(np.linalg.inv(omega_h0)).dot(g_h0)
    pvalue_gmm_score = 1 - st.chi2.cdf(score, df=n_treatments)

    likelihood_ratio = gmm_h0 - gmm
    pvalue_gmm_lr = 1 - st.chi2.cdf(likelihood_ratio, df=n_treatments)
    
    return pvalue_gmm_score, pvalue_gmm_lr


# In[5]:


# c = cdg.control(1000)
# t = cdg.treatment(1000)
# X = np.row_stack([c['X'], t[0]['X']])
# Y = np.concatenate([c['Y'], t[0]['Y']])
# print(gmm(X, Y, [1000, 1000], 5))


# In[9]:


tests_str = ['t test','F test','CUPED', 'GMM score test', 'GMM LR test']

def power(cdg: ClinicalDataGenerator,
          n_samples_c: int, n_samples_t: list, 
          level=0.05, n_reps=1000, test_arm=0):
    n_samples_t = np.reshape(n_samples_t, -1) # take care of only one treatment arm
    count_t, count_F, count_CUPED, count_gmm_score, count_gmm_lr = [0]*5

    for i in range(n_reps):
        if (i+1) % (n_reps/100) < 1:
            print(format((i+1)/n_reps*100, '.0f') +
                  '% replicates complete...', end='\r')
        
        if i+1 == n_reps:
            print('100% replicates complete   ', end='\r')
            print(end='\n')
        
        
        control_arm = cdg.control(n_samples_c)
        Xc = control_arm['X']
        Yc = control_arm['Y']
        
        treatment_arms = cdg.treatment(*n_samples_t)
        Xt = [arm['X'] for arm in treatment_arms]
        Yt = [arm['Y'] for arm in treatment_arms]
        
        pvalue_t = t_test(Yc, Yt[test_arm])
        if pvalue_t < level:
            count_t += 1
            
        pvalue_F = F_test(Yc, *Yt)
        if pvalue_F < level:
            count_F += 1
        
        pvalue_CUPED = CUPED(Xc, Yc, Xt[test_arm], Yt[test_arm])
        if pvalue_CUPED < level:
            count_CUPED += 1
        
#         X = np.row_stack([Xc, *Xt])
#         Y = np.concatenate([Yc, *Yt])

#         pvalue_gmm_score, pvalue_gmm_lr = gmm(X, Y, [n_samples_c, *n_samples_t], 5)
#         if pvalue_gmm_score < level:
#             count_gmm_score += 1
#         if pvalue_gmm_lr < level:
#             count_gmm_lr += 1
    
    return {tests_str[0]: count_t/n_reps,
            tests_str[1]: count_F/n_reps,
            tests_str[2]: count_CUPED/n_reps, 
            tests_str[3]: count_gmm_score/n_reps,
            tests_str[4]: count_gmm_lr/n_reps}


# In[567]:


# test
n_samples_c = 1000
n_samples_t = [1000] * 1
cdg = ClinicalDataGenerator(
#                       alphaX = ('G-copula', [[1,0,0],[0,1,0],[0,0,1]],
#                                 {'alphaX_alpha': (('normal',0,1),),
#                                  'alphaX_X': (('normal',[0,1],1),)}),
                      alpha = (('constant',0.05),),
                      X = (('normal',0,1), ('normal',0,1),),
                      beta = (('constant',[1]*2),), 
                      epsilon = ('normal',0,1)
)

pwr = power(cdg, n_samples_c, n_samples_t)
print(pwr)


# In[15]:


# alpha1 in [0, 0.2]
points = 200
alpha1s = np.linspace(0,0.3, points)
pwrs_alpha1 = np.zeros([points, 3])
for i,alpha1 in enumerate(alpha1s):
    print(i+1)
    n_samples_c = 1000
    n_samples_t = [1000] * 2
    cdg = ClinicalDataGenerator(
                      alpha = (('constant',alpha1), ('constant',0)),
                      X = (('normal',0,1), ) * 3,
                      beta = (('constant',[1] * 3),), 
                      epsilon = ('normal',0,1)
    )
    pwr = power(cdg, n_samples_c, n_samples_t)
    pwrs_alpha1[i,:] = [pwr[tests_str[0]], pwr[tests_str[1]], pwr[tests_str[2]]]


# np.save('pwrs_alpha1', pwrs_alpha1)
    
    
for i,test in enumerate(tests_str[:3]):
    plt.plot(alpha1s, pwrs_alpha1[:,i], label=test)
plt.xlabel('First Treatment Effect')
plt.ylabel('Power')
plt.legend()


# In[13]:


# p -> infinity
points = 200
n_cors = np.ceil(10**np.linspace(0,2, points)).astype(int)
pwrs_p_infty = np.zeros([points, 3])
for i,n_cor in enumerate(n_cors):
    print(i+1)
    n_samples_c = 1000
    n_samples_t = [1000] * 2
    cdg = ClinicalDataGenerator(
                      alpha = (('constant',0.3), ('constant',0)),
                      X = (('normal',0,1), ) * n_cor,
                      beta = (('constant',[1] * n_cor),), 
                      epsilon = ('normal',0,1)
    )
    pwr = power(cdg, n_samples_c, n_samples_t)
    pwrs_p_infty[i,:] = [pwr[tests_str[0]], pwr[tests_str[1]], pwr[tests_str[2]]]


# np.save('pwrs_p_infty', pwrs_p_infty)
    
    
for i,test in enumerate(tests_str[:3]):
    plt.plot(n_cors, pwrs_p_infty[:,i], label=test)
plt.xscale('log')
plt.xlabel('Number of Covariates')
plt.ylabel('Power')
plt.legend()


# In[10]:


# X_rho in [-1,1]
points = 200
rhos = np.linspace(-1,1,points)
pwrs_Xrho = np.zeros([points, 3])
for i,rho in enumerate(rhos):
    print(i+1)
    n_samples_c = 1000
    n_samples_t = [1000] * 2
    cdg = ClinicalDataGenerator(
                      alpha = (('constant',0.2), ('constant',0)),
                      X = (('multi-normal',[0]*3,[[1,rho,rho],[rho,1,rho],[rho,rho,1]]), ),
                      beta = (('constant',[1]*3),), 
                      epsilon = ('normal',0,1)
    )
    pwr = power(cdg, n_samples_c, n_samples_t)
    pwrs_Xrho[i,:] = [pwr[tests_str[0]], pwr[tests_str[1]], pwr[tests_str[2]]]

# np.save('pwrs_Xrho', pwrs_Xrho)


for i,test in enumerate(tests_str[:3]):
    plt.plot(rhos, pwrs_Xrho[:,i], label=test)
plt.xlabel('Correlation within X')
plt.ylabel('Power')
plt.legend()


# In[12]:


# alphaX_rho in [-1,1]
points = 200
rhos = np.linspace(-1,1,points)
pwrs_alphaXrho = np.zeros([points, 3])
for i,rho in enumerate(rhos):
    print(i+1)
    n_samples_c = 1000
    n_samples_t = [1000] * 2
    cdg = ClinicalDataGenerator(
                      alphaX = ('G-copula', [[1,0,rho,rho,rho],[0,1,rho,rho,rho],[rho,rho,1,0,0],[rho,rho,0,1,0],[rho,rho,0,0,1]], 
                                {'alphaX_alpha': (('normal',0.2,1), ('normal',0,1)),
                                 'alphaX_X': (('normal',[0]*3,1),)}),
                      beta = (('constant',[1]*3),), 
                      epsilon = ('normal',0,1)
    )
    pwr = power(cdg, n_samples_c, n_samples_t)
    pwrs_alphaXrho[i,:] = [pwr[tests_str[0]], pwr[tests_str[1]], pwr[tests_str[2]]]

# np.save('pwrs_alphaXrho', pwrs_alphaXrho)


for i,test in enumerate(tests_str[:3]):
    plt.plot(rhos, pwrs_alphaXrho[:,i], label=test)
plt.xlabel('Correlation between Treatment Effect and Covariate')
plt.ylabel('Power')
plt.legend()