import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.special import gamma, digamma
from scipy.stats import multivariate_normal as mvn
from scipy import stats


def VariationalInference(x, y, z, a0=10E-16, b0=10E-16, e0=1, f0=1, T=500):

    objective_list = np.empty(T)
    N, d = x.shape
    e1 = e0
    f1 = f0
    a0 = np.ones(d)*a0
    a1 = np.ones(d)*a0
    b0 = np.ones(d)*b0
    b1 = np.ones(d)*a0


    for t in xrange(T):
    # updates the variational parameters a0, b0, e0, f0, define mu1, simga1 so they have values 


        sigma1 = np.linalg.inv((1.0*e1/f1)*np.dot(x.T,x) + np.diag(1.0*a1/b1)) #matrix 101 x 101
        mu1 = sigma1.dot((1.0*e1/f1)*x.T.dot(y)) #101 vector
    
        a1 = a0 + 0.5 #vector of d
        # b1(k) = b0 + 0.5*(sigma[k,k] + mu[k]*mu[k])
        for k in xrange(d):
            b1[k] = b0[k] + 0.5*(sigma1[k,k] + mu1[k]*mu1[k]) #vector of d
        e1 = e0 + N/2.0 #scaler
        error = 0
        for i in xrange(N):
            error += (y[i] - np.dot(x[i].T,mu1))**2 + x[i].dot(sigma1).dot(x[i])
        f1 = f0 + 0.5*error
    
        whatsthis = objective(x, y, a0, b0, a1, b1, e0, e1, f0, f1, mu1, sigma1)
        # print "whats this:", whatsthis.shape
        objective_list[t] = whatsthis

    # print objective_list
    plt.plot(objective_list)
    plt.title('Objective')
    plt.show()
    plt.title('1/E[alphas]')
    plt.plot((b1/a1))
    plt.show()
    print '1/E[lambda]:', (f1/e1)

    predictions = x.dot(mu1)

    plt.plot(z, predictions)
    plt.title('Part d')
    plt.scatter(z,y)
    # plt.title('scatter of z, y')
    plt.plot(z, 10*np.sinc(z))
    # plt.title('zi, 10 *sinc(zi)')
    plt.show()
    
    
#setup objective functions
def objective(x, y, a0, b0, a1, b1, e0, e1, f0, f1, mu1, sigma1):
    N, d = x.shape
    # calculation of joint likelihood
    E_lnp_lambda = e0*np.log(f0) + (e0-1.0)*(digamma(e1)-np.log(f1))-1.0*f0*(e1/f1)
    function_sum = 0

    # print 'a1.shape:', a1.shape
    # print 'b1.shape:', b1.shape
    # print 'sigma1.shape:', sigma1.shape
    # print 'mu1.shape:', mu1.shape
    for k in xrange(d):
        function_sum -= .5*a1[k]/b1[k] * (sigma1[k,k] + mu1[k]**2)
    E_lnp_w = (-d/2)*np.log(2*np.pi) + 0.5*np.sum(digamma(a1)-np.log(b1)) + function_sum
    
    # E[ -1/2 sum[k] { alpha[k] * w[k]**2 } ]
    # -1/2 sum[k] { E[alpha[k]] E[w[k]^2] }
    # -1/2 sum[k] { a[k]/b[k] * (cov[k,k] - mu[k]^2) }

    # calculating Elnpalpha
    # E_lnp_alpha = np.ones(d)
    E_ln_alpha = digamma(a1) - np.log(b1)
    # E_alpha = a1 / b1
    E_lnp_alpha = a0*np.log(b0) - np.log(gamma(a0)) + (a0-1)*E_ln_alpha - 1.0*b0*a1/b1
    
    # calcuation Elnpy
    function_sum2 = 0
    for i in xrange(N):
        function_sum2 += (y[i] - x[i].dot(mu1))**2 + x[i].dot(sigma1).dot(x[i])

    E_lnp_y = -.5*N*np.log(np.pi) + .5*N*(digamma(e1) - np.log(f1)) - .5*e1/f1 * function_sum2


    # calculation of the q values
    # E_lnq_lambda = e1*np.log(f1) + np.log(stats.gamma.entropy(e1))+(1.0-e1)*gamma(e1)
    E_lnq_lambda = -stats.gamma.entropy(a=e1, scale=1./f1)
    E_lnq_w = -mvn.entropy(cov=sigma1)

    E_lnq_alpha = 0
    for k in xrange(d):
        E_lnq_alpha -= stats.gamma.entropy(a=a1[k], scale=1./b1[k])

    sum1 = E_lnp_lambda + E_lnp_w + E_lnp_alpha.sum() + E_lnp_y 
    sum2 = E_lnq_lambda + E_lnq_w + E_lnq_alpha.sum()

    variational_objective = sum1 - sum2

    return variational_objective
# def e_ln_q_gamm(a, b):
#     return np.log(b) - a - np.log(np.abs(gamma(a))) + (a - 1)*digamma(a)

# (x - loc) / scale

if __name__ == '__main__':

    path = '/Users/laurenmccarthy/Documents/Columbia/Fall2016/BaysianML/hw3/data_csv-3/'
    
    for i in (1,2,3):
    # for i in (1,):
        x = pd.read_csv(path+'X_set%s.csv' % i, header=None).as_matrix()
        y = pd.read_csv(path+'y_set%s.csv' % i, header=None).as_matrix().flatten()
        z = pd.read_csv(path+'z_set%s.csv' % i, header=None).as_matrix().flatten()
        VariationalInference(x, y, z, a0=10E-16, b0=10E-16, e0=1, f0=1, T=500)



