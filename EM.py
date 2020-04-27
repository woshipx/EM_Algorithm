# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:03:04 2020

@author: xpanz
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time

# DO NOT IMPORT ANY OTHER PACKAGE(S) THAN ABOVE!!!

# DO NOT MODIFY THIS FUNCTION (GMM_EM)
def GMM_EM(x, init_mu, init_Sigma, init_pi, epsilon=0.001, maxiter=100):
    '''
    GMM-EM algorithm with shared covariance matrix
    arguments:
     - x:          np.ndarray of shape [no_data, no_dimensions]
                   input 2-d data points
     - init_mu:    np.ndarray of shape [no_components, no_dimensions]
                   means of Gaussians
     - init_Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
                   covariance matrix of Gaussians
     - init_pi:    np.ndarray of shape [no_components]
                   prior probabilities of P(z)
     - epsilon:    floating-point scalar
                   stop iterations if log-likelihood increase is smaller than epsilon
     - maxiter:    integer scaler
                   max number of iterations
    returns:
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    '''
    mu = init_mu
    Sigma = init_Sigma
    pi = init_pi
    no_iterations = 0
    
    # compute log-likelihood of P(x)
    logp = np.log(incomplete_likelihood(x, mu, Sigma, pi)).sum()
    print("Init log P(x) is {:.4e}".format(logp))
    
    while True:
        no_iterations = no_iterations + 1
        
        # E step
        gamma = E_step(x, mu, Sigma, pi)
        # M step
        mu, Sigma, pi = M_step(x, gamma)
        
        # exit loop if log-likelihood increase is smaller than epsilon
        # or iteration number reaches maxiter
        new_logp = np.log(incomplete_likelihood(x, mu, Sigma, pi)).sum()
        if new_logp < epsilon + logp or no_iterations > maxiter:
            print("Iteration {:03} log P(x) is {:.4e}".format(no_iterations, new_logp))
            break 
        else:
            print("Iteration {:03} log P(x) is {:.4e}".format(no_iterations, new_logp), end="\r")
            logp = new_logp

    return mu, Sigma, pi

# DO NOT MODIFY THIS FUNCTION (plt_gmm)
def plt_gmm(x, mu, Sigma, pi, path):
    '''
    plots the 3 bivariate Gaussians and save to file
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
     - path:  string
              path to save figure
    '''
    assert x.shape[1] == 2, "must be bi-variate Gaussians"
    assert mu.shape[0] == 3, "must have 3 components"
    gamma = E_step(x, mu, Sigma, pi)
    eigval, eigvec = np.linalg.eig(Sigma)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.scatter(x[:,0], x[:,1], 0.01, c=gamma)
    for centroid,alpha in zip(mu,pi):
        for i in range(0,6,2):
            ell = Ellipse(centroid, np.sqrt(eigval[0])*i, np.sqrt(eigval[1])*i,
                          np.rad2deg(np.arctan2(*eigvec[:,0][::-1])), edgecolor='black', fc='None', lw=2, alpha=alpha)
            ax.add_artist(ell)
    plt.grid(linestyle='--', alpha=0.3)
    plt.savefig(path, dpi=300)
    plt.close(fig)

def E_step(x, mu, Sigma, pi):
    '''
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    returns:
     - gamma: np.ndarray of shape [no_data, no_components]
              probabilities of P(z|x)
    '''
    # todo: implement this function
    (data_size, b) = x.shape
    sum = np.zeros((data_size, 1))
    Numo = np.zeros((data_size, 3))

    for k in range (0, 3):
          G_pdf = Pdf_of_Guassian(x, mu[k, :], Sigma).reshape(data_size, 1)     # calculate the guassian distribution of each point to each cluster, shape of [no_data, 1]
          N = pi[k]*G_pdf   # shape of [no_data, 1]
          Numo[:, k:k+1] = N
          sum = sum + N

    gamma = Numo / sum
    
    return gamma

def M_step(x, gamma):
    '''

    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - gamma: np.ndarray of shape [no_data, no_components]
              probabilities of P(z|x)
    returns:
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    '''
    # todo: implement this function

    (data_size, d) = x.shape
    mu = np.zeros((3, 2))
    Sigma = np.zeros((2, 2))
    sum_gamma = np.sum(gamma, axis=0)     # sum of gamma in each colonm
    pi = (np.sum(gamma, axis=0))/data_size     # update pi
    
    for i in range(0, 3):
        mu[i, :] = np.sum(gamma[:, i].reshape(data_size, 1)*x, axis=0)/sum_gamma[i]     # update mu

        Dif = x - mu[i, :]      # difference between each x and new mu, shape of [no_data, 2]
        Sigma_num = np.dot(np.transpose((gamma[:, i].reshape(data_size, 1) * Dif)), Dif)      # update Sigma
        Sigma = Sigma + Sigma_num
    
    Sigma = Sigma / data_size
        #print('sigma= ', Sigma)

    return mu, Sigma, pi


def incomplete_likelihood(x, mu, Sigma, pi):
    '''
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    returns:
     - px:    np.ndarray of shape [no_data]
              probabilities of P(x) at each data point in x
    '''
    # todo: implement this function
    (data_size, d) = x.shape
    px = np.zeros(data_size)
    
    for k in range(0, 3):
        G_pdf = Pdf_of_Guassian(x, mu[k, :], Sigma)
        pi = pi.reshape(1, 3)
        p = pi[:, k]*G_pdf
        px = px + p

    return px


# YOU MAY DEFINE OTHER FUNCTION(S) AS YOU SEE FIT
#
# def my_function_name(...):
def Pdf_of_Guassian(x, mu,Sigma):
    '''
    arguments:
    - x:  np.ndarray of shape [1, no_dimensions]
    - mu: np.ndarray of shape [1, no_dimensions]
    
    returns:
    - G_pdf: the probability of gaussian distribution
    '''    
    
    Dif = x - mu    # difference of input x and mu is shape of [data_size * 2]
    Sigma_inv = np.linalg.inv(Sigma)
    # print(Sigma_inv)
    G_1 = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))   # first part o
    # print('G_1', G_1)
    G_2 = np.exp(-0.5 * np.sum(Dif * np.transpose(np.dot(Sigma_inv, np.transpose(Dif))), axis=1))   #second part
    # print('G_2',G_2)
    G_pdf = G_1 * G_2
    
    
    return G_pdf
    
#     ...
#     return ...


# ALL TESTS SHOULD GO INSIDE THIS CONDITIONAL
if __name__ == '__main__':
    # set random seed for PRNG
    np.random.seed(int(time.time()))
    print(np.random.seed(int(time.time())))
    # load data
    x = np.load('x.npy')

    # a =  np.random.normal(loc=5, scale=1, size=(10000,2))
    # y = np.random.normal(loc=15, scale=1, size=(10000,2))
    # z = np.random.normal(loc=10, scale=1, size=(10000,2))
    # x = np.vstack((a,y,z))


    ### todo: try different initialisation methods ###
    ### by replacing this block ######################
    # below is a not-so-accurate initial guess of params


    init_mu = np.random.random((3,2))
    init_mu = np.array([[0.2, 0.5], [2, -0.5], [1, 1]])
    # init_mu = np.array([[0.2, 0.5],[2,-0.5],[1, 1]])
    # init_Sigma = np.eye(2)
    init_Sigma = np.diag(np.diag(np.random.random((2, 2))))
    init_pi = np.array([0.4, 0.3, 0.3])
    ### initialisation block ends here ###############

    mu, Sigma, pi = GMM_EM(x, init_mu, init_Sigma, init_pi)
    # print('mu{:03} =' .format(i), mu)
    # print('Sigma{:03} = '.format(i), Sigma)
    # print('pi{:03} = '.format(i), pi)

        

    # plot figure and save it as 'result.png'
    # Please include this figure in your PDF
    # together with final values of mu, Sigma, pi and
    # log-likelihood log P(x) under this Gaussian mixture
    plt_gmm(x, mu, Sigma, pi, 'result1.png')


    # D = E_step(x, mu, Sigma, pi)
    # E = M_step(x,D)
    # print(E)


