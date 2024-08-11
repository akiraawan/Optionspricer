import numpy as np
import scipy
from scipy.stats import norm
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
import os


class BSMFormulae:
    ''' 
    Class to compute the Black-Scholes-Merton formulae for European options.
    Assumes that interest rates and volatility are constant and stock doesn't pay dividends.
    '''
    def __init__(self, S0, K, r, T, sigma, type='call'):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = float(T)/365
        self.sigma = sigma
        self.type = type
        self.price = self.price()

    def d1(self):
        return (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def price(self):
        if self.type == 'call':
            return self.S0 * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S0 * norm.cdf(-self.d1())
        
    def _delta(self):
        if self.type == 'call':
            return norm.cdf(self.d1())
        else:
            return norm.cdf(self.d1()) - 1
        
    def _gamma(self):
        return norm.pdf(self.d1()) / (self.S0 * self.sigma * np.sqrt(self.T))

    def _theta(self):
        if self.type == 'call':
            return -self.S0 * norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())
        else:
            return -self.S0 * norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2())
        
    def _vega(self):     
        return self.S0 * norm.pdf(self.d1()) * np.sqrt(self.T)
        
    def _rho(self):
        if self.type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2())
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2())
        


    