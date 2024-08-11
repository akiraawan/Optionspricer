import numpy as np


class BinomialTree:
    def __init__(self, S0, K, r, T, sigma, steps, q=None, type='call'):
        self.type = type
        self.steps = steps
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = float(T)/365
        self.sigma = sigma
        self.q = q
        self.dt = T/steps
        self.u = np.exp(self.sigma * np.sqrt(self.dt/steps))
        self.d = 1/self.u
        

    def price(self):
        stock_tree = np.zeros((self.steps+1, self.steps+1))
        for i in range(self.steps+1):
            for j in range(i+1):
                stock_tree[j, i] = self.S0 * (self.u ** j) * (self.d ** (i-j))
        
        option_tree = np.zeros((self.steps+1, self.steps+1))
        for j in range(self.steps+1):
            option_tree[j, self.steps] = max(0, option_tree[j, self.steps] - self.K) if (self.type == 'call') else max(0, self.K - option_tree[j, self.steps])

        for 

    def delta(self):
        pass

    def gamma(self):
        pass

    def theta(self):
        pass

    def vega(self):
        pass

    def rho(self):
        pass