import numpy as np


class BinomialTreeAmerican:
    def __init__(self, S0, K, r, T, sigma, steps, q=0, option_type='call'):
        self.option_type = option_type
        self.steps = steps
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = float(T)/365
        self.sigma = sigma
        self.q = q
        self.dt = self.T/steps
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1/self.u
        self.p = (np.exp((self.r-self.q)*self.dt) - self.d) / (self.u - self.d)
        self.price = self._price()

    def _price(self):
        stock_tree = np.zeros((self.steps+1, self.steps+1))
        for j in range(self.steps+1):
            for i in range(j+1):
                stock_tree[i, j] = self.S0 * (self.u ** (j-i)) * (self.d ** (i))
        
        option_tree = np.zeros((self.steps+1, self.steps+1))
        for i in range(self.steps+1):
            option_tree[i, self.steps] = max(0, stock_tree[i, self.steps] - self.K) if (self.option_type == 'call') else max(0, self.K - stock_tree[i, self.steps])

        for j in range(self.steps-1, -1, -1):
            for i in range(j+1):
                if (self.option_type == 'call'):
                    option_tree[i, j] = max(stock_tree[i, j] - self.K, np.exp(-(self.r-self.q)*self.dt) * (self.p * option_tree[i, j+1] + (1-self.p) * option_tree[i+1, j+1]))
                else:
                    option_tree[i, j] = max(self.K - stock_tree[i, j], np.exp(-(self.r-self.q)*self.dt) * (self.p * option_tree[i, j+1] + (1-self.p) * option_tree[i+1, j+1]))
        return option_tree[0, 0]
        

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