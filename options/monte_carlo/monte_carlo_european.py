from abc import abstractmethod
import numpy as np
from scipy.stats import norm


class MonteCarlo:
    def __init__(self, S0, K, r, T, sigma, steps, sims, option_type='call'):
        """
        Construct a Monte Carlo Simulation for european option pricing.

        Parameters
        ----------
        S0: float
            current price of the underlying asset
        K: float
            strike price of the option
        r: float
            risk-free interest rate
        sigma: float
            volatility of the underlying asset
        T: float
            time to maturity of the option specified in days
        steps: int
            number of steps in the binomial tree
        sims: int
            number of simulations
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = float(T)/365
        self.sigma = sigma
        self.steps = steps
        self.sims = sims
        self.option_type = option_type
        self.dt = self.T/steps

        self.price = self._price()

    def _price(self):
        self.stock_prices = np.zeros((self.sims, self.steps+1)) # each row is a path
        self.stock_prices[:, 0] = self.S0
        
        # Generate stock prices
        for t in range(1, self.steps+1): 
            Z = norm.rvs(size=self.sims)
            self.stock_prices[:, t] = self.stock_prices[:, t-1] * np.exp((self.r - 0.5*self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*Z)

        self.payoffs = np.maximum(0, self.stock_prices[:, -1] - self.K) if (self.option_type == 'call') else np.maximum(0, self.K - self.stock_prices[:, -1])

        self.discounted_payoffs = np.exp(-self.r * self.T) * self.payoffs

        return np.mean(self.discounted_payoffs)

    def delta(self):
        raise NotImplementedError

    def gamma(self):
        raise NotImplementedError

    def theta(self):
        raise NotImplementedError

    def vega(self):
        raise NotImplementedError

    def rho(self):
        raise NotImplementedError

    def display(self):
        pass
y
