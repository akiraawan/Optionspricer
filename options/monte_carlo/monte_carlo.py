from abc import abstractmethod
import numpy as np


class MonteCarlo:
    def __init__(self, S0, K, r, T, sigma, steps, sims, q=0, option_type='call'):
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
        q: float
            dividend yield of the underlying asset
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = float(T)/365
        self.sigma = sigma
        self.steps = steps
        self.sims = sims
        self.q = q
        self.option_type = option_type
        self.dt = self.T/steps

        self.price = self.get_price()

    def get_price(self):
        self.calculate_price()
        return np.mean(self.discounted_payoffs)

    @abstractmethod
    def calculate_price(self):
        pass

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
