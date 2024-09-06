import numpy as np
from scipy.stats import norm
from .monte_carlo import MonteCarlo


class MonteCarloEuropean(MonteCarlo):
    def calculate_price(self):
        self.stock_prices = np.zeros((self.sims, self.steps+1))  # each row is a path
        self.stock_prices[:, 0] = self.S0

        # Generate stock prices
        for t in range(1, self.steps+1):
            Z = norm.rvs(size=self.sims)
            self.stock_prices[:, t] = self.stock_prices[:, t-1] * np.exp(((self.r - self.q) - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)

        self.payoffs = np.maximum(0, self.stock_prices[:, -1] - self.K) if (self.option_type == 'call') else np.maximum(0, self.K - self.stock_prices[:, -1])

        self.discounted_payoffs = np.exp(-self.r * self.T) * self.payoffs

        return np.mean(self.discounted_payoffs)
