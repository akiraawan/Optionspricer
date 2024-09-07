import numpy as np
from .monte_carlo import MonteCarlo


class MonteCarloAmerican(MonteCarlo):
    def calculate_price(self):
        discount_factor = np.exp(-self.r * self.dt)

        # Step 1: Generate asset price paths
        self.stock_prices = np.zeros((self.steps + 1, self.sims))
        self.stock_prices[0] = self.S0
        for t in range(1, self.steps + 1):
            Z = np.random.standard_normal(self.sims)
            self.stock_prices[t] = self.stock_prices[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)

        # Step 2: Calculate payoffs
        self.payoffs = np.maximum(self.K - self.stock_prices, 0)

        # Step 3: Propogate backwards to estimate the continuation value
        for t in range(self.steps - 1, 0, -1):
            is_profitable = self.payoffs[t] > 0
            X = self.stock_prices[t, is_profitable]
            Y = self.payoffs[t + 1, is_profitable] * discount_factor

            # Degree 2 polynomial regression
            if len(X) > 0:
                A = np.vstack([np.ones_like(X), X, X**2]).T
                coefficients = np.linalg.lstsq(A, Y, rcond=None)[0]
                continuation_value = np.dot(A, coefficients)

                exercise = self.payoffs[t, is_profitable] > continuation_value
                self.payoffs[t, is_profitable] = np.where(exercise, self.payoffs[t, is_profitable], self.payoffs[t + 1, is_profitable] * discount_factor)

        # Step 4: Discount the payoffs
        option_price = np.mean(self.payoffs[1:] * discount_factor)

        return option_price
