import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Simulator:
    def __init__(self, option_type='call', initial_stock_price_range=(0, 150), strike_price_range=(50, 150), volatility_range=(0.1, 0.5), maturity_range=(0, 2), risk_free_rate_range=(0.01, 0.1), dividend_yield_range=(0, 0.1), num_samples=1000, create_label=False):
        self.option_type = option_type
        self.initial_stock_price_range = initial_stock_price_range
        self.strike_price_range = strike_price_range
        self.volatility_range = volatility_range
        self.maturity_range = maturity_range
        self.risk_free_rate_range = risk_free_rate_range
        self.dividend_yield_range = dividend_yield_range
        self.num_samples = num_samples
        self.df = self.simulate(include_labels=create_label)

    def simulate(self, include_labels: bool):
        S = self._uniform(self.initial_stock_price_range, self.num_samples)
        K = self._uniform(self.strike_price_range, self.num_samples)
        sigma = self._uniform(self.volatility_range, self.num_samples)
        T = self._uniform(self.maturity_range, self.num_samples)
        r = self._uniform(self.risk_free_rate_range, self.num_samples)
        q = self._uniform(self.dividend_yield_range, self.num_samples)

        data = {
            'S': S,
            'K': K,
            'sigma': sigma,
            'T': T,
            'r': r,
            'q': q,
        }

        if include_labels:
            label = self.generate_labels(S, K, sigma, T, r, q)
            data['label'] = label

        df = pd.DataFrame(data)
        return df

    @staticmethod
    def _uniform(tuple, num_samples):
        return np.random.uniform(tuple[0], tuple[1], num_samples)

    @staticmethod
    def _normal(mean, std, num_samples):
        return np.random.normal(mean, std, num_samples)

    def generate_labels(self, S, K, sigma, T, r, q):
        raise NotImplementedError(f"Not implemented in abstract class {type(self)}")
    
    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.df, test_size=test_size, random_state=random_state)
