import numpy as np
from .binomial_tree import BinomialTree


class BinomialTreeAmerican(BinomialTree):
    def generate_trees(self):
        self.stock_tree = np.zeros((self.steps+1, self.steps+1))
        for j in range(self.steps+1):
            for i in range(j+1):
                self.stock_tree[i, j] = self.S0 * (self.u ** (j-i)) * (self.d ** i)

        self.option_tree = np.zeros((self.steps+1, self.steps+1))
        for i in range(self.steps+1):
            self.option_tree[i, self.steps] = max(0, self.stock_tree[i, self.steps] - self.K) \
                if (self.option_type == 'call') \
                else max(0, self.K - self.stock_tree[i, self.steps])

        for j in range(self.steps-1, -1, -1):
            for i in range(j+1):
                if (self.option_type == 'call'):
                    self.option_tree[i, j] = max(self.stock_tree[i, j] - self.K, np.exp(-(self.r-self.q)*self.dt) * (self.p * self.option_tree[i, j+1] + (1-self.p) * self.option_tree[i+1, j+1]))
                else:
                    self.option_tree[i, j] = max(self.K - self.stock_tree[i, j], np.exp(-(self.r-self.q)*self.dt) * (self.p * self.option_tree[i, j+1] + (1-self.p) * self.option_tree[i+1, j+1]))
