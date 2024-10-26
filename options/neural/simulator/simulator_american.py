from .simulator import Simulator
from options.binomial_tree import BinomialTreeAmerican
import numpy as np


class SimulatorAmerican(Simulator):
    def generate_labels(self, S, K, sigma, T, r, q):
        labels = np.zeros(self.num_samples)
        for i in range(self.num_samples):
            bt = BinomialTreeAmerican(S0=S[i], K=K[i], sigma=sigma[i], T=T[i], r=r[i], q=q[i], steps=20)
            labels[i] = bt.price
        return labels
