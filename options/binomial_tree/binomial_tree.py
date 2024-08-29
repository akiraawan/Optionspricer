from abc import abstractmethod
import numpy as np


class BinomialTree:
    def __init__(self, S0, K, r, T, sigma, steps, q=0, option_type='call'):
        """
        Construct a Binomial Tree for option pricing.

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
        q: float
            dividend yield of the underlying asset
        """
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

        self.generate_trees()
        self.price = self._price()

    def _price(self):
        return self.option_tree[0, 0]

    @abstractmethod
    def generate_trees(self):
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
        """
        Display the option tree and stock tree in the console.
        """
        print("Option Tree:")
        self.__display_tree(self.option_tree)

        print("Stock Tree:")
        self.__display_tree(self.stock_tree)

    @staticmethod
    def __display_tree(tree):
        for i in range(len(tree)):
            print("\t"*i, end="")
            print("\t".join(["{0:.2f}".format(i) for i in tree[i][i:]]))

    def visualise(self):
        """
        Prints out the graphical representation of the binomial tree.
        """
        pass
