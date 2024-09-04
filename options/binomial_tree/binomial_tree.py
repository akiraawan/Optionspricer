from abc import abstractmethod
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
        self.price = self.get_price()

    def get_price(self):
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
        def display_tree(tree):
            for i in range(len(tree)):
                print("\t"*i, end="")
                print("\t".join(["{0:.2f}".format(i) for i in tree[i][i:]]))

        print("Stock Tree:")
        display_tree(self.stock_tree)

        print("Option Tree:")
        display_tree(self.option_tree)

    def visualise(self):
        """
        Prints out the graphical representation of the binomial tree.
        """
        G = nx.DiGraph()
        n = self.stock_tree.shape[0]

        for i in range(n):
            for j in range(i, n):
                G.add_node((i, j))

        for i in range(n-1):
            for j in range(i, n-1):
                G.add_edge((i, j), (i, j+1))
                G.add_edge((i, j), (i+1, j+1))

        pos = {}
        layer_gap = 1.5
        node_gap = 2.0

        for i in range(n):
            for j in range(i, n):
                pos[(i, j)] = (j * node_gap - i * node_gap / 2, -i * layer_gap)

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=False, node_size=2000, node_color='lightblue',
                font_size=10, font_color='black', arrowstyle='-|>', arrowsize=20)

        for (i, j) in G.nodes():
            x, y = pos[(i, j)]
            plt.text(x, y + 0.2, f"{self.stock_tree[i, j]:.2f}", fontsize=10, ha='center', va='center', color='red')
            plt.text(x, y - 0.2, f"{self.option_tree[i, j]:.2f}", fontsize=10, ha='center', va='center', color='blue')

        red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Stock Value', markerfacecolor='red', markersize=10)
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Option Value', markerfacecolor='blue', markersize=10)
        plt.legend(handles=[red_patch, blue_patch], loc='lower right')

        plt.show()
