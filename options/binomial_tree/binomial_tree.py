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
        def encode(i, j):
            return f"[{i} {j}]"

        def label(i, j):
            return f"{round(self.stock_tree[i][j], 2)}\n{round(self.option_tree[i][j], 2)}"

        G = nx.DiGraph()
        n = self.steps + 1
        for i in range(n):
            for j in range(i, n):
                G.add_node(encode(i, j), label=label(i, j))

        for i in range(n-1):
            for j in range(i, n-1):
                G.add_edge(encode(i, j), encode(i, j+1))
                G.add_edge(encode(i, j), encode(i+1, j+1))
                
        pos = dict()
        ver_gap = 0.4
        hor_gap = 1.0

        for i in range(n):
            level_nodes = [encode(i, j) for j in range(i, n)]
            for idx, node in enumerate(level_nodes):
                pos[node] = (idx * hor_gap - (len(level_nodes) - 1) * hor_gap, -i * ver_gap)

        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue', font_size=10, arrowstyle="-|>", arrowsize=20)

        plt.show()
