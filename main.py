from options import binomial_tree
from options import black_scholes_merton
from options import monte_carlo
from options import neural
from options.neural.supervised import supervised_european
from options.neural.unsupervised import unsupervised_european


def test_american_pricer():
    S = 503.0  # Current price of the underlying asset
    K = 450.0  # Strike price of the option
    r = 0.035  # Risk-free interest rate
    sigma = 0.22738  # Volatility of the underlying asset
    T = 420  # Time to maturity of the option specified in days
    steps = 10000
    q = 0.2  # Dividend yield of the underlying asset
    bt = binomial_tree.BinomialTreeAmerican(S, K, r, T, sigma, steps, option_type='put')
    print(bt.price)


def test_european_pricer():
    # Option parameters
    S = 100.0  # Current price of the underlying asset
    K = 110.0  # Strike price of the option
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility of the underlying asset
    T = 365  # Time to maturity of the option specified in days
    steps = 1000
    q = 0.2

    BS_price = black_scholes_merton.BSMFormulae(S, K, r, T, sigma, q, option_type='call')
    print(BS_price.price)

    bt = binomial_tree.BinomialTreeEuropean(S, K, r, T, sigma, steps, q, option_type='call')
    print(bt.price)


def test_visualise_and_display():
    S = 100.0  # Current price of the underlying asset
    K = 110.0  # Strike price of the option
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility of the underlying asset
    T = 365  # Time to maturity of the option specified in days
    steps = 5
    q = 0.2  # Dividend yield of the underlying asset
    bt = binomial_tree.BinomialTreeAmerican(S, K, r, T, sigma, steps, q, option_type='call')
    print("Price:", bt.price)
    bt.display()
    bt.visualise()


def test_monte_carlo():
    S = 100.0  # Current price of the underlying asset
    K = 110.0  # Strike price of the option
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility of the underlying asset
    T = 365  # Time to maturity of the option specified in days
    steps = 5
    sims = 100000
    q = 0.2  # Dividend yield of the underlying asset

    monte_eur = monte_carlo.MonteCarloEuropean(S, K, r, T, sigma, steps, sims, q)
    monte_ame = monte_carlo.MonteCarloAmerican(S, K, r, T, sigma, steps, sims, q)
    print(monte_eur.price)
    print(monte_ame.price)

def test_supervised_european_neural_network():
    simulator = supervised_european.SimulatorEuropean(create_label=True)
    model = supervised_european.European_Sup_NN()
    train_loss = supervised_european.train(model, simulator)
    return model, train_loss

def test_unsupervised_european_neural_network():
    simulator = unsupervised_european.SimulatorEuropean()
    model = unsupervised_european.European_Unsup_NN()
    train_loss = unsupervised_european.train(model, simulator)
    return model, train_loss

def compare(is_european):
    S = 100.0  # Current price of the underlying asset
    K = 110.0  # Strike price of the option
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility of the underlying asset
    T = 365  # Time to maturity of the option specified in days
    steps = 100
    sims = 100000
    q = 0.2  # Dividend yield of the underlying asset

    if is_european:
        bt = binomial_tree.BinomialTreeEuropean(S, K, r, T, sigma, steps, q, option_type='call')
        monte = monte_carlo.MonteCarloEuropean(S, K, r, T, sigma, steps, sims, q)
    else:
        bt = binomial_tree.BinomialTreeAmerican(S, K, r, T, sigma, steps, q, option_type='call')
        monte = monte_carlo.MonteCarloAmerican(S, K, r, T, sigma, steps, sims, q)

    print(bt.price)
    print(monte.price)


def test_simulator():
    s = neural.SimulatorAmerican(create_label=True)
    print(s.df.head())


def main():
    test_american_pricer()
    # test_european_pricer()
    # test_visualise_and_display()
    # test_monte_carlo()
    # test_supervised_european_neural_network()
    # model, train_loss = test_unsupervised_european_neural_network()
    

if __name__ == '__main__':
    # compare(is_european=False)
    main()
