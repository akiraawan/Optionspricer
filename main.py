from options import binomial_tree
from options import black_scholes_merton
from options import monte_carlo
from options import neural
from options.neural.supervised import supervised_european
from options.neural.unsupervised import unsupervised_european
import torch
import matplotlib.pyplot as plt


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
    option_type = 'call'
    S = 100.0  # Current price of the underlying asset
    K = 110.0  # Strike price of the option
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility of the underlying asset
    T = 365  # Time to maturity of the option specified in days
    q = 0.2  # Dividend yield of the underlying asset

    simulator = supervised_european.SimulatorEuropean(option_type=option_type, create_label=True)
    model = supervised_european.European_Sup_NN()
    train_loss = supervised_european.train(model, simulator)
    supnn = model(torch.tensor([[S, K, r, sigma, float(T)/365, q]], dtype=torch.float32))

    print("Supervised Neural Network Price:", supnn.item())

    # Plot the training loss
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()


def test_unsupervised_european_neural_network():
    option_type = 'call'
    S = 100.0  # Current price of the underlying asset
    K = 110.0  # Strike price of the option
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility of the underlying asset
    T = 365  # Time to maturity of the option specified in days
    q = 0.2  # Dividend yield of the underlying asset
    simulator = unsupervised_european.SimulatorEuropean(option_type=option_type)
    model = unsupervised_european.European_Unsup_NN()
    train_loss = unsupervised_european.train(model, simulator)
    unsupnn = model(torch.tensor([[S, K, r, sigma, T, q]], dtype=torch.float32))

    print("Unsupervised Neural Network Price:", unsupnn.item())

    # Plot the training loss
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()


def test_simulator():
    s = neural.SimulatorAmerican(create_label=True)
    print(s.df.head())


def main():
    # test_american_pricer()
    test_european_pricer()
    # test_visualise_and_display()
    # test_monte_carlo()
    test_supervised_european_neural_network()
    # model, train_loss = test_unsupervised_european_neural_network()
    

if __name__ == '__main__':
    # compare(is_european=False)
<<<<<<< HEAD
    test_american_pricer()
=======
    main()
>>>>>>> 24b436ec41064c894705002f5c50b7b5e6822789
