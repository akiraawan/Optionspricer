from options import binomial_tree 
from options import black_scholes_merton

def test_american_pricer():
    S = 100.0  # Current price of the underlying asset
    K = 110.0  # Strike price of the option
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility of the underlying asset
    T = 365  # Time to maturity of the option specified in days
    steps = 1000
    q = 0.2  # Dividend yield of the underlying asset
    bt = binomial_tree.BinomialTreeAmerican(S, K, r, T, sigma, steps, q, option_type='call')
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


if __name__ == '__main__':
    test_american_pricer()
    test_european_pricer()