from options import american_pricer, european_pricer

def test_american_pricer():
    S = 100.0  # Current price of the underlying asset
    K = 110.0  # Strike price of the option
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility of the underlying asset
    T = 365  # Time to maturity of the option specified in days
    steps = 1000
    q = 0.2  # Dividend yield of the underlying asset
    bt = american_pricer.BinomialTreeAmerican(S, K, r, T, sigma, steps, q, option_type='call')
    print(bt.price)

if __name__ == '__main__':
    test_american_pricer()