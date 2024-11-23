# options pricer

The `options` library consists of various implentations to calculate the price of American and European Options. This includes:
- Black-Scholes Formulae
- Binomial Tree 
- Monte Carlo
- Neural Network (Supervised and Unsupervised)

## Requirements

- Python 3.x
- NumPy 2.x
- Pytorch 2.4.x
- scipy 1.14.x

You can install the necessary dependencies using:
```bash
pip install torch numpy scipy
```

## Options

An Option is option is a derivative contract which conveys to its owner, the holder, the right, but not the obligation, to buy (call option) or sell (put option) a specific quantity of an underlying asset or instrument at a specified strike price on (European Options) or before (American Options) a specified date.

A European option is a type of option where the holder of an option has the option to exercise the option only at the time of maturity.

An American option is a type of option where the holder of an option has the option to exercise the option any time between time of purchase and time of maturity.

### Black-Scholes Equation

The Black-Scholes model (Geometric Brownian Motion) is a Stochastic Differential Equation that represents the evolution of a stock price given as:

$$ 
dS_t = \mu S_tdt + \sigma S_tdW_t
$$

where $W_t \sim N(0, t)$ is a Wiener process or Brownian motion, and 
$\mu$ (mean/drift) and $\sigma$ (volatility/diffusion) are constants.

This is also known as the log-normal distribution.

The Black-Scholes Equation is a Partial Differential Equation governing the price evolution of derivatives under the Black–Scholes model. Given as:


$$ 
\frac{\partial V}{\partial t} + \frac 12 \sigma^2 S^2 \frac{\partial^2V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0
$$

The initial value and boundary conditions for this PDE are unique to all derivatives, but for an European call option we have

$$
\begin{split}
 C(T, S_T) &= \max(S_T-K, 0) \text{ for all }  S_T \\
 C(t, 0) &= 0 \quad \text{for all } t \\
 C(t, S_t) &\sim S_t - Ke^{-r(T-t)} \quad \text{as } S_t \to \infty, \text{ for fixed } t
\end{split}
$$

where we have $T$ the maturity time, $C$ the price of the European call option, $S_t$ the underlying stock price at time $t$.

There is a closed form solution to this, famously known as the Black-Scholes formulae:

$$C(t, S_t) = S_t\Phi(d_1) - Ke^{-r(T-t)}\Phi(d_2)$$

where $K$ is the strike price. $d_1$ and $d_2$ are defined as follows:

$$
\begin{split}
  d_{1,2} &= \frac{\log(S/K) + (r \pm \tfrac 12 \sigma^2)(T-t)}{\sigma\sqrt{T-t}} \\
\end{split}
$$

and $\Phi$ is the CDF of the Standard Normal Distribution given as:

$$ \Phi(x) = \frac 1{\sqrt {2\pi}} \int_{-\infty}^x e^{-\tfrac 12 s^2}\text ds$$

For a European Put option we have the boundry conditions as:

$$
\begin{split}
 P(T, S_T) &= \max(K - S_T, 0) \text{ for all }  S_T \\
 P(t, 0) &= Ke^{-r(T-t)} \quad \text{for all } t \\
 P(t, S_t) &\sim 0 \quad \text{as } S_t \to \infty, \text{ for fixed } t
\end{split}
$$

An important result to price a European Put Option is Put-Call Parity. This is an equation that gives a relationship between the price of a call option and a put option.

$$
C - P = F
$$

Where $F_t = S_t - Ke^{-(T-t)}$ is the price of a forward contract at time $t$. Using this result, we can get the explicit Black-Scholes Formulae for European Put Options, given as:

$$
P(t, S_t) = Ke^{-r(T-t)}\Phi(-d_2) - S_t\Phi(-d_1)
$$

### Payoff

Before we go any further, let's quickly define the payoff of an option. 

The payoff of an option is the amount you make/lose based on your decision of exercising an option or not. 

Let's start with a European Call Option. This allows the holder to choose whether they want to buy or not for $K$ at maturity $T$. Intuitively, if the price of the underlying at maturity $S_T$ is more than the strike price $K$, then we would exercise the option and make $S_T-K$. Whereas, if the price of the underlying is less than the strike price, then we would not exercise the option and thus make nothing. So the payoff will be $0$. We can see that this can be simplified into a single equation:
$$
Y = (S_T - K)^+
$$

Where $x^+ := \max(0, x)$

For the European Put Option, this allows the holder to choose whether they want to sell or not for $K$ at maturity $T$. Using a similar argument as a call option, we can see that the payoff for a European Put Option is:

$$
Y = (K - S_T)^+
$$


It may also be useful that the price of a derivative at time $0$ is the expectation of the discounted payoff.

$$
V = \mathbb{E}^\mathbb{Q}[e^{-(r-q)T}Y]
$$

## Black-Scholes-Merton

`options/black_scholes_merton` calculates the price of a European Option using the Black-Scholes Formulae defined above:

$$
C(t, S_t) = S_t\Phi(d_1) - Ke^{-r(T-t)}\Phi(d_2)
$$

for European Call Options and,


$$
P(t, S_t) = Ke^{-r(T-t)}\Phi(-d_2) - S_t\Phi(-d_1)
$$

for European Put Options.

## Binomial Tree 

`options/binomial_tree` prices European and American Options using the binomial tree model. 

Option valuation using this method is, as described, a three-step process:

1.  Price tree generation.
2. Calculation of option value using the payoff at each final node.
3. Sequential calculation of the option value at each preceding node.

### Step 1: Create the Binomial Price tree of the underlying

The tree of prices is produced by working forward from valuation date to expiration. At each step, it is assumed that the underlying instrument will move up or down by a specific factor (u or d) per step. So, if 
S is the current price, then in the next period the price will either be $S_{up} = S\cdot u$ or $S_{down} = S \cdot d$. 

The up and down factors are calculated using the underlying volatility, $\sigma$ and the time duration of a step, $t$ measured in years (using the day count convention of the underlying instrument). From the condition that the variance of the log of the price is $\sigma^2t$, we have

$$
\begin{split}
u &= e^{\sigma \sqrt{t}} \\
d &= e^{\sigma \sqrt{t}} = 1/u
\end{split}
$$

where the probability of the stock going up is defined as

$$
p = \frac{e^{(r-q)\Delta t} - d}{u - d}
$$

where $q$ is the dividend yield, $r$ is the risk-free rate, and probability going down is $1 - p$

### Step 2: Find option value at each final node

For every Stock price at the final node (at maturity), write the option price at maturity as the payoff of the maturity. 

$$
C(T, S_T) = \max(S_T-K, 0)
$$

for a call option and 

$$
P(T, S_T) = \max(K - S_T, 0)
$$

for a put option.

### Step 3: Find option value at earlier nodes

This is where the difference in European and American options become important. 

For European options, we cannot exercise the option before the maturity, so starting from the second last node, we calculate the expected discounted value of the option. 

$$
\text{Binomial Value} = e^{-(r-q)\Delta t}\left( p * \text{Option up} + (1-p)*\text{Option down} \right)
$$

For American options, we can exercise the option before the maturity, so at every earlier node, we need to also see if the payoff at that node is less or greater than the binomial value. So starting from the second last node, we ca calculate

$$
\text{Binomial Value} = \max(e^{-(r-q)\Delta t}\left( p * \text{Option up} + (1-p)*\text{Option down} \right), Y)
$$

where Y is the payoff for a call/put option. 

## Monte Carlo Simulation

`options/monte_carlo` calculates the price of an option using a monte carlo simulation approach. 

Option valuation using this method is, as described, a three-step process:

1. Generate stock price simulations.
2. calcualte the payoffs at the maturity for every simulation and discount using the risk-free rate.
3. Find the mean (expectation) of the discounted payoffs.

### Step 1:

Similar to the Bionomial Tree model, we are simulating the evolution of the stock price from time $0$ to maturity $T$. However instead of modeling the price as a binomial tree, we take advantage of the fact that the stock price under the Black-Scholes model is given as:

$$
dS_t = \mu S_tdt + \sigma S_tdW_t
$$

by solving this Stochastic Differential Equation, we get an explicit form for $S_t$

$$
S_t = S_0\exp\left( \left( r - \frac{\sigma^2}{2} \right)t \, + \sigma \sqrt{t}N(0,1)\right)
$$

or in our case, we want:

$$
S_{t + \Delta t} = S_t\exp\left( \left( (r-q) - \frac{\sigma^2}{2} \right)\Delta t \, + \sigma\sqrt{\Delta t}N(0, 1) \right)
$$

So the only randomness in the equation is the Normal Random Variable $N(0, 1)$.

Using the explicit formulae above, we can construct $\text{n\_sims}$ number of simulations, each with $\text{n\_steps}$ number of steps. 

### Step 2: Calculate the discounted payoffs

Now for each simulation, on the last stock price of the simulatiom $S_T$, first calculate the payoffs using Y.

Then for each payoff, discount them to the present value using $e^{-(r-q)T}$. Overall, for each simulation, find:

$$
e^{-(r-q)T}Y
$$

### Step 3: Find the Expectation of the discounted payoff

Finally, to find the expectation of the discounted payoff, we can simply find the arithmetic mean of all $\text{n\_sims}$ discounted payoffs. This results in an approximate value for the price of an option:

$$
V = \mathbb{E}^\mathbb{Q}[e^{-(r-q)T}Y] 
$$

## Neural Network

`options/neural` prices options using neural networks, implemented in `pytorch`. 

### Data

The most important thing with the neural network approach is the data. We had two approaches:

1. Collect data from real trading data using yfinance and use the price of the option traded in the market as the 'real' values (targets).

2. Simualte our own data. Where the labels are gathererd by using my previously implemented models. 

In the end we went with simulating our own data. This is because yfinance is restrictive with the number of samples we can get. Also pricing options requires being able to implement the boundary conditions. However if we used real traded data, we would not have been able to get trades with (for example) the underlying price of $0$.

### Simulator

The simulator to simulate the data is in `options/neural/simulator`. It consists of the base class `Simulator` and two subclasses `SimulatorAmerican` and `SimulatorEuropean`. 

The constructor of `Simulator` base class takes tuple parameters of every options variable and these are the ranges for the variable that we want to randomly simulate. 

The subclasses inherit the base class `Simulator`. The only difference between `SimulatorAmerican` and `Simulatoreuropean` is that the labels for the supervised learning model uses either the European options pricer or the American options pricer. 

### Imports
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from options.neural.simulator import SimulatorEuropean
```

### Neural Network Architecture

```python
class European_Sup_NN(nn.Module):
    def __init__(self):
        super(European_Sup_NN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(6, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, inputs):
        outputs = self.seq(inputs).squeeze(1)
        return outputs
```

This is a very general neural network architecture where $\tanh$ is used as an activation function. The reason for this is to keep the model differentiable. 

### Dataset

For a supervised model:

```python
class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input = torch.tensor(self.df.iloc[idx][['S', 'K', 'r', 'sigma', 'T', 'q']].values, dtype=torch.float32)
        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float32)
        return input, label
```
This is a custom dataset, used to customise how your data is fed into the model during training. Here, every iteration of the `CustomDataset` returns a tensor object of the input data and also a tensor object of the label.

For a unsupervised model:

```python
class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        S = torch.tensor(self.df.iloc[idx]['S'], dtype=torch.float32, requires_grad=True)
        K = torch.tensor(self.df.iloc[idx]['K'], dtype=torch.float32, requires_grad=True)
        r = torch.tensor(self.df.iloc[idx]['r'], dtype=torch.float32, requires_grad=True)
        sigma = torch.tensor(self.df.iloc[idx]['sigma'], dtype=torch.float32, requires_grad=True)
        T = torch.tensor(self.df.iloc[idx]['T'], dtype=torch.float32, requires_grad=True)
        q = torch.tensor(self.df.iloc[idx]['q'], dtype=torch.float32, requires_grad=True)

        return S, K, r, sigma, T, q
```

In the unsupervised model, each input variable is outputed separately. The reason for this is the design of the loss function. For the unsupervised model, each variable is used, and thus the dataset returns each separetely. 

### Loss Function

For the supervised model we have chosen the loss function `F.mse_loss` mean square error. 

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

where $y_i$ is the observed value and $\hat{y}_i$ is the predicted value. 

The reason for is that the problem is simply a single label regression problem. And so the mean square error is a reasonable choice of loss function. 

For the unsupervised model, we cannot use the mean square error as we do not have the label. Instead, we gotta use the fact that the derivatives price $V$ before maturity must follow the Black-Scholes Equation

$$
\frac{\partial V}{\partial t} + \frac 12 \sigma^2 S^2 \frac{\partial^2V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0
$$

and for call options, the boundary conditions

$$
\begin{split}
 C(T, S_T) &= \max(S_T-K, 0) \text{ for all }  S_T \\
 C(t, 0) &= 0 \quad \text{for all } t \\
 C(t, S_t) &\sim S_t - Ke^{-r(T-t)} \quad \text{as } S_t \to \infty, \text{ for fixed } t
\end{split}
$$

must be satisfied. 

So we design the loss function such that these requirements are met as close as possible. We will define the loss function for the PDE as `L_PDE` and the loss function for the boundary conditions as `L_BC`.

In practice, instead of implementing all three boundary conditions separately, we can combine them into one loss function $(S_t-Ke^{-r(T-t)})^+$ for European Call options and $(Ke^{-r(T-t)} - S_t)^+$ for European Put options for all $t$. Using this we can define the loss functions

$$
L_{PDE} = \frac{\partial V}{\partial t} + \frac 12 \sigma^2 S^2 \frac{\partial^2V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV
$$
for all $t < T$

$$
L_{BC} = V - (S_t-Ke^{-r(T-t)})^+
$$

for call options, and

$$
L_{BC} = V - (Ke^{-r(T-t)}-S_t)^+
$$

for put options for all $t \le T$


```python
def loss_fn(V, S, K, r, sigma, T, q, type='call'):
    L_PDE = 0.0
    L_BC = 0.0

    dVdT = torch.autograd.grad(V, T, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dVdS = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    d2VdS2 = torch.autograd.grad(dVdS, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    indices_Tg0 = np.where(T > 0)[0]

    S_Tg0 = S[indices_Tg0]
    K_Tg0 = K[indices_Tg0]
    r_Tg0 = r[indices_Tg0]
    sigma_Tg0 = sigma[indices_Tg0]
    q_Tg0 = q[indices_Tg0]
    dVdT_Tg0 = dVdT[indices_Tg0]
    dVdS_Tg0 = dVdS[indices_Tg0]
    d2VdS2_Tg0 = d2VdS2[indices_Tg0]
    V_Tg0 = V[indices_Tg0]

    L_PDE += torch.mean(torch.square(-dVdT_Tg0 + (r_Tg0 - q_Tg0) * S_Tg0 * dVdS_Tg0 + 0.5 * sigma_Tg0 ** 2 * S_Tg0 ** 2 * d2VdS2_Tg0 - r_Tg0 * V_Tg0))

    if type == 'call':
        L_BC += torch.mean(torch.square(V - torch.max(S - K * torch.exp(-r * T), torch.zeros_like(S))))
    else:
        L_BC += torch.mean(torch.square(V - torch.max(K * torch.exp(-r * T) - S, torch.zeros_like(S))))

    return L_PDE + L_BC
```

in this implementation, we have defined `T` as the time TO maturity. So `T` is equivalent to $\tau = T-t$ mathematically. 