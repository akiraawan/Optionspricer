# optionspricer

Define $V$ the derivative price, $S$ the underlying price, $\sigma$ the volatility and $r$ the risk-free rate (e.g. interest), we have the Black-Scholes Equation

$$ \frac{\partial V}{\partial t} + \frac 12 \sigma^2 S^2 \frac{\partial^2V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$

The initial value and boundary conditions for this PDE are unique to all derivatives, but for an European call option we have

$$
\begin{split}
 C(T, S) &= \max(S-E, 0) \text{ for all }  S \\
 C(t, 0) &= 0 \quad \text{for all } t \\
 C(t, S) &\sim S \quad \text{as } S \to \infty, \text{ for fixed } t
\end{split}
$$

where we have $T$ the maturity time, $C$ the price of the European call option, $S$ the underlying stock price at time $t$.

There is a closed form solution to this, famously known as the Black-Scholes formulae:

$$C(S, t) = SN(d_1) - Ke^{-r(T-t)}N(d_2)$$

where $K$ is the strike price. $d_1$ and $d_2$ are defined as follows:

$$
\begin{split}
  d_1 &= \frac{\log(S/K) + (r +\tfrac 12 \sigma^2)(T-t)}{\sigma\sqrt{T-t}} \\
  d_2 &= \frac{\log(S/K) + (r - \tfrac 12 \sigma^2)(T-t)}{\sigma\sqrt{T-t}}
\end{split}
$$

and we have the cumulative distribution function

$$ N(x) = \frac 1{\sqrt {2\pi}} \int_{-\infty}^x e^{-\tfrac 12 s^2}\;\text ds$$
