import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm


class BlackScholes__Pricing_Simulations:
    
    
            
    def __init__(self, call, stock, strike, maturity, interest, volatility, dividend, Npath, mu):
        self.call = call
        self.stock = stock
        self.strike = strike
        self.maturity = maturity
        self.interest = interest
        self.volatility = volatility
        self.dividend = dividend
        self.dt = 1/360
        self.Npath = Npath
        self.mu = mu
        self.d1 = (self.volatility * np.sqrt(self.maturity)) ** (-1) * (
                np.log(self.stock / self.strike) + (
                    self.interest - self.dividend + self.volatility ** 2 / 2) * self.maturity)
        self.d2 = self.d1 - self.volatility * np.sqrt(self.maturity)

    def price(self):
        if self.call:
            return np.exp(-self.dividend * self.maturity) * norm.cdf(self.d1) * self.stock - norm.cdf(self.d2) * self.strike * np.exp(-self.interest * self.maturity)
        else:
            return norm.cdf(-self.d2) * self.strike * np.exp(-self.interest * self.maturity) - norm.cdf(-self.d1) * self.stock * np.exp(-self.dividend * self.maturity)

    def delta(self):
        if self.call:
            return norm.cdf(self.d1) * np.exp(-self.dividend * self.maturity)
        else:
            return (norm.cdf(self.d1) - 1) * np.exp(-self.dividend * self.maturity)

    def gamma(self):
        return np.exp(-self.dividend * self.maturity) * norm.pdf(self.d1) / (
                self.stock * self.volatility * np.sqrt(self.maturity))

    def vega(self):
        return self.stock * norm.pdf(self.d1) * np.sqrt(self.maturity) * np.exp(-self.dividend * self.maturity)

    def theta(self):
        if self.call:
            return -np.exp(-self.dividend * self.maturity) * (self.stock * norm.pdf(self.d1) * self.volatility) / (
                    2 * np.sqrt(self.maturity)) - self.interest * self.strike * np.exp(
                -self.interest * np.sqrt(self.maturity)) * norm.cdf(self.d2) + self.dividend * self.stock * np.exp(-self.dividend * self.maturity) * norm.cdf(self.d1)
        else:
            return -np.exp(-self.dividend * self.maturity) * (self.stock * norm.pdf(self.d1) * self.volatility) / (
                    2 * np.sqrt(self.maturity)) + self.interest * self.strike * np.exp(
                -self.interest * np.sqrt(self.maturity)) * norm.cdf(-self.d2) - self.dividend * self.stock * np.exp(
                -self.dividend * self.maturity) * norm.cdf(-self.d1)

    def rho(self):
        if self.call:
            return self.strike * self.maturity * np.exp(-self.interest * self.maturity) * norm.cdf(self.d2)
        else:
            return -self.strike * self.maturity * np.exp(-self.interest * self.maturity) * norm.cdf(-self.d2)
    
    def monte_carlo_bs(self):
    
        St_P = [self.stock] * self.Npath
        St_Q = [self.stock] * self.Npath

        
        for t in range(0, int(self.tau/self.dt)):
            rand = np.random.normal(0, 1, [1, self.Npath])

            St_P *= ( np.exp ( self.dt * ( self.mu + self.interest - 0.5 * self.volatility ** 2) + self.volatility * np.sqrt(self.dt)* rand))
            St_Q *= ( np.exp ( self.dt * ( self.mu - 0.5 * self.volatility ** 2) + self.volatility * np.sqrt(self.dt)* rand))

        return {"Call Price under P measure": np.exp(-1.0 * (self.mu + self.interest) * self.tau) * np.maximum(St_P - K, 0).mean(), "Call Price under Q measure": np.exp(-1.0 * self.interest * self.tau)*np.maximum(St_Q - K, 0).mean()} if call else {"Put Price under P": np.exp(-1.0 * (self.mu + self.interest) * self.tau) * np.maximum(K - St_P, 0).mean(), "Put Pricing under Q": np.exp(-1.0 * self.interest * self.tau) * np.maximum(K - St_Q, 0).mean()}
    
    def info(self):
        print ("""Calculate the option price, delta, gamma, vega, theta and rho according to Black Scholes Merton model.\n

Attributions\n
============\n
call: True if the option is call, False if it is put (boolean)\n
stock: Price of the underlying security (float)\n
strike: Strike price of the option (float)\n
maturity: Time to maturity of the option in years (float)\n
interest: Annual interest rate expressed as decimal (float)\n
volatility: Annual volatility expressed as decimal (float)\n
dividend: Annual dividend yield expressed as decimal (float)\n

Methods\n
=======\n
price: Returns the price of the option according to Black Scholes Merton.\n
delta: Returns the delta of the option\n
gamma: Returns the gamma of the option\n
vega: Returns the vega of the option\n
theta: Returns the theta of the option\n
rho: Returns the rho of the option\n
monte_carlo_bs: Calculate the call/put price under both P and Q measure by simulating npaths at each\n
\t\ttime steps along the path. Returns the call/put option. \n
\t\tArgument: call = 1, put = 0, maturity = the fraction of the year""")
        
 
