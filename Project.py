# All the libraries + API's needed for the program
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Name of the asset classes (stocks) used for the program
assets = ['META', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
print("Selected assets:", assets)

# Assign weights to the stocks
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
print("Initial portfolio weights:", weights)

# Get the stock/portfolio starting date
stockStartDate = '2013-01-01'

# Get the stock/portfolio ending date
today = datetime.today().strftime('%Y-%m-%d')
print("Date range:", stockStartDate, "to", today)

# Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()

# Store the adjusted close price of the stock into the df
print("\nDownloading historical stock data...")
for stock in assets:
    df[stock] = yf.download(stock, start=stockStartDate, end=today)['Close']
    print(f"Downloaded data for {stock}")

# Show first few rows of the data
print("\nSample stock price data:")
print(df.head())

# Plotting the table
title = 'Portfolio Adj. Close Price History'
my_stocks = df
for c in my_stocks.columns.values:
    plt.plot(my_stocks[c], label=c)

plt.title(title)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adj. Price USD ($)', fontsize=18)
plt.legend(my_stocks.columns.values, loc='upper left')
# plt.show()

# Fenwick tree implementaton
def fenwick_update(bit, size, index, value):
    while index <= size:
        bit[index] += value
        index += index & -index

def fenwick_query(bit, index):
    total = 0
    while index > 0:
        total += bit[index]
        index -= index & -index
    return total

# Daily returns
returns = df.pct_change().fillna(0)
print("\nSample daily returns (percentage change):")
print(returns.head())

# Convert DataFrame to a 1D array of total daily returns
daily_returns = returns.sum(axis=1).values
print("\nTotal daily portfolio return sample:")
print(daily_returns[:5])

# Initialize Fenwick Tree
size = len(daily_returns)
bit = [0] * (size + 1)

# Build Fenwick Tree with daily return data
print("\nBuilding Fenwick Tree...")
for i, ret in enumerate(daily_returns, 1):
    fenwick_update(bit, size, i, ret)

print("Fenwick Tree built successfully (first 10 elements):")
print(bit[:10])

# Annualized covariance matrix
cov_matrix_annual = returns.cov() * 252
print("\nAnnualized Covariance Matrix:")
print(cov_matrix_annual)

# Portfolio risk calculations
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
port_volitility = np.sqrt(port_variance)
portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252

percent_var = str(round(port_variance, 2) * 100) + ' %'
percent_vols = str(round(port_volitility, 2) * 100) + ' %'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + ' %'

print("\nPortfolio Metrics:")
print("Expected annual return:", percent_ret)
print("Annual volatility / risk:", percent_vols)
print("Annual variance:", percent_var)

# Portfolio Optimization using PyPortfolioOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

mew = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

print("\nExpected returns from historical data:")
print(mew)
print("\nSample covariance matrix:")
print(S)

ef = EfficientFrontier(mew, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("\nOptimized portfolio weights (Max Sharpe Ratio):")
print(cleaned_weights)

# Discrete Allocation
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=15000)

allocation, leftover = da.lp_portfolio()

print("\nDiscrete share allocation for $15,000 investment:")
print('Discrete allocation:', allocation)
print('Funds remaining:  ${:.2f}'.format(leftover))
