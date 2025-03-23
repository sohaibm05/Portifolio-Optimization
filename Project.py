# All the libraries + API's needed for the program
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Name of the asset classes (stocks) used for the program
assets = ['META', 'AMZN', 'AAPL', 'NFLX', 'GOOG']

# Assign weights to the stocks
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Get the stock/portfolio starting date
stockStartDate = '2013-01-01'

# Get the stock/portfolio ending date
today = datetime.today().strftime('%Y-%m-%d')

# Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()

# Store the adjusted close price of the stock into the df
for stock in assets:
  df[stock] = yf.download(stock, start=stockStartDate, end=today)['Close']
#  print(df)

title = 'Portfolio Adj. Close Price History'
my_stocks = df
for c in my_stocks.columns.values:
  plt.plot(my_stocks[c], label = c)

# Plotting the table
plt.title(title)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Adj. Price USD ($)', fontsize = 18)
plt.legend(my_stocks.columns.values, loc= 'upper left')
#  plt.show()

#  Show the daily simple return
# returns = df.pct_change()
#   print(returns)



# Fenwick tree implementaton
def fenwick_update(bit, size, index, value):
    # Updates the Fenwick Tree by adding value at the given index. (how a fenwick tree updates the query)
    while index <= size:
        bit[index] += value
        index += index & -index  # Move to parent

def fenwick_query(bit, index):
    # """Computes the cumulative sum from 1 to index.""" (math aspect)
    total = 0
    while index > 0:
        total += bit[index]
        index -= index & -index  # Move to previous index
    return total



returns = df.pct_change().fillna(0)  # Ensure no NaNs in the table

# Convert DataFrame to a 1D array of total daily returns (sum of all stock returns)
daily_returns = returns.sum(axis=1).values  # Sum across all stocks

# Initialize Fenwick Tree with 0s
size = len(daily_returns)
bit = [0] * (size + 1)  # 1-based index

# Build Fenwick Tree with daily return data
for i, ret in enumerate(daily_returns, 1):  # 1-based indexing
    fenwick_update(bit, size, i, ret)



# Create and show the annualized covariance matrix
cov_matrix_annual = returns.cov() * 252
#   print(cov_matrix_annual)

# Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
#   print(port_variance)

# Calculate the portfolio volatility aka standard deviation
port_volitility = np.sqrt(port_variance)
#   print(port_volatility)

# Calculate the annual portfolio return
portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252
#   print(portfolioSimpleAnnualReturn)

# Show the expected annual return, volatility (risk), and variance

percent_var = str(round(port_variance, 2) * 100) + ' %'
percent_vols = str(round(port_volitility, 2) * 100) + ' %'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + ' %'

# print('Expected annual return', percent_ret)
# print('Annual volatility / risk', percent_vols)
# print('Annual Variance', percent_var)

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Portfolio Optimization !

# Calculate the expected returns and the annualized sample covariance matrix of asset returns
mew = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for max sharpe ratio (a way to describe how much excess return you recieve based on volatitlity)

ef = EfficientFrontier(mew, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
# print(cleaned_weights)
# print(ef.portfolio_performance(verbose = True))

# Get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = 15000)

allocation, leftover = da.lp_portfolio()
print('Discrete allocation: ', allocation)
print('Funds remaining:  ${:.2f}'.format(leftover))


