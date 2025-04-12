import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox
import requests

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

plt.style.use('fivethirtyeight')


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


import yfinance as yf

import requests
from tkinter import messagebox

def search_ticker():
    query = search_entry.get().strip()

    if not query:
        messagebox.showwarning("Input Error", "Please enter a company name or ticker.")
        return

    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data.get('quotes'):
            messagebox.showerror("Search Error", f"No ticker found for '{query}'")
            return

        suggestions_listbox.delete(0, tk.END)
        for item in data['quotes']:
            symbol = item.get('symbol', 'N/A')
            name = item.get('shortname', 'N/A')
            suggestions_listbox.insert(tk.END, f"{symbol} - {name}")

    except Exception as e:
        messagebox.showerror("Search Error", f"Could not complete search for '{query}'\n{str(e)}")





def add_selected_ticker():
    selected = suggestions_listbox.get(tk.ACTIVE)
    if selected:
        symbol = selected.split(' - ')[0]
        current_text = stock_entry.get()
        new_text = f"{current_text},{symbol}" if current_text else symbol
        stock_entry.delete(0, tk.END)
        stock_entry.insert(0, new_text)


def analyze_portfolio():
    try:
        assets = stock_entry.get().replace(" ", "").split(",")
        stockStartDate = start_date_entry.get()
        today = datetime.today().strftime('%Y-%m-%d')

        df = pd.DataFrame()
        for stock in assets:
            df[stock] = yf.download(stock, start=stockStartDate, end=today)['Close']

        weights = np.array([1/len(assets)] * len(assets))

        returns = df.pct_change().fillna(0)
        daily_returns = returns.sum(axis=1).values

        size = len(daily_returns)
        bit = [0] * (size + 1)
        for i, ret in enumerate(daily_returns, 1):
            fenwick_update(bit, size, i, ret)

        cov_matrix_annual = returns.cov() * 252
        port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
        port_volitility = np.sqrt(port_variance)
        portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252

        percent_var = str(round(port_variance, 2) * 100) + ' %'
        percent_vols = str(round(port_volitility, 2) * 100) + ' %'
        percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + ' %'

        mew = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)
        ef = EfficientFrontier(mew, S)
        opt_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        latest_prices = get_latest_prices(df)
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=15000)
        allocation, leftover = da.lp_portfolio()

        results = f"""Expected Annual Return: {percent_ret}
Annual Volatility: {percent_vols}
Annual Variance: {percent_var}

Optimized Weights:
{cleaned_weights}

Discrete Allocation (for $15,000):
{allocation}
Leftover Funds: ${leftover:.2f}
"""
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, results)

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in df.columns:
            ax.plot(df[c], label=c)
        ax.set_title('Portfolio Adj. Close Price History')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adj. Price USD ($)')
        ax.legend(loc='upper left')

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    except Exception as e:
        messagebox.showerror("Error", str(e))


# GUI Setup
window = tk.Tk()
window.title("Portfolio Analyzer with Ticker Search")

# --- Ticker input area ---
tk.Label(window, text="Enter Stock Tickers (comma-separated):").pack()
stock_entry = tk.Entry(window, width=50)
stock_entry.insert(0, "META,AMZN,AAPL,NFLX,GOOG")
stock_entry.pack()

# --- Start date input ---
tk.Label(window, text="Enter Start Date (YYYY-MM-DD):").pack()
start_date_entry = tk.Entry(window, width=20)
start_date_entry.insert(0, "2013-01-01")
start_date_entry.pack()

# --- Ticker Search Section ---
tk.Label(window, text="\nSearch Ticker by Company Name:").pack()
search_entry = tk.Entry(window, width=30)
search_entry.pack()

search_button = tk.Button(window, text="Search", command=search_ticker)
search_button.pack()

suggestions_listbox = tk.Listbox(window, width=60, height=5)
suggestions_listbox.pack()

add_button = tk.Button(window, text="Add Selected Ticker", command=add_selected_ticker)
add_button.pack(pady=5)

# --- Analyze Button ---
analyze_button = tk.Button(window, text="Analyze Portfolio", command=analyze_portfolio)
analyze_button.pack(pady=10)

# --- Results Display ---
result_text = tk.Text(window, height=20, width=80)
result_text.pack()

window.mainloop()
