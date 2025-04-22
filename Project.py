import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, ttk
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
        portfolio_value = float(funds_entry.get())
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
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_value)
        allocation, leftover = da.lp_portfolio()

        results = f"""Expected Annual Return: {percent_ret}
Annual Volatility: {percent_vols}
Annual Variance: {percent_var}

Optimized Weights:
{cleaned_weights}

Discrete Allocation (for ${portfolio_value:,.2f}):
{allocation}
Leftover Funds: ${leftover:.2f}
"""
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, results)

        # Clear previous plot if it exists
        for widget in graph_frame.winfo_children():
            widget.destroy()

        # Create new plot
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in df.columns:
            ax.plot(df[c], label=c)
        ax.set_title('Portfolio Adj. Close Price History')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adj. Price USD ($)')
        ax.legend(loc='upper left')

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
window = tk.Tk()
window.title("Portfolio Analyzer with Ticker Search")

# Create main frames
input_frame = tk.Frame(window)
input_frame.pack(fill=tk.X, padx=10, pady=5)

output_frame = tk.Frame(window)
output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# Left side for results
results_frame = tk.Frame(output_frame)
results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Right side for graph
graph_frame = tk.Frame(output_frame)
graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# --- Input elements ---
# Ticker input
tk.Label(input_frame, text="Enter Stock Tickers (comma-separated):").grid(row=0, column=0, sticky='w')
stock_entry = tk.Entry(input_frame, width=50)
stock_entry.insert(0, "META,AMZN,AAPL,NFLX,GOOG")
stock_entry.grid(row=0, column=1, padx=5, pady=2)

# Start date input
tk.Label(input_frame, text="Enter Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky='w')
start_date_entry = tk.Entry(input_frame, width=20)
start_date_entry.insert(0, "2013-01-01")
start_date_entry.grid(row=1, column=1, sticky='w', padx=5, pady=2)

# Funds input
tk.Label(input_frame, text="Portfolio Value ($):").grid(row=2, column=0, sticky='w')
funds_entry = tk.Entry(input_frame, width=20)
funds_entry.insert(0, "15000")
funds_entry.grid(row=2, column=1, sticky='w', padx=5, pady=2)

# Ticker Search Section
tk.Label(input_frame, text="Search Ticker by Company Name:").grid(row=3, column=0, sticky='w', pady=(10, 0))
search_entry = tk.Entry(input_frame, width=30)
search_entry.grid(row=3, column=1, sticky='w', padx=5, pady=(10, 0))

search_button = tk.Button(input_frame, text="Search", command=search_ticker)
search_button.grid(row=3, column=2, padx=5, pady=(10, 0))

suggestions_listbox = tk.Listbox(input_frame, width=60, height=5)
suggestions_listbox.grid(row=4, column=0, columnspan=3, sticky='w', padx=5, pady=2)

add_button = tk.Button(input_frame, text="Add Selected Ticker", command=add_selected_ticker)
add_button.grid(row=5, column=0, columnspan=3, pady=5)

# Analyze Button
analyze_button = tk.Button(input_frame, text="Analyze Portfolio", command=analyze_portfolio)
analyze_button.grid(row=6, column=0, columnspan=3, pady=10)

# --- Results Display ---
result_text = tk.Text(results_frame, height=20, width=55)
result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Graph will be displayed in graph_frame when analysis is run

window.mainloop()
