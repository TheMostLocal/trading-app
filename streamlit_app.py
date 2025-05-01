import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

# Function to load price data and fundamentals
def load_price_data(ticker):
    df = yf.download(ticker, period="3y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def load_fundamentals(ticker):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    return {
        "Market Cap": f"${info.get('marketCap', 0):,}",
        "Trailing P/E": info.get("trailingPE", "N/A"),
        "Forward P/E": info.get("forwardPE", "N/A"),
        "PEG Ratio": info.get("pegRatio", "N/A"),
        "Price/Sales (TTM)": info.get("priceToSalesTrailing12Months", "N/A"),
        "Price/Book (MRQ)": info.get("priceToBook", "N/A"),
        "EPS (TTM)": info.get("trailingEps", "N/A"),
        "Revenue (TTM)": f"${info.get('totalRevenue', 0):,}" if info.get("totalRevenue") else "N/A",
        "EBITDA": f"${info.get('ebitda', 0):,}" if info.get("ebitda") else "N/A",
        "Return on Equity (ROE)": info.get("returnOnEquity", "N/A"),
        "Operating Margin": info.get("operatingMargins", "N/A")
    }

# Sidebar for navigation between pages
def app():
    st.set_page_config(page_title="Stock Dashboard", layout="wide")
    page = st.sidebar.selectbox("Select a Page", ["Stock Dashboard", "Options & Implied Volatility"])

    if page == "Stock Dashboard":
        stock_dashboard()
    elif page == "Options & Implied Volatility":
        options_page()

# Stock Dashboard
def stock_dashboard():
    st.title("📊 Stock Tracker Dashboard")

    # ----------- Ticker Selection -----------
    ticker_symbol = st.text_input("Enter Stock Ticker (e.g., GME, AAPL):", "GME").upper()

    # ----------- Load Data -----------
    df = load_price_data(ticker_symbol)
    financials = load_fundamentals(ticker_symbol)
    last_30 = df.tail(30)

    # ----------- Display Stock Information ----------- 
    st.subheader(f"Stock Information for {ticker_symbol}")
    st.dataframe(financials)

    # ----------- Price Chart (3-Year) -----------
    st.subheader("📈 Price Chart (3-Year)")
    price_chart_data = df.reset_index()
    line_chart = alt.Chart(price_chart_data).mark_line().encode(
        x='Date:T',
        y=alt.Y('Close:Q', title='Price'),
        color=alt.value('white'),
        tooltip=['Date:T', 'Close:Q']
    ).properties(height=400)

    st.altair_chart(line_chart.interactive(), use_container_width=True)

# Options & Implied Volatility Page
def options_page():
    import scipy.stats as stats
    # Function to fetch options data using yfinance
    def fetch_options_data(ticker_symbol):
        ticker = yf.Ticker(ticker_symbol)
        expiration_dates = ticker.options
        st.write(f"Available Expiration Dates: {expiration_dates}")
        option_chain = ticker.option_chain(expiration_dates[0])  # Get the first expiration date
        calls = option_chain.calls
        puts = option_chain.puts
        return calls, puts

    # Black-Scholes Model for calculating implied volatility
    def black_scholes_implied_volatility(S, K, T, r, market_price, option_type='call'):
        sigma = 0.2  # Start with 20% volatility
        tolerance = 1e-5  # Convergence tolerance
        max_iterations = 100
        
        for i in range(max_iterations):
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            if option_type == 'call':
                option_price = S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
            elif option_type == 'put':
                option_price = K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            vega = S * math.sqrt(T) * stats.norm.pdf(d1)
            price_diff = option_price - market_price
            if abs(price_diff) < tolerance:
                return sigma
            sigma = sigma - price_diff / vega
        return sigma

    def calculate_implied_volatility(ticker_symbol):
        calls, puts = fetch_options_data(ticker_symbol)
        r = 0.01  # Risk-free rate assumption
        ticker = yf.Ticker(ticker_symbol)
        stock_price = ticker.history(period="1d")['Close'].iloc[0]
        implied_volatilities = []
        
        for i in range(min(5, len(calls))):
            call_option = calls.iloc[i]
            strike = call_option['strike']
            market_price = call_option['lastPrice']
            T = (pd.to_datetime(call_option['expiration']) - pd.to_datetime('today')).days / 365
            iv = black_scholes_implied_volatility(stock_price, strike, T, r, market_price, option_type='call')
            implied_volatilities.append({
                'strike': strike,
                'market_price': market_price,
                'implied_volatility': iv
            })
        return implied_volatilities

    st.set_page_config(page_title="Options & Implied Volatility", layout="wide")
    st.title("📈 Stock Options and Implied Volatility")

    # Ticker input for options
    ticker_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, GME):", "AAPL").upper()

    # Calculate and display implied volatilities
    implied_volatilities = calculate_implied_volatility(ticker_symbol)
    st.write(f"Implied Volatilities for {ticker_symbol}:")
    st.write(implied_volatilities)

    # Display the options data (calls and puts)
    calls, puts = fetch_options_data(ticker_symbol)

    st.subheader(f"Call Options Data for {ticker_symbol}:")
    st.dataframe(calls)

    st.subheader(f"Put Options Data for {ticker_symbol}:")
    st.dataframe(puts)

if __name__ == '__main__':
    app()
