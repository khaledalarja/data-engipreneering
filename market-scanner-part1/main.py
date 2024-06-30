import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Define sectors and top companies (20 stocks per sector)
sectors = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AVGO', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'INTC', 'AMD', 'IBM', 'TXN', 'QCOM', 'PYPL', 'NOW', 'ACN'],
    'Healthcare': ['JNJ', 'UNH', 'LLY', 'PFE', 'ABT', 'TMO', 'MRK', 'ABBV', 'DHR', 'BMY', 'AMGN', 'CVS', 'ISRG', 'GILD', 'MDT', 'VRTX', 'SYK', 'ZTS', 'REGN', 'BIIB'],
    'Financials': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB', 'PNC', 'TFC', 'SPGI', 'CME', 'COF', 'CB', 'MMC', 'AON', 'MET', 'AIG'],
    'Consumer Discretionary': ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'BKNG', 'TJX', 'MAR', 'ROST', 'F', 'GM', 'EBAY', 'YUM', 'DPZ', 'ETSY', 'LVS', 'RCL', 'CCL'],
    'Industrials': ['HON', 'UNP', 'UPS', 'BA', 'CAT', 'GE', 'MMM', 'LMT', 'RTX', 'DE', 'FDX', 'EMR', 'ETN', 'ITW', 'CSX', 'NSC', 'WM', 'JCI', 'ROK', 'CMI'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'KMI', 'WMB', 'OXY', 'PXD', 'HAL', 'DVN', 'HES', 'OKE', 'BKR', 'MRO', 'APA', 'FANG'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'EL', 'CL', 'GIS', 'KMB', 'SYY', 'ADM', 'KHC', 'STZ', 'KR', 'HSY', 'TSN', 'CAG', 'K'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'WEC', 'ED', 'ES', 'ETR', 'FE', 'AEE', 'CMS', 'DTE', 'AWK', 'AES', 'EIX'],
    'Real Estate': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'WELL', 'AVB', 'EQR', 'DLR', 'SPG', 'O', 'VICI', 'ARE', 'CBRE', 'VTR', 'BXP', 'UDR', 'ESS', 'EXR', 'KIM'],
    'Materials': ['LIN', 'SHW', 'APD', 'FCX', 'ECL', 'NEM', 'DOW', 'NUE', 'VMC', 'ALB', 'CTVA', 'MLM', 'FMC', 'DD', 'PPG', 'IFF', 'CF', 'MOS', 'LYB', 'CE']
}

def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close'], data['Volume']

def calculate_indicators(df):
    df_ta = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume"
    )
    return df_ta

def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'P/E Ratio': info.get('trailingPE', np.nan),
        'PEG Ratio': info.get('pegRatio', np.nan),
        'Debt/Equity': info.get('debtToEquity', np.nan),
        'ROE': info.get('returnOnEquity', np.nan),
        'Profit Margin': info.get('profitMargins', np.nan)
    }

def analyze_sector(sector_data, sector_name, individual_stocks):
    adx = ADXIndicator(sector_data['High'], sector_data['Low'], sector_data['Close'])
    atr = AverageTrueRange(sector_data['High'], sector_data['Low'], sector_data['Close'])
    
    sector_data['ADX'] = adx.adx()
    sector_data['ATR'] = atr.average_true_range()
    sector_data['RSI'] = calculate_indicators(sector_data)['momentum_rsi']
    sector_data['MACD'] = calculate_indicators(sector_data)['trend_macd']
    sector_data['Signal'] = calculate_indicators(sector_data)['trend_macd_signal']
    sector_data['MA50'] = sector_data['Close'].rolling(window=50).mean()
    sector_data['MA200'] = sector_data['Close'].rolling(window=200).mean()
    
    # Calculate returns for individual stocks
    stock_returns = individual_stocks.pct_change().iloc[-1].sort_values(ascending=False)
    top_5_stocks = stock_returns.head(5)
    
    sector_analysis = {
        'Sector': sector_name,
        'Avg ADX': sector_data['ADX'].mean(),
        'Avg Volatility': sector_data['ATR'].mean(),
        'Avg RSI': sector_data['RSI'].mean(),
        'Bullish MA Cross': (sector_data['MA50'] > sector_data['MA200']).mean(),
        'Avg Volume Change': sector_data['Volume'].pct_change().mean(),
        '1M Return': sector_data['Close'].pct_change(periods=20).mean(),
        '3M Return': sector_data['Close'].pct_change(periods=60).mean(),
        'MACD Bullish': (sector_data['MACD'] > sector_data['Signal']).mean(),
        'Top 5 Stocks': ', '.join(top_5_stocks.index),
        'Top 5 Returns': ', '.join([f'{r:.2%}' for r in top_5_stocks.values])
    }
    
    return pd.Series(sector_analysis)

def generate_report(sector_analysis):
    # Sort sectors by a composite score
    sector_analysis['Trend Score'] = (
        sector_analysis['Avg ADX'] * 0.2 +
        sector_analysis['Bullish MA Cross'] * 0.2 +
        sector_analysis['Avg RSI'] * 0.1 +
        sector_analysis['1M Return'] * 0.2 +
        sector_analysis['3M Return'] * 0.2 +
        sector_analysis['MACD Bullish'] * 0.1
    )
    
    sorted_sectors = sector_analysis.sort_values('Trend Score', ascending=False)
    
    report = "Market Sector Trend Analysis Report\n"
    report += "===================================\n\n"
    
    for idx, (sector, data) in enumerate(sorted_sectors.iterrows(), 1):
        report += f"{idx}. {sector}\n"
        report += f"   Trend Score: {data['Trend Score']:.2f}\n"
        report += f"   Average ADX: {data['Avg ADX']:.2f}\n"
        report += f"   Bullish MA Cross: {data['Bullish MA Cross']*100:.2f}%\n"
        report += f"   Average RSI: {data['Avg RSI']:.2f}\n"
        report += f"   1-Month Return: {data['1M Return']*100:.2f}%\n"
        report += f"   3-Month Return: {data['3M Return']*100:.2f}%\n"
        report += f"   MACD Bullish: {data['MACD Bullish']*100:.2f}%\n"
        report += f"   Top 5 Stocks: {data['Top 5 Stocks']}\n"
        report += f"   Top 5 Returns: {data['Top 5 Returns']}\n\n"
    
    report += "Analysis Summary:\n"
    report += "The top 3 trending sectors with the strongest money flow are:\n"
    for i in range(3):
        report += f"{i+1}. {sorted_sectors.index[i]}\n"
    
    return report

def create_dashboard(sector_analysis, historical_data):
    plt.figure(figsize=(20, 15))
    plt.suptitle("Market Overview Dashboard", fontsize=20)
    
    # Sector Performance Heatmap
    plt.subplot(2, 2, 1)
    performance_data = sector_analysis[['1M Return', '3M Return']]
    sns.heatmap(performance_data, annot=True, cmap='RdYlGn', fmt='.2%', cbar=False)
    plt.title('Sector Performance')
    
    # Top Sectors by Trend Score
    plt.subplot(2, 2, 2)
    top_sectors = sector_analysis.sort_values('Trend Score', ascending=False).head(5)
    plt.barh(top_sectors.index, top_sectors['Trend Score'])
    plt.title('Top 5 Sectors by Trend Score')
    plt.xlabel('Trend Score')
    
    # Historical Sector Performance
    plt.subplot(2, 2, 3)
    for sector in historical_data.columns:
        plt.plot(historical_data.index, historical_data[sector], label=sector)
    plt.title('3-Month Sector Performance')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Average RSI by Sector
    plt.subplot(2, 2, 4)
    plt.bar(sector_analysis.index, sector_analysis['Avg RSI'])
    plt.title('Average RSI by Sector')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('RSI')
    
    plt.tight_layout()
    plt.savefig('market_overview_dashboard.png')
    plt.close()

def identify_opportunities(sector_analysis, all_stock_data):
    opportunities = []
    
    for sector, sector_info in sector_analysis.iterrows():
        sector_return = sector_info['3M Return']
        sector_stocks = sectors[sector]
        
        for stock in sector_stocks:
            stock_data = all_stock_data[stock]
            stock_return = stock_data['Close'].pct_change(periods=60).iloc[-1]  # 3M return
            
            # Calculate relative performance
            relative_performance = stock_return - sector_return
            
            # Get fundamental data
            fundamentals = get_fundamental_data(stock)
            
            # Score the stock (you may adjust these thresholds as needed)
            score = 0
            if fundamentals['P/E Ratio'] and fundamentals['P/E Ratio'] < 20:
                score += 1
            if fundamentals['PEG Ratio'] and fundamentals['PEG Ratio'] < 1.5:
                score += 1
            if fundamentals['Debt/Equity'] and fundamentals['Debt/Equity'] < 1:
                score += 1
            if fundamentals['ROE'] and fundamentals['ROE'] > 0.15:
                score += 1
            if fundamentals['Profit Margin'] and fundamentals['Profit Margin'] > 0.1:
                score += 1
            
            # Identify underperforming stocks with good fundamentals
            if relative_performance < -0.05 and score >= 3:
                opportunities.append({
                    'Stock': stock,
                    'Sector': sector,
                    'Relative Performance': relative_performance,
                    'Fundamental Score': score,
                    'P/E Ratio': fundamentals['P/E Ratio'],
                    'PEG Ratio': fundamentals['PEG Ratio'],
                    'Debt/Equity': fundamentals['Debt/Equity'],
                    'ROE': fundamentals['ROE'],
                    'Profit Margin': fundamentals['Profit Margin']
                })
    
    return pd.DataFrame(opportunities)

def generate_opportunity_report(opportunities):
    report = "Potential Investment Opportunities\n"
    report += "=================================\n\n"
    
    for _, opportunity in opportunities.iterrows():
        report += f"Stock: {opportunity['Stock']} (Sector: {opportunity['Sector']})\n"
        report += f"   Relative Performance: {opportunity['Relative Performance']:.2%}\n"
        report += f"   Fundamental Score: {opportunity['Fundamental Score']}/5\n"
        report += f"   P/E Ratio: {opportunity['P/E Ratio']:.2f}\n"
        report += f"   PEG Ratio: {opportunity['PEG Ratio']:.2f}\n"
        report += f"   Debt/Equity: {opportunity['Debt/Equity']:.2f}\n"
        report += f"   ROE: {opportunity['ROE']:.2%}\n"
        report += f"   Profit Margin: {opportunity['Profit Margin']:.2%}\n\n"
    
    return report

# Main execution
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

sector_analysis = []
all_stock_data = {}
historical_data = pd.DataFrame()

for sector, tickers in sectors.items():
    prices, volumes = fetch_data(tickers, start_date, end_date)
    sector_data = pd.DataFrame({'Close': prices.mean(axis=1), 'Volume': volumes.sum(axis=1)})
    sector_data['Open'] = sector_data['Close'].shift(1)
    sector_data['High'] = sector_data['Close'].rolling(window=2).max()
    sector_data['Low'] = sector_data['Close'].rolling(window=2).min()
    sector_analysis.append(analyze_sector(sector_data, sector, prices))
    
    # Normalize prices for historical comparison
    historical_data[sector] = sector_data['Close'] / sector_data['Close'].iloc[0]
    
    # Populate all_stock_data correctly
    for ticker in tickers:
        all_stock_data[ticker] = pd.DataFrame({
            'Close': prices[ticker], 
            'Volume': volumes[ticker],
            'Open': prices[ticker].shift(1),
            'High': prices[ticker].rolling(window=2).max(),
            'Low': prices[ticker].rolling(window=2).min()
        })

sector_analysis = pd.DataFrame(sector_analysis)
sector_analysis.set_index('Sector', inplace=True)

# Generate and print the sector analysis report
report = generate_report(sector_analysis)
print(report)

# Create and save the market overview dashboard
create_dashboard(sector_analysis, historical_data)
print("Market overview dashboard has been saved as 'market_overview_dashboard.png'")

# Identify investment opportunities
opportunities = identify_opportunities(sector_analysis, all_stock_data)

# Generate and print the opportunity report
opportunity_report = generate_opportunity_report(opportunities)
print(opportunity_report)

# Create a scatter plot of opportunities
plt.figure(figsize=(12, 8))
plt.scatter(opportunities['Relative Performance'], opportunities['Fundamental Score'], alpha=0.6)
plt.xlabel('Relative Performance')
plt.ylabel('Fundamental Score')
plt.title('Investment Opportunities: Relative Performance vs Fundamental Score')

for i, txt in enumerate(opportunities['Stock']):
    plt.annotate(txt, (opportunities['Relative Performance'].iloc[i], opportunities['Fundamental Score'].iloc[i]))

plt.axvline(x=0, color='r', linestyle='--')
plt.axhline(y=3, color='r', linestyle='--')
plt.savefig('investment_opportunities.png')
plt.close()

print("Investment opportunities chart has been saved as 'investment_opportunities.png'")