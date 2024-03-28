import yfinance as yf

def search_stock_data(ticker_symbol):
    """
    Search for historical stock data and recent news headlines for a given stock ticker symbol.
    
    Args:
    - ticker_symbol (str): Ticker symbol of the stock (e.g., AAPL for Apple Inc.).
    
    Returns:
    - Tuple containing historical stock data DataFrame and list of recent news headlines.
    """
    try:
        # Initialize a Ticker object with the provided ticker symbol
        ticker = yf.Ticker(ticker_symbol)

        # Get historical stock data (default: last 30 days)
        stock_data = ticker.history(period='1mo')

        # Get recent news headlines
        news_headlines = ticker.get_news()  # Default: last 1 week
        
        return stock_data, news_headlines

    except Exception as e:
        print("An error occurred:", e)
        return None, None

# Example usage:
ticker_symbol = input("Enter a ticker symbol: ").upper()  # Convert input to uppercase

# Search for historical stock data and recent news headlines
stock_data, news_headlines = search_stock_data(ticker_symbol)

# Print the fetched stock data
if stock_data is not None:
    print("Historical Stock Data:")
    print(stock_data)
else:
    print("No stock data available for the provided ticker symbol.")

# Print the fetched news headlines
if news_headlines is not None:
    print("\nRecent News Headlines:")
    for headline in news_headlines:
        print(headline['title'])
else:
    print("No recent news headlines available.")
