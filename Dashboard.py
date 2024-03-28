import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf

# Read company data from the CSV file
company_info = pd.read_csv('Companys_data.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Stock Analysis Dashboard"),
    html.Label("Enter Stock Ticker Symbol:"),
    dcc.Input(id="ticker-input", value="AAPL", type="text"),
    dcc.Graph(id="stock-chart"),
    html.H2("Recent News Headlines:"),
    html.Div(id="news-headlines"),
    html.Div(id="company-info")
])

# Callback function to update stock chart based on user input
@app.callback(
    Output("stock-chart", "figure"),
    [Input("ticker-input", "value")]
)
def update_stock_chart(ticker_symbol):
    # Get historical stock data for the specified ticker symbol
    ticker = yf.Ticker(ticker_symbol)
    stock_data = ticker.history(period='1mo')
    
    # Create a line chart of the stock's closing price
    fig = {
        "data": [{"x": stock_data.index, "y": stock_data["Close"], "type": "line", "name": "Closing Price"}],
        "layout": {"title": f"Historical Stock Data for {ticker_symbol}"}
    }
    return fig

# Callback function to update news headlines based on user input
@app.callback(
    Output("news-headlines", "children"),
    [Input("ticker-input", "value")]
)
def update_news_headlines(ticker_symbol):
    # Get recent news headlines for the specified ticker symbol
    ticker = yf.Ticker(ticker_symbol)
    news_headlines = ticker.get_news()  # Default: last 1 week
    
    # Display the news headlines as a list
    headlines_list = [html.P(headline['title']) for headline in news_headlines]
    return headlines_list

# Callback function to update company info based on user input
@app.callback(
    Output("company-info", "children"),
    [Input("ticker-input", "value")]
)
def update_company_info(ticker_symbol):
    # Retrieve company data for the specified ticker symbol
    company_data = company_info[company_info['Company'] == ticker_symbol]
    if not company_data.empty:
        industry = company_data['Industry'].values[0]
        profit_margin = company_data['ProfitMargins'].values[0]
    else:
        industry = "N/A"
        profit_margin = "N/A"
    
    # Display company info
    return [html.P(f"Industry: {industry}"), html.P(f"Profit Margin: {profit_margin}")]

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
