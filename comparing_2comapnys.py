import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def compare_companies(symbols):
    companies_data = []

    for symbol in symbols:
        company = yf.Ticker(symbol)
        company_data = {
            'Company': symbol,
            'Revenue': company.info.get('totalRevenue', None),
            'Industry': company.info.get('industry', None),
            'ProfitMargins': company.info.get('profitMargins',None),
            'NetIncome': company.info.get('netIncomeToCommon',None)
        }
        companies_data.append(company_data)

    return pd.DataFrame(companies_data)

def plot_comparison(comparison_df):
    plt.figure(figsize=(10, 6))

    for index, row in comparison_df.iterrows():
        plt.bar(index, row['Revenue'], label=row['Company'])

    plt.xlabel('Company')
    plt.ylabel('Revenue')
    plt.title('Comparison of Companies Based on Revenue')
    plt.xticks(range(len(comparison_df)), comparison_df['Company'])
    plt.legend()
    plt.show()

# Input symbols of companies separated by space
symbols = input("Enter symbols of companies separated by space: ").split()

# Compare companies based on industry
comparison_df = compare_companies(symbols)
comparison_df.to_csv('Companys_data.csv')

# Plot comparison
plot_comparison(comparison_df)
