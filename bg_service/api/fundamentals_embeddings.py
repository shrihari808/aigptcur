import requests
import json
import os
import tempfile
from datetime import datetime
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import csv
import chromadb
from chromadb.utils import embedding_functions
from fuzzywuzzy import process
from config import chroma_server_client

import chromadb.utils.embedding_functions as embedding_functions
import os
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Response, APIRouter, HTTPException

router=APIRouter()

load_dotenv(override=True)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )


default_ef = embedding_functions.DefaultEmbeddingFunction()



client=chroma_server_client
print(client)
token=os.getenv("CMOTS_BEARER_TOKEN")




def find_stock_code(stock_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv("csv_data\company_codes2.csv")

    # Extract the company names and codes from the DataFrame
    company_names = df["Company Name"].tolist()
    company_codes = df["Company Code"].tolist()
    
    threshold=80
    # Use fuzzy matching to find the closest match to the input stock name
    match = process.extractOne(stock_name, company_names)
   
    if match and match[1] >= threshold:
        # Get the index of the closest match
        idx = company_names.index(match[0])
        # Return the corresponding stock code
        return company_codes[idx]
    else:
        return 0

def get_yearly_results(stock_name):


    if find_stock_code:
        stock_code=find_stock_code(stock_name)
        stock_code=int(stock_code)
        # Define the API endpoint URL with the stock code
        api_url = f'http://airrchipapis.cmots.com/api/Yearly-Results/{stock_code}/s'

        # Set up the headers with the global token
        headers = {'Authorization': f'Bearer {token}'}

        # Make the GET request to the API with headers
        response = requests.get(api_url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Restructure the data with properly formatted dates
            original_data = response.json()
            original_data = original_data['data']
            restructured_data = {}
            for entry in original_data:
                column_name = entry["COLUMNNAME"]
                rid = entry["RID"]
                for year_key in entry.keys():
                    if year_key.startswith("Y"):
                        year = year_key[1:5]  # Extract year from the year_key
                        month = year_key[5:]  # Extract month from the year_key
                        formatted_date = f"Yearly_Results_{year}-{month} in crs"  # Format date as "Yearly_Results_YYYY-MM"
                        if formatted_date not in restructured_data:
                            restructured_data[formatted_date] = {}
                        restructured_data[formatted_date][column_name] = entry[year_key]

            # Convert the restructured data to JSON format
            json_data = json.dumps(restructured_data, indent=4)
            return json_data
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code} - {response.text}")
            return None
        

def half_yearly_results(stock_name):
    
    if find_stock_code:
        stock_code=find_stock_code(stock_name)
        stock_code=int(stock_code)
        # Define the API endpoint URL with the stock code
        api_url = f'http://airrchipapis.cmots.com/api/Half-Yearly-Results/{stock_code}/s'

        # Set up the headers with the global token
        headers = {'Authorization': f'Bearer {token}'}

        # Make the GET request to the API with headers
        response = requests.get(api_url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Restructure the data with properly formatted dates
            original_data = response.json()
            original_data = original_data['data']
            restructured_data = {}
            for entry in original_data:
                column_name = entry["COLUMNNAME"]
                rid = entry["RID"]
                for year_key in entry.keys():
                    if year_key.startswith("Y"):
                        year = year_key[1:5]  # Extract year from the year_key
                        month = year_key[5:]  # Extract month from the year_key
                        formatted_date = f"Half_Yearly_Results_{year}-{month} in crs(6 months results)"  # Format date as "Yearly_Results_YYYY-MM"
                        if formatted_date not in restructured_data:
                            restructured_data[formatted_date] = {}
                        restructured_data[formatted_date][column_name] = entry[year_key]

            # Convert the restructured data to JSON format
            json_data = json.dumps(restructured_data, indent=4)
            return json_data
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code} - {response.text}")
            return None
        
def nine_month_results(stock_name):
    
    if find_stock_code:
        stock_code=find_stock_code(stock_name)
        stock_code=int(stock_code)
        # Define the API endpoint URL with the stock code
        api_url = f'http://airrchipapis.cmots.com/api/Nine-Month-Result/{stock_code}/s'

        # Set up the headers with the global token
        headers = {'Authorization': f'Bearer {token}'}

        # Make the GET request to the API with headers
        response = requests.get(api_url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Restructure the data with properly formatted dates
            original_data = response.json()
            original_data = original_data['data']
            restructured_data = {}
            for entry in original_data:
                column_name = entry["COLUMNNAME"]
                rid = entry["RID"]
                for year_key in entry.keys():
                    if year_key.startswith("Y"):
                        year = year_key[1:5]  # Extract year from the year_key
                        month = year_key[5:]  # Extract month from the year_key
                        formatted_date = f"Nine_Month_Result_{year}-{month} in crs"  # Format date as "Yearly_Results_YYYY-MM"
                        if formatted_date not in restructured_data:
                            restructured_data[formatted_date] = {}
                        restructured_data[formatted_date][column_name] = entry[year_key]

            # Convert the restructured data to JSON format
            json_data = json.dumps(restructured_data, indent=4)
            return json_data
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code} - {response.text}")
            return None
        

def get_balance_sheet(stock_name):
    stock_code = find_stock_code(stock_name)
    if stock_code == 0:
        return "The stock you inquired about is not part of the Nifty 50 index, ONLY ASK QUESTIONS ABOUT NIFTY 50 STOCKS atm."

    # Construct the URL with the company code
    url = f"http://airrchipapis.cmots.com/api/BalanceSheet/{stock_code}/S"
    
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Return the response content as JSON
        data = response.json()

        columns_to_remove = ['Non-Current Assets:', 'Intangible Assets under Development', 'Capital Work in Progress',
                             'Investment Properties', 'Investments of Life Insurance Business',
                             'Biological Assets other than Bearer Plants (Non Current)', 'Loans - Long-term',
                             'Insurance Related Assets (Non Current)',
                             'Deferred Tax Assets', 'Current Assets:',
                             'Biological Assets other than Bearer Plants (Current)', 'Current Tax Assets - Short-term',
                             'Insurance Related Assets (Current)', 'Assets Classified as Held for Sale',
                             'Current Liabilities:', 'Insurance Related Liabilities (Current)',
                             'Liabilities Directly Associated with Assets Classified as Held for Sale',
                             'Current Tax Liabilities - Short-term', 'Other Short term Provisions',
                             'Non-Current Liabilities:', 'Debt Securities', 'Lease Liabilities (Non Current)',
                             'Other Long term Liabilities', 'Others Financial Liabilities - Long-term',
                             'Insurance Related Liabilities (Non Current)', 'Other Long term Provisions',
                             'Deferred Tax Liabilities', 'Shareholders Funds:', 'Preference Capital',
                             'Unclassified Capital', 'Reserves and Surplus', 'Other Equity Components',
                             'Total Shareholder\'s Fund',
                             'Contingent Liabilities and Commitments (to the Extent Not Provided for)',
                             'Ordinary Shares :', 'Authorised:', 'Number of Equity Shares - Authorised',
                             'Amount of Equity Shares - Authorised', 'Par Value of Authorised Shares',
                             'Susbcribed & fully Paid up :', 'Par Value', 'Susbcribed & fully Paid up Shares',
                             'Susbcribed & fully Paid up CapItal', 'Right-of-Use Assets']

        # Remove entries with specified 'COLUMNNAME', considering leading white spaces
        filtered_data = [entry for entry in data['data'] if entry['COLUMNNAME'].strip() not in columns_to_remove]

        # Update the 'data' field with the filtered data
        data['data'] = filtered_data

        # Dictionary to store year-wise data
        yearwise_data = {}

        # Process each data dictionary
        for data_entry in data['data']:
            asset_type = data_entry['COLUMNNAME'].rstrip(':')  # Remove trailing ':' if present

            for key, value in data_entry.items():
                if key.startswith('Y20'):
                    # Extracting the year from the key
                    year = key[1:5]  # Extracting from the 2nd character onwards
                    # Creating year entry if it doesn't exist
                    if year not in yearwise_data:
                        yearwise_data[year] = {"BalanceSheet": {}}

                    # Adding asset type and value for the current year
                    yearwise_data[year]["BalanceSheet"][asset_type] = f"{value} Crs"

        # Dictionary to store data with modified keys
        modified_data = {}

        # Process each data dictionary
        for year, data in yearwise_data.items():
            modified_key = f"BalanceSheet_{year}"
            modified_data[modified_key] = data["BalanceSheet"]

        # Convert the dictionary to a JSON string with double quotes
        json_output = json.dumps(modified_data, indent=4)
        return json_output
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)
        return

def get_profit_loss(stock_name):
    stock_code = find_stock_code(stock_name)

    if stock_code == 0:
        return "The stock you inquired about is not part of the Nifty 50 index, ONLY ASK QUESTIONS ABOUT NIFTY 50 STOCKS atm."

    # Construct the URL with the company code
    url = f"http://airrchipapis.cmots.com/api/ProftandLoss/{stock_code}/S"

    
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Return the response content as JSON
        data = response.json()

        columns_to_remove = ['Income from Investment and Financial Services',
                              'Income from Insurance Operations',
                              'Other Operating Revenue',
                              'Less: Excise Duty / GSTInternally Manufactured Intermediates Consumed',
                              'Purchases of Stock-in-Trade',
                              '   Administrative and Selling Expenses',
                              'Profit Before Exceptional Items and Tax',
                              'Profit Before Extraordinary Items and Tax',
                              'Other Adjustments Before Tax',
                              'MAT Credit Entitlement',
                              'Other Tax',
                              'Adjust for Previous Year',
                              'Extraordinary Items After Tax',
                              'Discontinued Operations After Tax',
                              'Profit / (Loss) from Discontinuing Operations',
                              'Tax Expenses of Discontinuing Operations',
                              'Profit Attributable to Shareholders',
                              'Adjustments to Net Income',
                              'Profit Attributable to Equity Shareholders',
                              'Weighted Average Shares - Basic']

        # Remove entries with specified 'COLUMNNAME', considering leading white spaces
        filtered_data = [entry for entry in data['data'] if entry['COLUMNNAME'].strip() not in columns_to_remove]

        # Update the 'data' field with the filtered data
        data['data'] = filtered_data

        # Dictionary to store profit and loss data
        pnl_data = {}

        # Process each data dictionary
        for data_entry in data['data']:
            asset_type = data_entry['COLUMNNAME'].rstrip(':')  # Remove trailing ':' if present

            for key, value in data_entry.items():
                if key.startswith('Y20'):
                    # Extracting the year from the key
                    year = key[1:5]  # Extracting from the 2nd character onwards

                    # Creating a key for the year's profit and loss data
                    pnl_key = f"pnl_statement_{year}"

                    # Creating the profit and loss data entry if it doesn't exist
                    if pnl_key not in pnl_data:
                        pnl_data[pnl_key] = {}

                    # Adding asset type and value for the current year
                    pnl_data[pnl_key][asset_type] = f'"{value} Crs"'

        # Convert the dictionary to a JSON-like string with line breaks and indentation
        json_output = "{\n"
        for key, data in pnl_data.items():
            json_output += f'    "{key}": {{\n'
            pnl_items = [f'        "{k}": {v}' for k, v in data.items()]
            json_output += ',\n'.join(pnl_items)
            json_output += "\n    },"

        # Remove the trailing comma and add the closing brace
        json_output = json_output.rstrip(",") + "\n}"

        return json_output
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)
        return
# def get_all_results(stock_name):
#     results = {}
#     results['Yearly_Results'] = json.loads(get_yearly_results(stock_name))
#     results['Quarterly_Results'] = json.loads(half_yearly_results(stock_name))
#     results['Half_Yearly_Results'] = json.loads(nine_month_results(stock_name))
#     # add more functions here
#     return results


# def get_company_profile(stock_name):
#     # token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0'
#     code = int(find_stock_code(stock_name))
#     headers = {
#         'Authorization': f'Bearer {token}'
#     }
#     url = f'http://airrchipapis.cmots.com/api/CompanyProfile/{code}'
#     response = requests.get(url, headers=headers)
#     result = response.json()
#     company_profile = {}
#     for ratio in result['data']:
#         for key, value in ratio.items():
#             company_profile[key] = value
#     return json.dumps({"comapany_profile_data": company_profile})

def get_company_profile(stock_name):
    code = int(find_stock_code(stock_name))
    headers = {
        'Authorization': f'Bearer {token}'
    }
    url = f'http://airrchipapis.cmots.com/api/CompanyProfile/{code}'
    response = requests.get(url, headers=headers)
    result = response.json()
    company_profile = {}
    if result["data"] is not None:
        for key, value in result.items():
            if value is not None:
                company_profile[key] = value
    return json.dumps({"comapany_profile_data": company_profile})

def get_company_history(stock_name):

    code=int(find_stock_code(stock_name))
    headers = {
        'Authorization': f'Bearer {token}'
    }
    url = f'http://airrchipapis.cmots.com/api/CompanyHistory/{code}'
    response = requests.get(url, headers=headers)
    result=response.json()
    company_history = {}
    for ratio in result['data']:
        for key, value in ratio.items():
            if value is not None:
                company_history[key] = value
    return  json.dumps({"comapany_history_data" : company_history})

def get_daily_ratios(stock_name):
    code=int(find_stock_code(stock_name))
    headers = {
        'Authorization': f'Bearer {token}'
    }
    url = f'http://airrchipapis.cmots.com/api/DailyRatios/{code}/S'
    response = requests.get(url, headers=headers)
    result=response.json()
    daily_ratios = {}
    for ratio in result['data']:
        for key, value in ratio.items():
            daily_ratios[key] = value
    return json.dumps({"Financial ratios(updated daily)": daily_ratios})



def get_daily_ratios2(stock_name):
    code = int(find_stock_code(stock_name))
    if code is None:
        return json.dumps({"error": "Stock code not found"})
    
    headers = {
        'Authorization': f'Bearer {token}'
    }
    url = f'http://airrchipapis.cmots.com/api/DailyRatios/{code}/S'
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return json.dumps({"error": "Failed to retrieve data"})
    
    result = response.json()
    
    descriptive_mapping = {
        "MCAP": "Market Capitalization (MCAP)",
        "EV": "Enterprise Value (EV)",
        "PE": "Price to Earnings Ratio (PE)",
        "PBV": "Price to Book Value (PBV)",
        "DIVYIELD": "Dividend Yield (DIVYIELD)",
        "EPS": "Earnings Per Share (EPS)",
        "BookValue": "Book Value per Share (BookValue)",
        "ROA_TTM": "Return on Assets (ROA_TTM)",
        "ROE_TTM": "Return on Equity (ROE_TTM)",
        "ROCE_TTM": "Return on Capital Employed (ROCE_TTM)",
        "EBIT_TTM": "Earnings Before Interest and Taxes (EBIT_TTM)",
        "EBITDA_TTM": "Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA_TTM)",
        "EV_Sales_TTM": "EV to Sales Ratio (EV_Sales_TTM)",
        "EV_EBITDA_TTM": "EV to EBITDA Ratio (EV_EBITDA_TTM)",
        "NetIncomeMargin_TTM": "Net Income Margin (NetIncomeMargin_TTM)",
        "GrossIncomeMargin_TTM": "Gross Income Margin (GrossIncomeMargin_TTM)",
        "AssetTurnover_TTM": "Asset Turnover Ratio (AssetTurnover_TTM)",
        "CurrentRatio_TTM": "Current Ratio (CurrentRatio_TTM)",
        "Debt_Equity_TTM": "Debt to Equity Ratio (Debt_Equity_TTM)",
        "Sales_TotalAssets_TTM": "Sales to Total Assets Ratio (Sales_TotalAssets_TTM)",
        "NetDebt_EBITDA_TTM": "Net Debt to EBITDA Ratio (NetDebt_EBITDA_TTM)",
        "EBITDA_Margin_TTM": "EBITDA Margin (EBITDA_Margin_TTM)",
        "TotalShareHoldersEquity_TTM": "Total Shareholders' Equity (TotalShareHoldersEquity_TTM)",
        "ShorttermDebt_TTM": "Short-term Debt (ShorttermDebt_TTM)",
        "LongtermDebt_TTM": "Long-term Debt (LongtermDebt_TTM)",
        "SharesOutstanding": "Shares Outstanding (SharesOutstanding)",
        "EPSDiluted": "Earnings Per Share (Diluted) (EPSDiluted)",
        "NetSales": "Net Sales (NetSales)",
        "Netprofit": "Net Profit (Netprofit)",
        "AnnualDividend": "Annual Dividend (AnnualDividend)",
        "COGS": "Cost of Goods Sold (COGS)",
        "PEGRatio_TTM": "Price/Earnings to Growth Ratio (PEGRatio_TTM)",
        "DividendPayout_TTM": "Dividend Payout Ratio (DividendPayout_TTM)",
        "Industry_PE": "Industry PE Ratio (Industry_PE)"
    }
    
    daily_ratios = {}
    for ratio in result['data']:
        for key, value in ratio.items():
            descriptive_key = descriptive_mapping.get(key, key)
            daily_ratios[descriptive_key] = value
    
    return json.dumps({"Financial ratios(updated daily)": daily_ratios}, indent=4)






def get_quarterly_results(stock_name):
    if find_stock_code:
        stock_code = find_stock_code(stock_name)
        stock_code = int(stock_code)
        
        # Define the API endpoint URL with the stock code
        api_url = f'http://airrchipapis.cmots.com/api/QuarterlyResults/{stock_code}/S'
        
        # Set up the headers with the global token
        headers = {'Authorization': f'Bearer {token}'}
        
        # Make the GET request to the API with headers
        response = requests.get(api_url, headers=headers)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Get the original data from the API response
            original_data = response.json()['data']
            
            restructured_data = {}
            for entry in original_data:
                column_name = entry["COLUMNNAME"]
                rid = entry["RID"]
                
                for key, value in entry.items():
                    if key.startswith("Y"):
                        year = int(key[1:5])
                        month = int(key[5:7])
                        
                        # Map month number to month name and quarter
                        month_info = {
                            3: ("Mar", "Q4"),
                            6: ("Jun", "Q1"),
                            9: ("Sep", "Q2"),
                            12: ("Dec", "Q3")
                        }
                        if month not in month_info:
                            continue  # Skip if not a quarter-end month
                        
                        month_name, quarter = month_info[month]
                        
                        # Calculate the financial year
                        if month <= 3:
                            financial_year = f"{year-1}-{year}"
                        else:
                            financial_year = f"{year}-{year+1}"
                        
                        # Format the date and add financial year info
                        date_key = f"{month_name} {year}"
                        fy_info = f"{quarter} of FY {financial_year}"
                        full_key = f"Quarterly Results {date_key} ({fy_info})"
                        
                        if full_key not in restructured_data:
                            restructured_data[full_key] = {}
                        
                        restructured_data[full_key][column_name] = value
            
            # Sort the data by date in reverse order
            def sort_key(x):
                # Use regular expression to extract year and month
                match = re.search(r'(\w+)\s(\d{4})', x)
                if match:
                    month, year = match.groups()
                    year = int(year)
                    month_order = {'Mar': 3, 'Dec': 2, 'Sep': 1, 'Jun': 0}
                    return (-year, -month_order[month])
                return (0, 0)  # Default return if the pattern doesn't match
            
            sorted_dates = sorted(restructured_data.keys(), key=sort_key)
            
            sorted_data = {}
            for date in sorted_dates:
                sorted_data[date] = restructured_data[date]
            
            # Convert the restructured and sorted data to JSON format
            json_data = json.dumps(sorted_data, indent=4)
            return json_data
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code} - {response.text}")
            return None
        

def get_cashflow(stock_name):
    stock_code = find_stock_code(stock_name)
    if stock_code == 0:
        return "The stock you inquired about is not part of the Nifty 50 index, ONLY ASK QUESTIONS ABOUT NIFTY 50 STOCKS atm."

    url = f"http://airrchipapis.cmots.com/api/CashFlow/{stock_code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

    data = response.json()
    if 'data' not in data or data['data'] is None:
        print("Error: No data found in the response.")
        return None

    columns_to_remove = {'None', 'Adjustments : ', 'Dividend Received 1', 'Dividend Received 2', 'Others 1',
                         'Others 2', 'Others 3', 'Others 4', 'Others 5', 'Others 6', 'Excess Depreciation W/b',
                         'Premium on Lease of land', 'Payment Towards VRS', "Prior Year's Taxation",
                         'Gain on Forex Exch. Transactions', 'Others 4', 'Capital Subsidy Received',
                         'Investment in Subsidiaires', 'Investment in Group Cos.', 'Issue of Shares on Acquisition of Cos.',
                         'Cancellation of Investment in Cos. Acquired', 'Inter Corporate Deposits', 'Share Application Money',
                         'Shelter Assistance Reserve', 'Others 5', 'Cash Flow From Operating Activities'}

    filtered_data = [entry for entry in data.get('data', []) 
                     if entry.get('COLUMNNAME') is not None 
                     and entry.get('COLUMNNAME', '').strip() not in columns_to_remove]

    data['data'] = filtered_data

    modified_data = {}
    for data_entry in data['data']:
        asset_type = data_entry.get('COLUMNNAME', '').rstrip(':')
        for key, value in data_entry.items():
            if key.startswith('Y20'):
                year = key[1:5]
                if year in ('2020', '2021', '2022', '2023'):
                    modified_key = f"CashFlow_{year} in crs"
                    if modified_key not in modified_data:
                        modified_data[modified_key] = {}
                    if value != 0.0:
                        modified_data[modified_key][asset_type] = value

    json_output = json.dumps(modified_data, indent=4)
    return json_output

def share_holding_pattern(stock_name):

    code=int(find_stock_code(stock_name))
    headers = {
        'Authorization': f'Bearer {token}'
    }
    url = f'http://airrchipapis.cmots.com/api/ShareholdingMorethanonePerDetails/{code}'
    response = requests.get(url, headers=headers)
    result=response.json()
    
    shareholders = {}
    for ratio in result['data']:
        shareholders[ratio['name']] = {
            'type': ratio['type'],
            'percentage_stake_holding': ratio['PercentageStakeHolding']
        }
    
    return json.dumps({"share_holding_pattern" : shareholders})

def cashflowratios(stock_name):

    code=int(find_stock_code(stock_name))
    headers = {
        'Authorization': f'Bearer {token}'
    }
    url = f'http://airrchipapis.cmots.com/api/CashFlowRatios/{code}/S'
    response = requests.get(url, headers=headers)
    data=response.json()
    result = {}
    for item in data["data"]:
        year = str(int(item["YRC"]))[:4]
        cash_flow_ratios = {
            "CashFlowPerShare": item["CashFlowPerShare"],
            "PricetoCashFlowRatio": item["PricetoCashFlowRatio"],
            "FreeCashFlowperShare": item["FreeCashFlowperShare"],
            "PricetoFreeCashFlow": item["PricetoFreeCashFlow"],
            "FreeCashFlowYield": item["FreeCashFlowYield"],
            "Salestocashflowratio": item["Salestocashflowratio"]
        }
        result[f"CashFlowRatios_{year}"] = cash_flow_ratios
    return json.dumps(result)

# def get_all_results(stock_name):
#     return (
#         json.loads(get_company_profile(stock_name))
#         # |json.loads(get_company_history(stock_name))
#         |json.loads(get_yearly_results(stock_name))
#         |json.loads(half_yearly_results(stock_name))
#         |json.loads(get_quarterly_results(stock_name))
#         |json.loads(nine_month_results(stock_name))
#         |json.loads(get_balance_sheet(stock_name))
#         |json.loads(get_profit_loss(stock_name))
#         |json.loads(get_daily_ratios(stock_name))
#         |json.loads(cashflowratios(stock_name))
#         |json.loads(get_cashflow(stock_name))
#         # |json.loads(share_holding_pattern(stock_name))
        
#     )

def get_all_results(stock_name):
    results = {}
    
    api_functions = [
        get_company_profile,
        get_yearly_results,
        #half_yearly_results,
        get_quarterly_results,
        #nine_month_results,
        get_balance_sheet,
        get_profit_loss,
        # get_daily_ratios,
        #cashflowratios,
        get_cashflow
    ]
    
    for func in api_functions:
        try:
            func_result = json.loads(func(stock_name))
            results.update(func_result)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
    
    return results
# def get_all_results(stock_name):
#     result = {}
#     result.update(get_company_profile(stock_name))
#     result.update(get_company_history(stock_name))
#     result.update(get_yearly_results(stock_name))
#     result.update(half_yearly_results(stock_name))
#     result.update(nine_month_results(stock_name))
#     result.update(get_balance_sheet(stock_name))
#     result.update(get_profit_loss(stock_name))
#     result.update(get_daily_ratios(stock_name))
#     return result
# def save_to_file(stock_name,file_type='json'):
#     results=get_all_results(stock_name)
#     if file_type == 'json':
#         with open(f'{stock_name}_results.json', 'w') as f:
#             json.dump(results, f, indent=4)
#     elif file_type == 'txt':
#         with open(f'{stock_name}_results.txt', 'w') as f:
#             f.write(json.dumps(results, indent=4))

def save_to_file_local(stock_name, file_type='json'):
    results = get_all_results(stock_name)
    folder_path = 'apidata'  # specify the folder path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # create the folder if it doesn't exist
    if file_type == 'json':
        file_path = f'{folder_path}/{stock_name}_results.json'
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
    elif file_type == 'txt':
        file_path = f'{folder_path}/{stock_name}_results.txt'
        with open(file_path, 'w') as f:
            f.write(json.dumps(results, indent=4))
    return file_path


def save_to_file(stock_name, file_type='json'):
    results = get_all_results(stock_name)
    if file_type == 'json':
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f, indent=4)
            file_path = f.name
    elif file_type == 'txt':
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(json.dumps(results, indent=4))
            file_path = f.name
    return file_path


def collection_exists(collection_name):
    try:
        client.get_collection(collection_name)
        return True
    except Exception as e:
        return False
# def create_doc_embeddings(file_name,file_path):
#     persistent_client = chromadb.PersistentClient(path="chromadb_for_data")
    
#     if not collection_exists(file_name, persistent_client.list_collections()):
#         # data=get_data()
#         loader = JSONLoader(file_path=file_path, jq_schema=".", text_content=False)
#         with open(file_path, 'r') as file:
#             data = json.load(file)
#         document = loader.load()
#         splitter = RecursiveJsonSplitter(max_chunk_size=300)
#         # docs=splitter.split_text(json_data=document, convert_lists = True)
        
#         docs = splitter.create_documents(texts=[data])

        

#         collection = persistent_client.get_or_create_collection(name=file_name)
#         count = collection.count()

#         for index, doc in enumerate(docs):
#             count += 1
#             collection.add(
#                 documents=str(doc.page_content),
#                 ids=[str(file_name) + str(count)]
#             )
        
#         return {"message": "Collection created", "collection_name": file_name}
#     else:
#         return {"message": "Collection already exists", "collection_name": file_name}

def create_chunks(file_path):
    class Document1:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

        def __repr__(self):
            return f"Document1(page_content='{self.page_content}', metadata={self.metadata})"

    with open(f"{file_path}","r") as file:
        data = json.loads(file.read())

    sub_dicts = {}

    # Iterate over each key-value pair in the main dictionary
    for key, value in data.items():
        # Assign the key to the new sub-dictionary
        sub_dicts[key] = value

    documents = []
    for key, sub_dict in data.items():
        # Convert the sub-dictionary to a string representation
        sub_dict_str = ', '.join(f'{k}: {v}' for k, v in sub_dict.items())
        
        # Construct the text representation with both key and sub-dictionary
        text = f"{key} :{sub_dict_str}"
        mt_data = {'key': key}
        document = Document1(page_content=text, metadata=mt_data)
        documents.append(document)
    return documents

def get_collection_name(stock_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv("csv_data\company_codes2.csv")

    # Extract the company names and codes from the DataFrame
    company_names = df["Company Name"].tolist()
    company_codes = df["Company Code"].tolist()
    
    threshold=80
    # Use fuzzy matching to find the closest match to the input stock name
    match = process.extractOne(stock_name, company_names)
   
    if match and match[1] >= threshold:
        # Get the index of the closest match
        idx = company_names.index(match[0])
        # Return the company name and code as a string
        return f"stock_{company_codes[idx]}"
    else:
        return "Not Found"
    
def create_doc_embeddings(stock_name):
    collection_name = get_collection_name(stock_name)
    print(collection_name)
    
    # Check if the collection already exists
    if collection_exists(collection_name):
        print(f"Collection {collection_name} already exists. Skipping creation.")
        return {"message": "Collection already exists", "collection_name": collection_name}
    
    file_path = save_to_file(stock_name, file_type='json')
    documents = create_chunks(file_path)

    collection = client.create_collection(name=collection_name, embedding_function=default_ef)
    count = collection.count()

    for index, doc in enumerate(documents):
        count += 1
        collection.add(
            documents=str(doc.page_content),
            ids=[str(collection_name) + str(count)]
        )
    print(f"Embedding created for {stock_name}")
    return {"message": "Collection created", "collection_name": collection_name}

#deletes the previous collection nad creates new
def update_doc_embeddings(stock_name):
    
    collection_name=get_collection_name(stock_name)    
    file_path=save_to_file(stock_name,file_type='json')
    print(file_path)
    documents=create_chunks(file_path)

    try:
        collection = client.get_collection(collection_name)
        # If the collection exists, delete it
        client.delete_collection(collection_name)
    except Exception as e:
        # If the collection doesn't exist, ignore the error and continue
        print(f"Collection {collection_name} doesn't exist. Creating a new one.")

    collection = client.create_collection(name=collection_name, embedding_function=default_ef)
    # collection = persistent_client.get_or_create_collection(name=collection_name)
    count = collection.count()

    for index, doc in enumerate(documents):
        count += 1
        collection.add(
            documents=str(doc.page_content),
            ids=[str(collection_name) + str(count)]
        )
    
#     return {"message": "Collection created", "collection_name": collection_name}
    # else:
    #     return {"message": "Collection already exists", "collection_name": collection_name}

from itertools import islice
@router.post("/create_embeddings2")
async def create_embeddings():
    try:
        start_index = 1175  # Define the starting index
        
        with open('csv_data/company_codes2.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            
            # Use islice to skip to the desired starting index
            company_names = [row[0] for row in islice(reader, start_index, None)]
            print(company_names)
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
            executor.map(create_doc_embeddings, company_names)
        
        return Response(status_code=200, content="All collections created")
    except Exception as e:
        return Response(status_code=500, content=f"Error: {str(e)}")

# @router.post("/create_embeddings2")
# async def create_embeddings():
#     try:
#         batch_size = 1000  # Define your batch size
#         company_names = []
#         batches = []

#         with open('company_codes2.csv', 'r') as file:
#             reader = csv.reader(file)
#             next(reader)  # skip header

#             for row in reader:
#                 company_names.append(row[0])
#                 if len(company_names) == batch_size:
#                     batches.append(company_names)
#                     company_names = []  # Reset the list for the next batch

#             # Add any remaining companies in the last batch
#             if company_names:
#                 batches.append(company_names)

#         with ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
#             executor.map(create_doc_embeddings, batches)

#         return Response(status_code=200, content="All collections created")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# @router.get("/create_embeddings2")
# async def create_embeddings():
#     try:
#         with open('company_codes2.csv', 'r') as file:
#             reader = csv.reader(file)
#             next(reader)  # skip header
#             company_names = [row[0] for row in reader]
#             print(company_names)
        
#         with ThreadPoolExecutor(max_workers=os.cpu_count()-2) as executor:
#             executor.map(create_doc_embeddings, company_names)
        
#         return Response(status_code=200, content="All collections created")
#     except Exception as e:
#         return Response(status_code=500, content=f"Error: {str(e)}")