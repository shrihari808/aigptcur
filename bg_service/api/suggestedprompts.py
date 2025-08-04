from fastapi import FastAPI, HTTPException
import pandas as pd
import os
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import os
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
import google.generativeai as genai
import uvicorn
import pandas as pd
import requests
import json
import os
import re
import datetime
import spacy
import google.generativeai as genai
from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI, Query, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
import random
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from starlette.status import HTTP_403_FORBIDDEN

load_dotenv(override=True)
app = FastAPI()
router=APIRouter()

AI_KEY=os.getenv('AI_KEY')
async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )

# Configure the Google AI model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#@app.get("/generate_prompts")
async def prompts(num: int = Query(5, ge=2, le=20)):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model
    generation_config = {
            "temperature": 0.5,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }

    safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ]

    model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
    
    prompt = [
        f"""metrics = 
        'MCAP',  # Total market value of a company's outstanding shares.
        'EV',  # Total value of the company, including debt and excluding cash. Often used to evaluate a company's acquisition cost.
        
        # Valuation Ratios
        'PE',  # Price to Earnings Ratio. It measures the price investors are willing to pay for each unit of earnings.
        'PBV',  # Price to Book Value. Compares the market value of the company's stock to its book value.
        'DIVYIELD',  # Dividend Yield. The annual dividend income per share as a percentage of the share price.
        'DividendPayout',  # Percentage of earnings distributed as dividends to shareholders.
        
        # Earnings Metrics
        'EPS',  # Earnings Per Share. The net income divided by the total number of outstanding shares.
        'BookValue',  # The total value of the company's assets minus liabilities, indicating net worth per share.
        
        # Profitability Ratios
        'ROA',  # Return on Assets. Measures how efficiently a company is using its assets to generate profit.
        'ROE',  # Return on Equity. Reflects the return generated on shareholders' equity.
        'ROCE',  # Return on Capital Employed. Indicates the efficiency with which a company uses its capital.
        'NetIncomeMargin',  # The ratio of net profit to total revenue, indicating overall profitability.
        'GrossIncomeMargin',  # The ratio of gross profit to total revenue, reflecting profitability before operating expenses.
        'EBITDA_Margin',  # The ratio of EBITDA to total revenue, indicating core profitability.
        
        # Operational Performance
        'AssetTurnover',  # Indicates how efficiently a company generates revenue from its assets.
        'CurrentRatio',  # A measure of a company's ability to meet its short-term liabilities with its short-term assets.
        'Debt_Equity',  # The ratio of total debt to total equity, indicating financial leverage.
        'FCF_Margin',  # Free Cash Flow Margin. The proportion of revenue that translates to free cash flow.
        'Sales_TotalAsset',  # The ratio of sales to total assets, indicating asset efficiency.
        'NetDebt_FCF',  # The ratio of net debt to free cash flow, indicating how many years it would take to repay net debt.
        'NetDebt_EBITDA',  # The ratio of net debt to EBITDA, measuring leverage and debt coverage.
        
        # Earnings and Operations
        'EBIT',  # Earnings Before Interest and Taxes. A measure of operating profit.
        'EBITDA',  # Earnings Before Interest, Taxes, Depreciation, and Amortization. A broader measure of operating profitability.
        'EV_Sales',  # Enterprise Value to Sales. A valuation metric comparing enterprise value to revenue.
        'EV_EBITDA',  # Enterprise Value to EBITDA. A valuation metric comparing enterprise value to EBITDA.
        
        # Equity and Shares
        'TotalShareHoldersEquity',  # The total equity value for shareholders.
        'SharesOutstanding',  # The total number of shares issued and available for trading.
        
        # Debt and Financials
        'ShorttermDebt',  # The total short-term debt obligations due within one year.
        'LongtermDebt',  # The total long-term debt obligations due after one year.
        'NetSales',  # Total revenue generated from sales of goods or services.
        'Netprofit',  # Total profit after all expenses, taxes, and extraordinary items.
        'AnnualDividend',  # The total dividends paid to shareholders in a year.
        'COGS',  # Cost of Goods Sold. Direct costs associated with producing goods or services.
        'RetainedEarnings'  # Earnings retained in the company for reinvestment, not distributed as dividends.
        
        # Financial Performance
        'Gross Sales/Income from operations': "Total sales or income from operations before any deductions or expenses.",
        'Net Sales/Income from operations': "Total sales after deductions for returns, discounts, and allowances.",
        'Total Income from operations (net)': "Total operating income after deductions.",
        'Total Expenses': "Total costs incurred in running the company's operations.",
        'Finance Costs': "Costs related to financing, such as interest on loans.",
        'Depreciation, amortization, and depletion expense': "Non-cash expense reflecting the reduction in value of assets over time.",
        'Cost of Sales': "Direct costs associated with producing goods or services.",
        'Employee Cost': "Expenses related to employee salaries and benefits.",
        'Net profit from Ordinary Activities After Tax': "Profit after taxes from normal business operations.",
        'Profit / (Loss) from Discontinued Operations': "Profit or loss from operations that are no longer continuing.",
        'Net Profit after tax for the Period': "Net profit after all taxes and expenses for the given period.",
        'Net Profit after taxes, minority interest and share of Profit/(Loss) of associates': "Net profit after accounting for taxes, minority interest, and associate companies' profit or loss.",
        
        # Debt Ratios
        'Debt Equity Ratio': "Ratio of total debt to total equity, indicating financial leverage.",
        'Debt Service Coverage Ratio': "Measures the company's ability to service its debt with its operating income.",
        
        # Equity
        'Equity': "Total equity owned by shareholders.",
        'Face Value': "Nominal value of a share as stated on the share certificate.",
        
        # Earnings
        'Earnings before Interest and Tax (EBIT)': "Operating profit before interest and tax expenses.",
        'Earnings before Interest, Taxes, Depreciation, and Amortization (EBITDA)': "EBIT plus depreciation, amortization, and depletion expenses.",
        
        # Earnings Per Share
        'EPS before Exceptional/Extraordinary items-Basic': "Basic earnings per share before exceptional or extraordinary items.",
        'EPS before Exceptional/Extraordinary items-Diluted': "Diluted earnings per share before exceptional or extraordinary items."

        'Revenue From Operations - Net',  # Net revenue from operations after excise duty/GST.
        'Total Revenue',  # Total revenue including other income.
        
        # Profitability Metrics
        'Profit Before Tax',  # Profit before accounting for taxes, indicating operating profitability.
        'Profit After Tax',  # Net profit after accounting for all taxes.
        
        # Expense Metrics
        'Total Expenses',  # Total costs incurred by the company.
        'Cost of Material Consumed',  # Cost of raw materials consumed in production.
        'Employee Benefits',  # Costs related to employee salaries and benefits.
        'Finance Costs',  # Interest and other costs related to financing.
        'Depreciation and Amortization',  # Depreciation and amortization expenses, indicating wear and tear on fixed assets.
        
        # Tax Metrics
        'Taxation',  # Total tax expense, including current and deferred taxes.
        
        # Changes in Inventory
        'Changes in Inventories',  # Change in inventory levels over a specific period.
        
        # Exceptional and Extraordinary Items
        'Exceptional Items Before Tax',  # Exceptional or one-time items affecting profitability.
        'Extraordinary Items Before Tax',  # Extraordinary items impacting pre-tax profit.

        # Cash at Beginning and End
        'Cash and Cash Equivalents at Beginning of the year',  # Starting cash balance at the beginning of the fiscal year.
        'Cash and Cash Equivalents at End of the year',  # Ending cash balance at the end of the fiscal year.
        
        # Operating Activities
        'Net Cash from Operating Activities',  # Total cash generated from core operating activities.
        'Cash Generated from/(used in) Operations',  # Cash flow generated or used in business operations.
        'Operating Profit before Working Capital Changes',  # Operating profit excluding working capital adjustments.
        'Trade & Other Receivables',  # Change in trade receivables, indicating the impact of customer credit on cash flow.
        'Inventories',  # Change in inventories, indicating the impact of inventory changes on cash flow.
        'Trade Payables',  # Change in trade payables, indicating the impact of accounts payable on cash flow.
        'Direct Taxes Paid',  # Total taxes paid during the period.
        
        # Investing Activities
        'Net Cash used in Investing Activities',  # Total cash flows from investing activities.
        'Purchased of Fixed Assets',  # Cash outflows for purchasing fixed assets, indicating capital expenditures.
        'Sale of Fixed Assets',  # Cash inflows from the sale of fixed assets.
        'Purchase of Investments',  # Cash outflows for buying investments.
        'Sale of Investments',  # Cash inflows from selling investments.
        'Interest Received',  # Cash inflows from interest income.
        
        # Financing Activities
        'Net Cash used in Financing Activities',  # Total cash flows from financing activities.
        'Proceeds from Issue of shares (incl. share premium)',  # Cash inflows from issuing new shares.
        'Proceed from Short-Term Borrowings',  # Cash inflows from short-term borrowing.
        'Of the Long-Term Borrowings',  # Cash used to repay long-term borrowing.
        'Dividend Paid',  # Total dividends paid to shareholders.
        'Interest Paid in Financing Activities',  # Total interest paid related to financing activities.
        'company profile', #provides complete information of the company.

        
        Use this above metrics to generate the prompts please shuffle the metrics dont take metrics in order. The prompts must be framed to retrieve information about metrics of a company.
        make the prompts with exact metric mentioned above dont include description in the prompt.
        example of a prompt: 
        1. Provide an overview of Microsoft's business.
        2. Provide Costco's Forward EV/EBITDA ratio.
        3. List India companies under 50 billion in market cap growing EPS more than 20%.
        4. What was Amazon's Amazon Web Services revenue last year?

        generate ONLY {num} such prompts each prompt should contains a stock/company name and add them to the list below. Strictly DO NOT REPEAT THE SAME Company  or any company for a new prompt. don't repeat the same gist/fundamental idea for any other prompt. Make the prompts unique in itself.
        Avoid prompts relating to: LIST and OVIERVIEW about something. 
        State the definiton on a new line after the prompt. So the prompt structure will look like: 
         "Prompt :What is the Debt to Equity Ratio of Axis Bank?
         *Definiton: one-liner explaning about the financial term *" 
         Give the output in json format like:
              
        Above mentioned structure is just an example, don't use the same test-case.
        This are the company names which you can take randomly and genereate prompts and definition.
        Adani Enterprises
        Adani Ports and SEZ
        Asian Paints
        Axis Bank
        Bajaj Auto
        Bajaj Finance
        Bajaj Finserv
        Bharti Airtel
        Bharat Petroleum
        Britannia Industries
        Cipla
        Coal India
        Divi's Laboratories
        Dr. Reddy's Laboratories
        Eicher Motors
        Grasim Industries
        HCL Technologies
        HDFC
        HDFC Bank
        HDFC Life
        Hero MotoCorp
        Hindalco Industries
        Hindustan Unilever
        ICICI Bank
        IndusInd Bank
        Infosys
        ITC
        JSW Steel
        Kotak Mahindra Bank
        Larsen & Toubro
        Mahindra & Mahindra
        Maruti Suzuki
        Nestle India
        NTPC
        Oil and Natural Gas Corporation
        Power Grid Corporation of India
        Reliance Industries
        SBI Life Insurance
        Shree Cement
        State Bank of India
        Sun Pharmaceutical Industries
        Tata Consumer Products
        Tata Motors
        Tata Steel
        Tata Consultancy Services
        Tech Mahindra
        Titan Company
        UltraTech Cement
        UPL
        Wipro
        

    """
    ]

    print(prompt) 
    response = model.generate_content(prompt)
    input_json=response.text.strip()

    # Parse the JSON string
    parsed_json = json.loads(input_json)

    # Output the parsed JSON to verify it's correct
    print(json.dumps(parsed_json, indent=2))
            
    return parsed_json
  
class InputNum(BaseModel):
    num: int

suggest_prompt = """
    Your job is to provide prompts based on metrics and stocks given below.
    GENERATE THIS MANY {num} prompts.
 
    Rules for generating prompts:
    1.use exact same metrics provided above dont modify metrics.
    2.use only one stock name in prompt.
    3.shuffle metrics ans stock names while generating prompts ,dont take them in order.


    Metrics:

    [
        'MCAP', 'EV', 'PE', 'PBV', 'DIVYIELD', 'DividendPayout', 'EPS', 'BookValue',
        'ROA', 'ROE', 'ROCE', 'EBIT', 'EBITDA', 'EV_Sales', 'EV_EBITDA', 'NetIncomeMargin',
        'GrossIncomeMargin', 'AssetTurnover', 'CurrentRatio', 'Debt_Equity', 'FCF_Margin',
        'Sales_TotalAsset', 'NetDebt_FCF', 'NetDebt_EBITDA', 'EBITDA_Margin', 'TotalShareHoldersEquity',
        'ShorttermDebt', 'LongtermDebt', 'SharesOutstanding', 'NetSales', 'Netprofit', 'AnnualDividend',
        'COGS', 'RetainedEarnings',
        'Non-Current Assets:', 'Fixed Assets', '   Property, Plant and Equipments', '   Right-of-Use Assets', '    Intangible Assets', '    Intangible Assets under Development', 'Capital Work in Progress', 'Non-current Investments ', '   Investment Properties', '   Investments in Subsidiaries, Associates and Joint venture', '   Investments of Life Insurance Business', '   Investments - Long-term', 'Long-term Loans and Advances', 'Other Non-Current Assets', 'Long-term Loans and Advances and Other Non-Current Assets ', '   Biological Assets other than Bearer Plants (Non Current)', '   Loans - Long-term', '   Others Financial Assets - Long-term', '   Current Tax Assets - Long-term', '   Insurance Related Assets (Non Current)', '   Other Non-current Assets (LT)', 'Deferred Tax Assets', 'Total Non Current Assets', 'Current Assets:', 'Inventories', 'Biological Assets other than Bearer Plants (Current)', 'Current Investments', 'Cash and Cash Equivalents ', '   Cash and Cash Equivalents', '   Bank Balances Other Than Cash and Cash Equivalents', 'Trade Receivables', 'Short-term Loans and Advances', 'Other Current Assets', 'Short-term Loans and Advances and Other Current Assets ', '   Loans - Short-term', '   Others Financial Assets - Short-term', '   Current Tax Assets - Short-term', '   Insurance Related Assets (Current)', '   Other Current Assets (ST)', '   Assets Classified as Held for Sale', 'Total Current Assets', 'TOTAL ASSETS', 'Current Liabilities:', 'Short term Borrowings', 'Lease Liabilities (Current)', 'Trade Payables', 'Other Current Liabilities ', '   Others Financial Liabilities - Short-term', '   Insurance Related Liabilities (Current)', '   Other Current Liabilities', '   Liabilities Directly Associated with Assets Classified as Held for Sale', 'Provisions ', '   Current Tax Liabilities - Short-term', '   Other Short term Provisions', 'Total Current Liabilities', 'Net Current Asset', 'Non-Current Liabilities:', 'Long term Borrowings ', '   Debt Securities', '   Borrowings', '   Deposits', 'Lease Liabilities (Non Current)', 'Other Long term Liabilities ', '   Others Financial Liabilities - Long-term', '   Insurance Related Liabilities (Non Current)', '   Other Non-Current Liabilities', 'Long term Provisions ', '   Current Tax Liabilities - Long-term', '   Other Long term Provisions', 'Deferred Tax Liabilities', 'Total Non Current Liabilities', 'Shareholders’ Funds:', 'Share Capital ', '   Equity Capital', '   Preference Capital', '   Unclassified Capital', 'Other Equity ', '   Reserves and Surplus', '   Other Equity Components', "Total Shareholder's Fund", 'Total Equity', 'TOTAL EQUITY AND LIABILITIES', 'Contingent Liabilities and Commitments (to the Extent Not Provided for)', 'Ordinary Shares :', 'Authorised:', 'Number of Equity Shares - Authorised', 'Amount of Equity Shares - Authorised', 'Par Value of Authorised Shares', 'Susbcribed & fully Paid up :', 'Par Value', 'Susbcribed & fully Paid up Shares', 'Susbcribed & fully Paid up CapItal', 'Cash and Cash Equivalents at Beginning of the year', 'Cash Flow From Operating Activities', 'Net Profit before Tax & Extraordinary Items', 'Adjustments : ', 'Depreciation', 'Interest (Net)', 'Dividend Received 1', 'P/L on Sales of Assets', 'P/L on Sales of Invest', 'Prov. & W/O (Net)', 'P/L in Forex', 'Fin. Lease & Rental Charges', 'Others 1', 'Total Adjustments (PBT & Extraordinary Items)', 'Operating Profit before Working Capital Changes', 'Adjustments : ', 'Trade & 0ther Receivables', 'Inventories', 'Trade Payables', 'Loans & Advances', 'Investments', 'Net Stock on Hire', 'Leased Assets Net of Sale', 'Trade Bill(s) Purchased', 'Change in Borrowing', 'Change in Deposits', 'Others 2', 'Total Adjustments (OP before Working Capital Changes)', 'Cash Generated from/(used in) Operations', 'Adjustments : ', 'Interest Paid(Net)', 'Direct Taxes Paid', 'Advance Tax Paid', 'Others 3', 'Total Adjustments(Cash Generated from/(used in) Operations', 'Cash Flow before Extraordinary Items', 'Extraordinary Items :', 'Excess Depreciation W/b', 'Premium on Lease of land', 'Payment Towards VRS', "Prior Year's Taxation", 'Gain on Forex Exch. Transactions', 'Others 4', 'Total Extraordinary Items', 'Net Cash from Operating Activities', 'Cash Flow from Investing Activities', 'Investment in Assets :', 'Purchased of Fixed Assets', 'Capital Expenditure', 'Sale of Fixed Assets', 'Capital WIP', 'Capital Subsidy Received', 'Financial / Capital Investment :', 'Purchase of Investments', 'Sale of Investments', 'Investment Income', 'Interest Received', 'Dividend Received 2', 'Invest.In Subsidiaires', 'Loans to Subsidiaires', 'Investment in Group Cos.', 'Issue of Shares on Acquisition of Cos.', 'Cancellation of Investment in Cos. Acquired', 'Acquisition of Companies', 'Inter Corporate Deposits', 'Others 5', 'Net Cash used in Investing Activities', 'Cash Flow From Financing Activities', 'Proceeds :', 'Proceeds from Issue of shares (incl. share premium)', 'Proceed from Issue of Debentures', 'Proceed from 0ther Long Term Borrowings', 'Proceed from Bank Borrowings', 'Proceed from Short Tem Borrowings', 'Proceed from Deposits', 'Share Application Money', 'Cash/Capital Investment Subsidy', 'Loans from a Corporate Body', 'Payments :', 'Share Application Money Refund', 'On Redemption of Debenture', 'Of the Long Tem Borrowings', 'Of the Short Term Borrowings', 'Of Financial Liabilities', 'Dividend Paid', 'Shelter Assistance Reserve', 'Interest Paid in Financing Activities', 'Others 6', 'Net Cash used in Financing Activities', 'Net Inc./(Dec.) in Cash and Cash Equivalent', 'Cash and Cash Equivalents at End of the year', 'Gross Sales/Income from operations', 'Less: Excise duty', 'Net Sales/Income from operations', 'Other Operating Income', 'Total Income from operations (net)', 'Total Expenses', '    Cost of Sales', '    Employee Cost', '    Depreciation, amortization and depletion expense', '    Provisions & Write Offs', '    Administrative and Selling Expenses', '    Other Expenses', '    Pre Operation Expenses Capitalised', 'Profit from operations before other income, finance costs and exceptional items', 'Other Income', 'Profit from ordinary activities before finance costs and exceptional items', 'Finance Costs', 'Profit from ordinary activities after finance costs but before exceptional items', 'Exceptional Items', 'Other Adjustments Before Tax', 'Profit from ordinary activities before tax', 'Total Tax', 'Net profit from Ordinary Activities After Tax', 'Profit / (Loss) from Discontinued Operations', 'Net profit from Ordinary Activities/Discontinued Operations After Tax', 'Extraordinary items', 'Other Adjustments After Tax', 'Net Profit after tax for the Period', 'Other Comprehensive Income', 'Total Comprehensive Income', 'Equity', 'Reserve & Surplus', 'Face Value', 'EPS:', '    EPS before Exceptional/Extraordinary items-Basic', '    EPS before Exceptional/Extraordinary items-Diluted', '    EPS after Exceptional/Extraordinary items-Basic', '    EPS after Exceptional/Extraordinary items-Diluted', 'Book Value (Unit Curr.)', 'No. of Employees', 'Debt Equity Ratio', 'Debt Service Coverage Ratio', 'Interest Service Coverage Ratio', 'Debenture Redemption Reserve (Rs cr)', 'Paid up Debt Capital (Rs cr)'

    ]
    
    
    stocks:
    ["Adani Enterprises", "Adani Ports and SEZ", "Asian Paints", "Axis Bank", "Bajaj Auto"],
    ["Bajaj Finance", "Bajaj Finserv", "Bharti Airtel", "Bharat Petroleum", "Britannia Industries"],
    ["Cipla", "Coal India", "Divi's Laboratories", "Dr. Reddy's Laboratories", "Eicher Motors"],
    ["Grasim Industries", "HCL Technologies", "HDFC", "HDFC Bank", "HDFC Life"],
    ["Hero MotoCorp", "Hindalco Industries", "Hindustan Unilever", "ICICI Bank", "IndusInd Bank"],
    ["Infosys", "ITC", "JSW Steel", "Kotak Mahindra Bank", "Larsen & Toubro"],
    ["Mahindra & Mahindra", "Maruti Suzuki", "Nestle India", "NTPC", "Oil and Natural Gas Corporation"],
    ["Power Grid Corporation of India", "Reliance Industries", "SBI Life Insurance", "Shree Cement", "State Bank of India"],
    ["Sun Pharmaceutical Industries", "Tata Consumer Products", "Tata Motors", "Tata Steel", "Tata Consultancy Services"],
    ["Tech Mahindra", "Titan Company", "UltraTech Cement", "UPL", "Wipro"]


    The output should be in json format and it should contain {num} prompts.
        "Prompt":prompt generated 
        "Definition": One line explanation of the metric mentioned in the prompt.
        Make sure to provide the output in the same order as the input json objects.
    
    
    """


llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-1106")
R_prompt = PromptTemplate(template=suggest_prompt,input_variables=["num"])
llm_chain_res= LLMChain(prompt=R_prompt, llm=llm)
chain = R_prompt | llm | JsonOutputParser()

@router.get("/generate_prompts")
def get_prompts(num: int=Query(5, ge=2, le=20),ai_key_auth: str = Depends(authenticate_ai_key)):
    if not 2 <= num <= 20:
        raise HTTPException(status_code=400, detail="Number must be between 2 and 20")
    
    input_data = {"num":num}
    #res=llm_chain_res.predict(query=q)
    res=chain.invoke(input_data)
    print(res)
    return res