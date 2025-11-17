import pandas as pd


data=pd.read_csv("financial_regression.csv")


data=data.drop(["GDP","CPI","us_rates_%"],axis=1)
data=data.dropna()

