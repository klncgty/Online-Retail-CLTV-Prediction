import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)
from sklearn.preprocessing import MinMaxScaler

def outlier_tresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_tresholds(dataframe, variable):
    low_limit, up_limit = outlier_tresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit) , variable] = up_limit


df_ = pd.read_excel("CRM ödev/CLTV_Prediction/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()

df.isnull().sum()
df.dropna(inplace=True)
df= df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"]>0
df = df[df["Price"]>0]

replace_with_tresholds(df, "Price")
replace_with_tresholds(df, "Quantity")

df["Total_Price"] = df["Price"] * df["Quantity"]

today_date = dt.datetime(2011, 12, 11)

clyv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda date: ( date.max() - date.min()).days,
                                                          lambda date : (today_date - date.min()).days],
                                         "Invoice" : lambda num: num.nunique(),
                                          "Total_Price": lambda Total_Price: Total_Price.sum()})

clyv_df.columns = clyv_df.columns.droplevel[0]
clyv_df.columns = ["recency", "T", "frequency", "monetary"]

clyv_df["monetary"] = clyv_df["monetary"] / clyv_df["frequency"]

clyv_df.describe().T

clyv_df = clyv_df[(clyv_df["frequency"] > 1)]

clyv_df["recency"] = clyv_df["recency"] / 7
clyv_df["T"] = clyv_df["T"] / 7

#### BG/NBD MODEL #####

bgf = BetaGeoFitter(penalizer_coef= 0.001)

bgf.fit(clyv_df["frequency"],
        clyv_df["recency"],
        clyv_df["T"])


bgf.conditional_expected_number_of_purchase_up_to_time(1,
                                                       clyv_df["frequency"],
                                                       clyv_df["recency"],
                                                       clyv_df["T"].sort_values(ascending=False))


clyv_df["expected_purc_1_month"] = bgf.conditional_expected_number_of_purchase_up_to_time(4,
                                                       clyv_df["frequency"],
                                                       clyv_df["recency"],
                                                       clyv_df["T"].sort_values(ascending=False))



bgf.conditional_expected_number_of_purchase_up_to_time(4*3,
                                                       clyv_df["frequency"],
                                                       clyv_df["recency"],
                                                       clyv_df["T"].sort_values(ascending=False))


##### GAMMA-GAMMA MODEL #####

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(clyv_df["frequenc"], clyv_df["monetary"])

ggf.conditional_expected_average_profit(clyv_df["frequency"],
                                        clyv_df["monetary"]).head()


ggf.conditional_expected_average_profit(clyv_df["frequency"],
                                        clyv_df["monetary"]).head()










