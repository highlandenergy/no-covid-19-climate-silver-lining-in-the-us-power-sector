# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:21:52 2020

@author: 1000260103
"""
import pandas as pd
import numpy as np
from datetime import datetime

GENERATOR_DATA_PATH = "..\Data\Modified Gen. Characteristics.csv"
MKT_PRICES_DATA_PATH = "..\Data\Zone - Hist. En. Mkt. Prices.csv"

#--------------------------------------------------------------------------------

#STEO DATA for Counterfactual and Base case
#For counterfactual - 
# STEO_DATA_PATH = "..\Data\For. En. Mkt. Prices (STEO Jan 2020).csv"

#For Base - 
STEO_DATA_PATH = "..\Data\For. En. Mkt. Prices (STEO Jan 2021).csv"

#--------------------------------------------------------------------------------

#OUTPUT PATHS GIVEN BELOW - 

#Forecasted Market Prices for Counterfactual and Base case
#For counterfactual -
# FOR_MKT_PRICES_DATA_PATH = "..\Output\Gen. - For. Energy Mkt. Prices (Counterfactual).csv"

#For Base case -
FOR_MKT_PRICES_DATA_PATH = "..\Output\Gen. - For. Energy Mkt. Prices (Base).csv"

#--------------------------------------------------------------------------------

def getGeneratorData():
    generator_data_path = GENERATOR_DATA_PATH
    cols = ['Unit Name ','Dispatching ISO ','Dispatching ISO Zonal Location','Operating Capacity (Owner - GSC) (MW)','Variable Costs ($/MWh)']
    df = pd.read_csv(generator_data_path,usecols = cols)
    return df

def getHistoricZonalMktPrice():
    mkt_price_path = MKT_PRICES_DATA_PATH
    mkt_prices = pd.read_csv(mkt_price_path,header=None)
    iso = mkt_prices.iloc[0,:][1:]
    zones = mkt_prices.iloc[1,:][1:]
    mkt_prices_1 = mkt_prices.iloc[2:,:]
    mkt_prices_1.set_index(mkt_prices_1.columns[0], inplace=True)  
    mkt_prices_1.columns = zones
    mkt_prices_1.index = pd.to_datetime(mkt_prices_1.index)
    return [mkt_prices_1,iso,zones]

def getSTEOData():
    dataPath = STEO_DATA_PATH
    df = pd.read_csv(dataPath)
    df.columns = ['Date', 'CAISO', 'ERCOT', 'New England', 'MISO', 'New York','PJM', 'SPP']
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def avgMonthlyPriceofHistoricYear(year, month,df):
    data = df[(df.index.month == month) & (df.index.year == year)]
    avg_price = data.astype(float).mean()
    return avg_price

def getShiftValue(f_price, h_price_2018, h_price_2019):
    shift = (f_price - h_price_2018) if (abs(h_price_2018-f_price)<abs(h_price_2019-f_price)) else (f_price - h_price_2019)
    year = 2018 if (abs(h_price_2018-f_price)<abs(h_price_2019-f_price)) else 2019
    return [shift,year]

def shiftMktPrices(shift_val,values,date,zonal_mkt_price,f_year):
    for index,row in values.items():
        hour = datetime(f_year, index.month, index.day, index.hour, index.minute, index.second)
        date.append(hour)
        new_val = shift_val + float(row)
        zonal_mkt_price.append(new_val)
        
def getISO_ZoneMapping(iso,zone):
    iso_zone = {}
    name = iso.values[0]
    to_add = []
    for i in range(len(iso.values)):
        if(iso.values[i]==name):
            to_add.append(zone.values[i])
        else:
            iso_zone[iso.values[i-1]] = to_add
            to_add = []
            name = iso.values[i]
            to_add.append(zone.values[i])
    if(i==len(iso.values)-1):
        iso_zone[iso.values[i]] = to_add
    return iso_zone

def getISO(z,iso_zone):
    for key,value in iso_zone.items():
        for items in value:
            if z in items:
                return key

generator_data = getGeneratorData()
res = getHistoricZonalMktPrice()
mkt_prices = res[0]
iso = res[1]
zone = res[2]
mapping = getISO_ZoneMapping(iso, zone)
steo_data = getSTEOData()
final_df = pd.DataFrame()

i = 0
for z in zone:
    print(z)
    i=i+1
    iso_name = getISO(z, mapping)
    date = []
    zonal_mkt_price = []
    for index1,row1 in steo_data.iterrows():
        f_month = row1['Date'].month
        f_year = row1['Date'].year
        f_price = row1[iso_name]
        monthly_mkt_price = ((mkt_prices[z]).astype(str)).replace(np.nan,'0.0')
        mod_mkt_price_values = pd.Series([x.replace(",", "") for x in monthly_mkt_price.values],index = monthly_mkt_price.index)
        price_values = mod_mkt_price_values.astype(float)
        h_price_2018 = avgMonthlyPriceofHistoricYear(2018, f_month,price_values)
        h_price_2019 = avgMonthlyPriceofHistoricYear(2019, f_month,price_values)
        result = getShiftValue(f_price, h_price_2018, h_price_2019)
        shift_val = result[0]
        shift_year = result[1]
        values = price_values[(price_values.index.month == f_month) & (price_values.index.year == shift_year)]
        shiftMktPrices(shift_val,values,date,zonal_mkt_price,f_year)
    
 
    x = pd.Series(zonal_mkt_price, index = date) 
    final_df = pd.concat([final_df, x.rename(zone)], axis=1)

final_df.columns = [iso,zone]
final_df.to_csv(FOR_MKT_PRICES_DATA_PATH)

