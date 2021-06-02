# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 23:14:28 2020

@author: 1000260103
"""
import pandas as pd
import numpy as np

GENERATOR_DATA_PATH = "..\Data\Modified Gen. Characteristics.csv"

#--------------------------------------------------------------------------------
# MKT_PRICES_DATA_PATH = "..\Data\Zone - Hist. En. Mkt. Prices.csv"

# Once forecasted market price files for both Base and Counterfactual are creates replace MKT_PRICES_DATA_PATH
MKT_PRICES_DATA_PATH = "..\Output\Gen. - For. Energy Mkt. Prices (Counterfactual).csv"
# MKT_PRICES_DATA_PATH = "..\Output\Gen. - For. Energy Mkt. Prices (Base).csv"

#--------------------------------------------------------------------------------

#OUTPUT FILE PATHS GIVEN BELOW -

#For Revenue 
# MKT_REVENUE_DATA_PATH = "..\Output\Gen. - Hist. Energy Mkt. Rev.csv"
MKT_REVENUE_DATA_PATH = "..\Output\Gen. - For. Rev. (Counterfactual).csv"
# MKT_REVENUE_DATA_PATH = "..\Output\Gen. - For. Rev. (Base).csv"

#--------------------------------------------------------------------------------

#For costs 
# COSTS_DATA_PATH = "..\Output\Gen. - Hist. Variable Costs.csv"
COSTS_DATA_PATH = "..\Output\Gen. - For. Costs (Counterfactual).csv"
# COSTS_DATA_PATH = "..\Output\Gen. - For. Costs (Base).csv"



def getGeneratorData():
    generator_data_path = GENERATOR_DATA_PATH
    cols = ['Unit Name ','Dispatching ISO Zonal Location','Operating Capacity (Owner - GSC) (MW)','Variable Costs ($/MWh)']
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
    return mkt_prices_1

def getNoOfMonths(mkt_prices_1):
    mkt_prices_1.index = pd.to_datetime(mkt_prices_1.index)
    month_info = mkt_prices_1.resample('MS').sum().index
    return month_info

def getNetRevenue(price_values,vom_cost):
    revenue = 0
    op_hrs = 0
    for price in price_values:
        if price > vom_cost:
            revenue = revenue + price
            op_hrs = op_hrs + 1
    return [revenue,op_hrs]

def getVariableCost(vom_cost,op_hrs):
    return vom_cost * op_hrs


generator_data = getGeneratorData()
mkt_prices = getHistoricZonalMktPrice()
months = getNoOfMonths(mkt_prices)
i=0

revenue_list = []
cost_list = []
unit_list = []

for index,row in generator_data.iterrows():
    i=i+1
    unit_name = row['Unit Name ']
    zone = row['Dispatching ISO Zonal Location']
    vom_cost = row['Variable Costs ($/MWh)']
    op_cost = row['Operating Capacity (Owner - GSC) (MW)']
    r = []
    c = []
    if(zone in mkt_prices.columns):
        for date in months:
            monthly_mkt_price = mkt_prices[(pd.to_datetime(mkt_prices.index).month == date.month) & (pd.to_datetime(mkt_prices.index).year == date.year)][zone].astype(str)
            monthly_mkt_price = monthly_mkt_price.replace(np.nan,'0.0')
            mod_mkt_price_values = [x.replace(",", "") for x in monthly_mkt_price.values]
            price_values = list(map(float, mod_mkt_price_values))
            price_values.sort(reverse=True)
            revenue_data = getNetRevenue(price_values,vom_cost)
            gross_revenue = revenue_data[0]*op_cost
            op_hrs = revenue_data[1]
            variable_cost = (getVariableCost(vom_cost,op_hrs))*op_cost
            net_revenue = (gross_revenue - variable_cost)
            r.append(net_revenue)
            c.append(variable_cost)
        revenue_list.append(r)
        cost_list.append(c)
        unit_list.append(unit_name)
        print(i)
    else:
        for date in months:
            r.append(0.0)
            c.append(0.0)
        revenue_list.append(r)
        cost_list.append(c)
        unit_list.append(unit_name)
        print(str(i)+" NOT FOUND Zone: "+zone)

        
month_names = [str(k.day)+'-'+str(k.strftime("%b"))+'-'+str(k.year) for k in months ]        
revenue_df = pd.DataFrame(revenue_list,columns=month_names)
revenue_df.insert(loc=0, column='Unit Name', value=unit_list)
revenue_df.to_csv(MKT_REVENUE_DATA_PATH,index=False) 

variable_cost_df = pd.DataFrame(cost_list,columns=month_names)
variable_cost_df.insert(loc=0, column='Unit Name', value=unit_list) 
variable_cost_df.to_csv(COSTS_DATA_PATH,index=False) 
