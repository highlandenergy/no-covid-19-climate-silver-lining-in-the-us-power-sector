# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
from plotly.offline import init_notebook_mode, plot
from plotly.graph_objs import Scatter,Data
from plotly import graph_objects as go

GENERATOR_DATA_PATH = "..\Data\Modified Gen. Characteristics.csv"

#--------------------------------------------------------------------------------

#For Revenue 
# MKT_REVENUE_DATA_PATH = "..\Output\Gen. - Hist. Energy Mkt. Rev.csv"

#For Counterfactual
# MKT_REVENUE_DATA_PATH = "..\Output\Gen. - For. Rev. (Counterfactual).csv"

#For Base
MKT_REVENUE_DATA_PATH = "..\Output\Gen. - For. Rev. (Base).csv"

#--------------------------------------------------------------------------------

#OUTPUT FILE PATHS GIVEN BELOW -
# MKT_PROF_DATA_PATH = "..\Output\Hist. Mkt Prof.csv"
#For Counterfactual
# MKT_PROF_DATA_PATH = "..\Output\For. Mkt Prof (Counterfactual).csv"

#For Base
MKT_PROF_DATA_PATH = "..\Output\For. Mkt Prof (Base).csv"

#--------------------------------------------------------------------------------

def getGeneratorData():
    generator_data_path = GENERATOR_DATA_PATH;
    cols = ['Unit Name ','Dispatching ISO Zonal Location','Operating Capacity (Owner - GSC) (MW)','Variable Costs ($/MWh)','Fixed O&M Costs per kW-Year (Owner) ($/kW-year)']
    df = pd.read_csv(generator_data_path,usecols = cols)
    return df

def getZonalMktRev():
    mkt_rev_path = MKT_REVENUE_DATA_PATH
    mkt_rev = pd.read_csv(mkt_rev_path)
    x = mkt_rev.columns[1:]
    x = pd.to_datetime(x)
    return [mkt_rev,x]

def getFOM(gen_name,d_month,d_year,index):
    fom_perYear = generator_data.loc[generator_data.index == index ]['Fixed O&M Costs per kW-Year (Owner) ($/kW-year)'].values[0]
    op_cap = generator_data.loc[generator_data.index == index ]['Operating Capacity (Owner - GSC) (MW)'].values[0]
    fom = (fom_perYear * 1000 * op_cap*d_month)/d_year
    return fom

def getDaysInMonth(m,y):
    if(m<12):
        return abs((date(y, m, 1) - date(y, m+1, 1)).days)
    else:
        return abs((date(y, m, 1) - date(y+1, 1, 1)).days)

def getDaysInYear(y):
    return abs((date(y, 1, 1) - date(y+1,1, 1)).days)

def getMarketProfitForGenerator(gen_name,gen_mkt_rev,gen_mkt_prof,months,gen_index):
    k = 0
    for i in gen_mkt_rev:
        d_month = getDaysInMonth(months[k].month,months[k].year)
        d_year = getDaysInYear(months[k].year)
        fom = getFOM(gen_name,d_month,d_year,gen_index)
        price = i - fom
        gen_mkt_prof.append(price)
        k=k+1
    
generator_data = getGeneratorData()
res = getZonalMktRev()

hist_mkt_rev = res[0]
mkt_prof_values = []
months = pd.to_datetime(hist_mkt_rev.columns[1:])

for index,row in hist_mkt_rev.iterrows():
    print(index)
    gen_mkt_prof = []
    getMarketProfitForGenerator(row['Unit Name'],row[1:],gen_mkt_prof,months,index)
    mkt_prof_values.append(gen_mkt_prof)
    


file_name = MKT_PROF_DATA_PATH
mkt_prof_df = pd.DataFrame(mkt_prof_values,columns=hist_mkt_rev.columns[1:])
mkt_prof_df.insert(loc=0, column='Unit Name', value=hist_mkt_rev['Unit Name'])
mkt_prof_df.to_csv(file_name,index=False) 

