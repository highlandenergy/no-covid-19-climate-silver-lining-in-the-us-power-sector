import pandas as pd
import utils_eia923 as utils_eia923
import os, sys

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

cols = ['Operator Id',
        'Plant Id',
        'Census Region',
        'Reported\nFuel Type Code',
        'Elec_MMBtu\nJanuary',
        'Elec_MMBtu\nFebruary',
        'Elec_MMBtu\nMarch',
        'Elec_MMBtu\nApril',
        'Elec_MMBtu\nMay',
        'Elec_MMBtu\nJune',
        'Elec_MMBtu\nJuly',
        'Elec_MMBtu\nAugust',
        'Elec_MMBtu\nSeptember',
        'Elec_MMBtu\nOctober',
        'Elec_MMBtu\nNovember',
        'Elec_MMBtu\nDecember',
        'Netgen\nJanuary',
        'Netgen\nFebruary',
        'Netgen\nMarch',
        'Netgen\nApril',
        'Netgen\nMay',
        'Netgen\nJune',
        'Netgen\nJuly',
        'Netgen\nAugust',
        'Netgen\nSeptember',
        'Netgen\nOctober',
        'Netgen\nNovember',
        'Netgen\nDecember'
        ]

# load all raw excel files
eia923_2015_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                      'EIA923_Schedules_2_3_4_5_M_12_2015_Final_Revision.xlsx')
df_2015 = pd.read_excel(eia923_2015_path, skiprows=5, usecols=cols)
df_2015 = df_2015[df_2015['Census Region'] != 'PACN']
print('Read 2015')

eia923_2016_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                      'EIA923_Schedules_2_3_4_5_M_12_2016_Final_Revision.xlsx')
df_2016 = pd.read_excel(eia923_2016_path, skiprows=5, usecols=cols)
df_2016 = df_2016[df_2016['Census Region'] != 'PACN']
print('Read 2016')
eia923_2017_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                      'EIA923_Schedules_2_3_4_5_M_12_2017_Final_Revision.xlsx')
df_2017 = pd.read_excel(eia923_2017_path, skiprows=5, usecols=cols)
df_2017 = df_2017[df_2017['Census Region'] != 'PACN']
print('Read 2017')
eia923_2018_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                      'EIA923_Schedules_2_3_4_5_M_12_2018_Final_Revision.xlsx')
df_2018 = pd.read_excel(eia923_2018_path, skiprows=5, usecols=cols)
df_2018 = df_2018[df_2018['Census Region'] != 'PACN']
print('Read 2018')
eia923_2019_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                      'EIA923_Schedules_2_3_4_5_M_12_2019_Final_Revision.xlsx')
df_2019 = pd.read_excel(eia923_2019_path, skiprows=5, usecols=cols)
df_2019 = df_2019[df_2019['Census Region'] != 'PACN']
print('Read 2019')
eia923_2020_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                      'EIA923_Schedules_2_3_4_5_M_12_2020_19FEB2021.xlsx')
df_2020 = pd.read_excel(eia923_2020_path, skiprows=5, usecols=cols)
df_2020 = df_2020[df_2020['Census Region'] != 'PACN']
print('Read 2020')

# go through each row and calculate carbon emissions and record net generation data
all_dfs_summed = []
for df in [df_2015, df_2016, df_2017, df_2018, df_2019, df_2020]:
    emissions_df = utils_eia923.calc_emissions(df)
    emissions_df = emissions_df.drop(columns=['Reported Fuel Type Code', 'Fuel Category'])
    all_dfs_summed.append(emissions_df.sum(axis=0, skipna=True))

# rearrange into a final df where each row is a year
data_list = []
for row in all_dfs_summed:
    series_list = row.tolist()
    data_list.append(series_list)
df_final = pd.DataFrame(data_list)
df_final.columns = all_dfs_summed[0].index
df_final['Year'] = ['2015', '2016', '2017', '2018', '2019', '2020']
df_final.set_index('Year', inplace=True)

# calculate C/E columns
df_final['KG_CO2/MWh_Jan'] = df_final['co2_emissions_in_kg_of_co2_Jan'] / df_final['net_generation_Jan']
df_final['KG_CO2/MWh_Feb'] = df_final['co2_emissions_in_kg_of_co2_Feb'] / df_final['net_generation_Feb']
df_final['KG_CO2/MWh_Mar'] = df_final['co2_emissions_in_kg_of_co2_Mar'] / df_final['net_generation_Mar']
df_final['KG_CO2/MWh_Apr'] = df_final['co2_emissions_in_kg_of_co2_Apr'] / df_final['net_generation_Apr']
df_final['KG_CO2/MWh_May'] = df_final['co2_emissions_in_kg_of_co2_May'] / df_final['net_generation_May']
df_final['KG_CO2/MWh_Jun'] = df_final['co2_emissions_in_kg_of_co2_Jun'] / df_final['net_generation_Jun']
df_final['KG_CO2/MWh_Jul'] = df_final['co2_emissions_in_kg_of_co2_Jul'] / df_final['net_generation_Jul']
df_final['KG_CO2/MWh_Aug'] = df_final['co2_emissions_in_kg_of_co2_Aug'] / df_final['net_generation_Aug']
df_final['KG_CO2/MWh_Sep'] = df_final['co2_emissions_in_kg_of_co2_Sep'] / df_final['net_generation_Sep']
df_final['KG_CO2/MWh_Oct'] = df_final['co2_emissions_in_kg_of_co2_Oct'] / df_final['net_generation_Oct']
df_final['KG_CO2/MWh_Nov'] = df_final['co2_emissions_in_kg_of_co2_Nov'] / df_final['net_generation_Nov']
df_final['KG_CO2/MWh_Dec'] = df_final['co2_emissions_in_kg_of_co2_Dec'] / df_final['net_generation_Dec']

df_final.to_pickle(os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'processed', 'eia923_monthly_c_and_c_over_e_and_e.p'))