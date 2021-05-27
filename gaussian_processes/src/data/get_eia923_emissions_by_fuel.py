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
eia923_2016_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                'EIA923_Schedules_2_3_4_5_M_12_2016_Final_Revision.xlsx')
eia923_2017_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                'EIA923_Schedules_2_3_4_5_M_12_2017_Final_Revision.xlsx')
eia923_2018_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                'EIA923_Schedules_2_3_4_5_M_12_2018_Final_Revision.xlsx')
eia923_2019_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                'EIA923_Schedules_2_3_4_5_M_12_2019_Final_Revision.xlsx')
eia923_2020_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                'EIA923_Schedules_2_3_4_5_M_12_2020_19FEB2021.xlsx')

df_2015 = pd.read_excel(eia923_2015_path, skiprows=5, usecols=cols)
df_2015 = df_2015[df_2015['Census Region'] != 'PACN']
df_2015_w_emissions = utils_eia923.calc_emissions(df_2015)
df_2015_to_be_combined = utils_eia923.rearrange_datetime_vs_fuel_source(df_2015_w_emissions, 2015)
print('Processed 2015')

df_2016 = pd.read_excel(eia923_2016_path, skiprows=5, usecols=cols)
df_2016 = df_2016[df_2016['Census Region'] != 'PACN']
df_2016_w_emissions = utils_eia923.calc_emissions(df_2016)
df_2016_to_be_combined = utils_eia923.rearrange_datetime_vs_fuel_source(df_2016_w_emissions, 2016)
print('Processed 2016')

df_2017 = pd.read_excel(eia923_2017_path, skiprows=5, usecols=cols)
df_2017 = df_2017[df_2017['Census Region'] != 'PACN']
df_2017_w_emissions = utils_eia923.calc_emissions(df_2017)
df_2017_to_be_combined = utils_eia923.rearrange_datetime_vs_fuel_source(df_2017_w_emissions, 2017)
print('Processed 2017')

df_2018 = pd.read_excel(eia923_2018_path, skiprows=5, usecols=cols)
df_2018 = df_2018[df_2018['Census Region'] != 'PACN']
df_2018_w_emissions = utils_eia923.calc_emissions(df_2018)
df_2018_to_be_combined = utils_eia923.rearrange_datetime_vs_fuel_source(df_2018_w_emissions, 2018)
print('Processed 2018')

df_2019 = pd.read_excel(eia923_2019_path, skiprows=5, usecols=cols)
df_2019 = df_2019[df_2019['Census Region'] != 'PACN']
df_2019_w_emissions = utils_eia923.calc_emissions(df_2019)
df_2019_to_be_combined = utils_eia923.rearrange_datetime_vs_fuel_source(df_2019_w_emissions, 2019)
print('Processed 2019')

df_2020 = pd.read_excel(eia923_2020_path, skiprows=5, usecols=cols)
df_2020 = df_2020[df_2020['Census Region'] != 'PACN']
df_2020_w_emissions = utils_eia923.calc_emissions(df_2020)
df_2020_to_be_combined = utils_eia923.rearrange_datetime_vs_fuel_source(df_2020_w_emissions, 2020)
print('Processed 2020')

eia923_emissions_by_fuel_df = pd.concat([df_2015_to_be_combined, df_2016_to_be_combined, df_2017_to_be_combined, df_2018_to_be_combined, df_2019_to_be_combined, df_2020_to_be_combined])

eia923_emissions_by_fuel_df.to_pickle(os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'processed', 'eia923_monthly_c_by_fuel.p'))
print('asdf')