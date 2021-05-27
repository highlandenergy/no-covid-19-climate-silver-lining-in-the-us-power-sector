import numpy as np
import pandas as pd
import os, sys

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

# this is a function that helps users get lists of CO2 emissions and net generation in accordance with
# a given EIA923 database. It looks up and multiplies MMBtu measurements for a given fuel source by
# a look-up table to get CO2 emissions. It simply passes through net generation readings.

# setup - load CO2 emissions factors in (kg CO2/MMBtu)
# https://www.eia.gov/environment/emissions/co2_vol_mass.php
# https://www.eia.gov/electricity/annual/html/epa_a_03.html
# https://www.epa.gov/sites/production/files/2015-07/documents/emission-factors_2014.pdf
carbon_factors_df = pd.read_excel(os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'processed', 'eia_code_fuel_conversion.xlsx'))

def define_dict_of_lists():
    return {
        'Reported Fuel Type Code': [],
        'Fuel Category': [],
        
        'co2_emissions_in_kg_of_co2_Jan': [],
        'co2_emissions_in_kg_of_co2_Feb': [],
        'co2_emissions_in_kg_of_co2_Mar': [],
        'co2_emissions_in_kg_of_co2_Apr': [],
        'co2_emissions_in_kg_of_co2_May': [],
        'co2_emissions_in_kg_of_co2_Jun': [],
        'co2_emissions_in_kg_of_co2_Jul': [],
        'co2_emissions_in_kg_of_co2_Aug': [],
        'co2_emissions_in_kg_of_co2_Sep': [],
        'co2_emissions_in_kg_of_co2_Oct': [],
        'co2_emissions_in_kg_of_co2_Nov': [],
        'co2_emissions_in_kg_of_co2_Dec': [],

        'net_generation_Jan': [],
        'net_generation_Feb': [],
        'net_generation_Mar': [],
        'net_generation_Apr': [],
        'net_generation_May': [],
        'net_generation_Jun': [],
        'net_generation_Jul': [],
        'net_generation_Aug': [],
        'net_generation_Sep': [],
        'net_generation_Oct': [],
        'net_generation_Nov': [],
        'net_generation_Dec': []
    }


def record_processed_row(master_dict, row, factor):
    fuel_category_df = carbon_factors_df[carbon_factors_df['EIA Fuel Code'] == row['Reported\nFuel Type Code']]
    fuel_category = ''
    if not fuel_category_df.empty:
        fuel_category = fuel_category_df['Fuel Category'].values[0]

    master_dict['Reported Fuel Type Code'].append(row['Reported\nFuel Type Code'])
    master_dict['Fuel Category'].append(fuel_category)

    master_dict['net_generation_Jan'].append(float(row['Netgen\nJanuary']))
    master_dict['net_generation_Feb'].append(float(row['Netgen\nFebruary']))
    master_dict['net_generation_Mar'].append(float(row['Netgen\nMarch']))
    master_dict['net_generation_Apr'].append(float(row['Netgen\nApril']))
    master_dict['net_generation_May'].append(float(row['Netgen\nMay']))
    master_dict['net_generation_Jun'].append(float(row['Netgen\nJune']))
    master_dict['net_generation_Jul'].append(float(row['Netgen\nJuly']))
    master_dict['net_generation_Aug'].append(float(row['Netgen\nAugust']))
    master_dict['net_generation_Sep'].append(float(row['Netgen\nSeptember']))
    master_dict['net_generation_Oct'].append(float(row['Netgen\nOctober']))
    master_dict['net_generation_Nov'].append(float(row['Netgen\nNovember']))
    master_dict['net_generation_Dec'].append(float(row['Netgen\nDecember']))

    master_dict['co2_emissions_in_kg_of_co2_Jan'].append(float(row['Elec_MMBtu\nJanuary']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_Feb'].append(float(row['Elec_MMBtu\nFebruary']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_Mar'].append(float(row['Elec_MMBtu\nMarch']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_Apr'].append(float(row['Elec_MMBtu\nApril']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_May'].append(float(row['Elec_MMBtu\nMay']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_Jun'].append(float(row['Elec_MMBtu\nJune']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_Jul'].append(float(row['Elec_MMBtu\nJuly']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_Aug'].append(float(row['Elec_MMBtu\nAugust']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_Sep'].append(float(row['Elec_MMBtu\nSeptember']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_Oct'].append(float(row['Elec_MMBtu\nOctober']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_Nov'].append(float(row['Elec_MMBtu\nNovember']) * factor)
    master_dict['co2_emissions_in_kg_of_co2_Dec'].append(float(row['Elec_MMBtu\nDecember']) * factor)

    return master_dict


def calc_emissions(df):

    # setup
    master_dict = define_dict_of_lists()
    df = df.replace('.', 0)

    # for each row, get carbon factor corresponding to fuel type
    # and add row to new dataframe with carbon emissions calculations
    for index, row in df.iterrows():
        factor_df = carbon_factors_df[carbon_factors_df['EIA Fuel Code'] == row['Reported\nFuel Type Code']]['Factor (Kilograms of CO2 Per Million Btu)**']
        if (factor_df.empty):
            master_dict = record_processed_row(master_dict, row, 0.0)
        else:
            master_dict = record_processed_row(master_dict, row, factor_df.values[0])

    output_df = pd.DataFrame.from_dict(master_dict)

    return output_df

def rearrange_datetime_vs_fuel_source(df, year):

    # filter
    df_coal = df[df['Fuel Category'] == 'Coal']
    df_gas = df[df['Fuel Category'] == 'Gas']
    df_oil = df[df['Fuel Category'] == 'Oil']

    # add to list
    def add_to_list(df_fuel):
        output_list = []
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Jan'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Feb'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Mar'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Apr'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_May'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Jun'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Jul'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Aug'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Sep'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Oct'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Nov'].values))
        output_list.append(np.nansum(df_fuel['co2_emissions_in_kg_of_co2_Dec'].values))
        return output_list

    coal_list = add_to_list(df_coal)
    gas_list = add_to_list(df_gas)
    oil_list = add_to_list(df_oil)
    datetime_list = [pd.to_datetime(f"{month}/1/{year}", infer_datetime_format=False) for month in list(range(1,13))]

    output_dict = {
        'datetime': datetime_list,
        'Coal': coal_list,
        'Gas': gas_list,
        'Oil': oil_list
    }

    output_df = pd.DataFrame.from_dict(output_dict)
    output_df = output_df.set_index('datetime')

    return output_df


