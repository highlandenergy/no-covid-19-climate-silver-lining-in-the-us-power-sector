import os, sys, pickle
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

import GPy_1_0_5

# date config
current_month = datetime(2021, 1, 19)
analysis_month_start = datetime(2020, 3, 1)
analysis_month_end = datetime(2020, 12, 31)
num_months = (analysis_month_end.year - analysis_month_start.year) * 12 + (
        analysis_month_end.month - analysis_month_start.month) + 1
remove_months = (current_month.year - analysis_month_end.year) * 12 + (current_month.month - analysis_month_end.month)
plot_start = datetime(2016, 1, 1)
ref_start = datetime(2015, 1, 1)
buffer = (plot_start.year - ref_start.year) * 12 + (plot_start.month - ref_start.month)

# path config
output_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'out')
os.makedirs(output_path, exist_ok=True)


def load_cooling_heating_degree_days():
    # https://www.eia.gov/outlooks/steo/data/browser/#/?v=28&f=M&s=&start=199701&end=202212&id=&map=&ctype=linechart&maptype=0&linechart=ZWCDPUS~ZWHDPUS
    dd_csv_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'processed',
                               'eia_heating_cooling_degree_days_20210120.csv')
    dd_df = pd.read_csv(dd_csv_path, skiprows=4)
    dd_df['Month'] = pd.to_datetime(dd_df['Month'])

    # plot cooling and heating degree days
    dd_df_plot = dd_df[dd_df['Month'] > plot_start]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    color = 'tab:blue'
    ax1.set_xlabel('Time (year)')
    ax1.set_ylabel('Cooling Degree Days', color=color)
    ax1.plot(dd_df_plot['Month'], dd_df_plot['Cooling Degree days: U.S. (cooling degree days) cooling degree days'],
             '.-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Heating Degree Days', color=color)
    ax2.plot(dd_df_plot['Month'], dd_df_plot['Heating Degree days: U.S. (heating degree days) heating degree days'],
             '.--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Population-Weighted Heating and Cooling Degree Days \n by Month in the Contiguous U.S.')
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, '000_coolingheatingdegreedays.pdf'))
    plt.close()

    return dd_df


def clean_gp_input(X, Y, X_optional_0=None, X_optional_1=None):
    # if there are duplicates in X, take the correspoding Ys, average them,
    # and then only return an X, Y pair with unique X values.

    # additionally, if there are X_optional_0 and X_optional_1 passed,
    # average them and only return an additional X_optional_0 and X_optional_1
    # with unique X values

    Y_new = []
    X = X.flatten()
    Y = Y.flatten()

    if X_optional_0 is not None:
        X_optional_0 = X_optional_0.flatten()
        X_optional_0_new = []
    if X_optional_1 is not None:
        X_optional_1 = X_optional_1.flatten()
        X_optional_1_new = []

    uniques, counts = np.unique(X, return_counts=True)

    for unique, count in zip(uniques, counts):
        y = np.mean(Y[np.where(X == unique)])
        Y_new.append(y)

        if X_optional_0 is not None:
            X_optional_0_add = np.mean(X_optional_0[np.where(X == unique)])
            X_optional_0_new.append(X_optional_0_add)
        if X_optional_1 is not None:
            X_optional_1_add = np.mean(X_optional_1[np.where(X == unique)])
            X_optional_1_new.append(X_optional_1_add)

    X_all = uniques[:, np.newaxis]

    if X_optional_0 is not None:
        X_optional_0_new = np.array(X_optional_0_new).reshape(len(X_optional_1_new), 1)
        X_all = np.hstack((X_all, X_optional_0_new))
    if X_optional_1 is not None:
        X_optional_1_new = np.array(X_optional_1_new).reshape(len(X_optional_1_new), 1)
        X_all = np.hstack((X_all, X_optional_1_new))

    return X_all, np.array(Y_new)[:, np.newaxis]


def annualize_monthly_x_y(X, Y):
    dates = pd.DatetimeIndex(X)
    days_in_month = dates.daysinmonth
    Y = Y / np.array(days_in_month) * 365.25

    return Y


def get_eia923_c(eia923_df):
    co2_emissions_col_list = [
        'co2_emissions_in_kg_of_co2_Jan',
        'co2_emissions_in_kg_of_co2_Feb',
        'co2_emissions_in_kg_of_co2_Mar',
        'co2_emissions_in_kg_of_co2_Apr',
        'co2_emissions_in_kg_of_co2_May',
        'co2_emissions_in_kg_of_co2_Jun',
        'co2_emissions_in_kg_of_co2_Jul',
        'co2_emissions_in_kg_of_co2_Aug',
        'co2_emissions_in_kg_of_co2_Sep',
        'co2_emissions_in_kg_of_co2_Oct',
        'co2_emissions_in_kg_of_co2_Nov',
        'co2_emissions_in_kg_of_co2_Dec'
    ]

    # mod_dict = {}
    years = ['2015', '2016', '2017', '2018', '2019', '2020']
    datetime_list = []
    co2_emissions_list = []

    for year in years:
        for month, col in enumerate(co2_emissions_col_list):
            datetime_list.append(pd.to_datetime(f"{month + 1}/1/{year}", infer_datetime_format=False))
            co2_emissions_list.append(eia923_df.loc[year][col])

    output_dict = {
        'Date': datetime_list,
        'Co2 Emissions (kg)': co2_emissions_list
    }

    output_df = pd.DataFrame.from_dict(output_dict)

    return output_df


def get_eia923_c_over_e(eia923_df):
    c_over_e_col_list = [
        'KG_CO2/MWh_Jan',
        'KG_CO2/MWh_Feb',
        'KG_CO2/MWh_Mar',
        'KG_CO2/MWh_Apr',
        'KG_CO2/MWh_May',
        'KG_CO2/MWh_Jun',
        'KG_CO2/MWh_Jul',
        'KG_CO2/MWh_Aug',
        'KG_CO2/MWh_Sep',
        'KG_CO2/MWh_Oct',
        'KG_CO2/MWh_Nov',
        'KG_CO2/MWh_Dec'
    ]

    years = ['2015', '2016', '2017', '2018', '2019', '2020']
    datetime_list = []
    c_over_e_list = []

    for year in years:
        for month, col in enumerate(c_over_e_col_list):
            datetime_list.append(pd.to_datetime(f"{month + 1}/1/{year}", infer_datetime_format=False))
            c_over_e_list.append(eia923_df.loc[year][col])

    output_dict = {
        'Date': datetime_list,
        'C/E (kg/MWh)': c_over_e_list
    }

    output_df = pd.DataFrame.from_dict(output_dict)

    return output_df


def get_eia923_e(eia923_df):
    e_col_list = [
        'net_generation_Jan',
        'net_generation_Feb',
        'net_generation_Mar',
        'net_generation_Apr',
        'net_generation_May',
        'net_generation_Jun',
        'net_generation_Jul',
        'net_generation_Aug',
        'net_generation_Sep',
        'net_generation_Oct',
        'net_generation_Nov',
        'net_generation_Dec'
    ]

    # mod_dict = {}
    years = ['2015', '2016', '2017', '2018', '2019', '2020']
    datetime_list = []
    e_list = []

    for year in years:
        for month, col in enumerate(e_col_list):
            datetime_list.append(pd.to_datetime(f"{month + 1}/1/{year}", infer_datetime_format=False))
            e_list.append(eia923_df.loc[year][col])

    output_dict = {
        'Date': datetime_list,
        'Net Generation (MWh)': e_list
    }

    output_df = pd.DataFrame.from_dict(output_dict)

    return output_df


def fit_3d_gp(X_fit_reg, Y_fit_reg):
    # fitting 3d GP
    # * linear and bias terms on HDD and CDD
    # * linear, bias, and std periodic terms on time
    k_lin_0 = GPy_1_0_5.kern.Linear(1, active_dims=[0])
    k_bias_0 = GPy_1_0_5.kern.Bias(1, active_dims=[0])
    k_std_per_0 = GPy_1_0_5.kern.StdPeriodic(1, period=0.2, lengthscale=0.25, active_dims=[0])
    k_lin_1 = GPy_1_0_5.kern.Linear(1, active_dims=[1])
    k_bias_1 = GPy_1_0_5.kern.Bias(1, active_dims=[1])
    k_lin_2 = GPy_1_0_5.kern.Linear(1, active_dims=[2])
    k_bias_2 = GPy_1_0_5.kern.Bias(1, active_dims=[2])
    kernel_all = k_lin_0 + k_bias_0 + k_std_per_0 + k_lin_1 + k_bias_1 + k_lin_2 + k_bias_2

    m = GPy_1_0_5.models.GPRegression(X_fit_reg, Y_fit_reg, kernel_all)
    m['sum.std_periodic.period'].constrain_bounded(0.1, 0.3)
    m.optimize(messages=True, max_f_eval=1000)
    m.optimize_restarts(num_restarts=5)

    return m


def run_3d_gp(m, X_fit_reg, X_float_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, Y_fit_reg):
    # 3D GP run, with real data
    plot_mean, plot_var = m.predict(X_fit_reg)
    plot_std = np.sqrt(plot_var)

    # compile all X's for prediction
    X_pred_reg = X_float_compare
    X_pred_reg = np.hstack((X_pred_reg, X_cdd_compare))
    X_pred_reg = np.hstack((X_pred_reg, X_hdd_compare))

    plot_pred_mean, plot_pred_var = m.predict(X_pred_reg)
    plot_pred_std = np.sqrt(plot_pred_var)

    # undo normalization
    plot_mean = plot_mean * (Y_max - Y_min) + Y_min
    plot_std = plot_std * (Y_max - Y_min)
    plot_pred_mean = plot_pred_mean * (Y_max - Y_min) + Y_min
    plot_pred_std = plot_pred_std * (Y_max - Y_min)
    Y_fit_reg_unnormalized = Y_fit_reg * (Y_max - Y_min) + Y_min

    return plot_std, plot_mean, plot_pred_std, plot_pred_mean, Y_fit_reg_unnormalized


def plot_3d_gp(X_fit_datetime_reg, X_compare, plot_std, plot_mean, plot_pred_std, plot_pred_mean,
               Y_fit_reg_unnormalized, Y_compare_unnormalized, title=None, ylabel=None, savepath=None, units_multiplier=1.0):
    # plot final 3D GP results using stacked bars
    X_fit_datetime_reg_str = np.datetime_as_string(X_fit_datetime_reg, unit='M')
    X_compare_datetime_str = np.datetime_as_string(X_compare[:, 0], unit='M')
    plt.figure(figsize=(8, 4))
    bar_width = 0.75
    plt.xticks(rotation=90)
    plt.bar(X_fit_datetime_reg_str, 1.96 * plot_std[:, 0] * units_multiplier, bar_width,
            bottom=(plot_mean[:, 0]) * units_multiplier, color="lightcyan", edgecolor='k', zorder=1,
            label='GP 95% credible interval, historical')
    plt.bar(X_fit_datetime_reg_str, 1.96 * plot_std[:, 0] * units_multiplier, bar_width,
            bottom=(plot_mean[:, 0] - 1.96 * plot_std[:, 0]) * units_multiplier, color="lightcyan", edgecolor='k', zorder=1)
    plt.bar(X_compare_datetime_str, (1.96 * plot_pred_std[:, 0]) * units_multiplier, bar_width,
            bottom=plot_pred_mean[:, 0] * units_multiplier, color="lavenderblush", edgecolor='k', zorder=1,
            label='GP 95% credible interval, forecast')
    plt.bar(X_compare_datetime_str, (1.96 * plot_pred_std[:, 0]) * units_multiplier, bar_width,
            bottom=(plot_pred_mean[:, 0] - 1.96 * plot_pred_std[:, 0]) * units_multiplier, color="lavenderblush", edgecolor='k', zorder=1)
    ax = plt.gca()
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 3 != 0:
            label.set_visible(False)
    plt.scatter(X_fit_datetime_reg_str, Y_fit_reg_unnormalized[:, 0] * units_multiplier, c="black", marker="x", zorder=2,
                label='EIA data, used to fit GP')
    plt.scatter(X_compare_datetime_str, Y_compare_unnormalized[:, 0] * units_multiplier, c="red", marker="x", zorder=2,
                label='EIA data, not used to fit GP')
    plt.title(title)
    plt.xlabel('Time (year)')
    plt.ylabel(ylabel)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def get_logging_output(m, X_float_compare, Y_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, Y_compare_unnormalized, feature=None):

    X_compare_3d_float = X_float_compare
    X_compare_3d_float_reg, Y_compare_3d_reg = clean_gp_input(X_compare_3d_float, Y_compare, X_optional_0=X_cdd_compare,
                                                              X_optional_1=X_hdd_compare)

    # calculate percent error
    Y_compare_3d_preds_normalized = m.predict(X_compare_3d_float_reg)[0]
    Y_compare_3d_preds_normalized_std = np.sqrt(m.predict(X_compare_3d_float_reg)[1])
    Y_compare_3d_preds = Y_compare_3d_preds_normalized * (Y_max - Y_min) + Y_min
    Y_compare_3d_preds_std = Y_compare_3d_preds_normalized_std * (Y_max - Y_min)
    percent_error_3d = (Y_compare_unnormalized - Y_compare_3d_preds) / Y_compare_3d_preds
    percent_error_95_CI_range = (1.96 * Y_compare_3d_preds_std) / Y_compare_3d_preds

    # save output
    summary_3d_dict = {'feature': feature,
                      'march 2020 value': Y_compare_unnormalized[0][0],
                      'march 2020 counterfactual': Y_compare_3d_preds[0][0],
                      'march 2020 fraction deviation': percent_error_3d[0][0],
                      'march 2020 95% CI (+/- %)': percent_error_95_CI_range[0][0],
                      'march 2020 95% CI low': Y_compare_3d_preds[0][0] - 1.96 * Y_compare_3d_preds_std[0][0],
                      'march 2020 95% CI high': Y_compare_3d_preds[0][0] + 1.96 * Y_compare_3d_preds_std[0][0],

                      'april 2020 value': Y_compare_unnormalized[1][0],
                      'april 2020 counterfactual': Y_compare_3d_preds[1][0],
                      'april 2020 fraction deviation': percent_error_3d[1][0],
                      'april 2020 95% CI (+/- %)': percent_error_95_CI_range[1][0],
                      'april 2020 95% CI low': Y_compare_3d_preds[1][0] - 1.96 * Y_compare_3d_preds_std[1][0],
                      'april 2020 95% CI high': Y_compare_3d_preds[1][0] + 1.96 * Y_compare_3d_preds_std[1][0],

                      'may 2020 value': Y_compare_unnormalized[2][0],
                      'may 2020 counterfactual': Y_compare_3d_preds[2][0],
                      'may 2020 fraction deviation': percent_error_3d[2][0],
                      'may 2020 95% CI (+/- %)': percent_error_95_CI_range[2][0],
                      'may 2020 95% CI low': Y_compare_3d_preds[2][0] - 1.96 * Y_compare_3d_preds_std[2][0],
                      'may 2020 95% CI high': Y_compare_3d_preds[2][0] + 1.96 * Y_compare_3d_preds_std[2][0],

                      'june 2020 value': Y_compare_unnormalized[3][0],
                      'june 2020 counterfactual': Y_compare_3d_preds[3][0],
                      'june 2020 fraction deviation': percent_error_3d[3][0],
                      'june 2020 95% CI (+/- %)': percent_error_95_CI_range[3][0],
                      'june 2020 95% CI low': Y_compare_3d_preds[3][0] - 1.96 * Y_compare_3d_preds_std[3][0],
                      'june 2020 95% CI high': Y_compare_3d_preds[3][0] + 1.96 * Y_compare_3d_preds_std[3][0],

                      'july 2020 value': Y_compare_unnormalized[4][0],
                      'july 2020 counterfactual': Y_compare_3d_preds[4][0],
                      'july 2020 fraction deviation': percent_error_3d[4][0],
                      'july 2020 95% CI (+/- %)': percent_error_95_CI_range[4][0],
                      'july 2020 95% CI low': Y_compare_3d_preds[4][0] - 1.96 * Y_compare_3d_preds_std[4][0],
                      'july 2020 95% CI high': Y_compare_3d_preds[4][0] + 1.96 * Y_compare_3d_preds_std[4][0],

                      'august 2020 value': Y_compare_unnormalized[5][0],
                      'august 2020 counterfactual': Y_compare_3d_preds[5][0],
                      'august 2020 fraction deviation': percent_error_3d[5][0],
                      'august 2020 95% CI (+/- %)': percent_error_95_CI_range[5][0],
                      'august 2020 95% CI low': Y_compare_3d_preds[5][0] - 1.96 * Y_compare_3d_preds_std[5][0],
                      'august 2020 95% CI high': Y_compare_3d_preds[5][0] + 1.96 * Y_compare_3d_preds_std[5][0],

                      'september 2020 value': Y_compare_unnormalized[6][0],
                      'september 2020 counterfactual': Y_compare_3d_preds[6][0],
                      'september 2020 fraction deviation': percent_error_3d[6][0],
                      'september 2020 95% CI (+/- %)': percent_error_95_CI_range[6][0],
                      'september 2020 95% CI low': Y_compare_3d_preds[6][0] - 1.96 * Y_compare_3d_preds_std[6][0],
                      'september 2020 95% CI high': Y_compare_3d_preds[6][0] + 1.96 * Y_compare_3d_preds_std[6][0],

                      'october 2020 value': Y_compare_unnormalized[7][0],
                      'october 2020 counterfactual': Y_compare_3d_preds[7][0],
                      'october 2020 fraction deviation': percent_error_3d[7][0],
                      'october 2020 95% CI (+/- %)': percent_error_95_CI_range[7][0],
                      'october 2020 95% CI low': Y_compare_3d_preds[7][0] - 1.96 * Y_compare_3d_preds_std[7][0],
                      'october 2020 95% CI high': Y_compare_3d_preds[7][0] + 1.96 * Y_compare_3d_preds_std[7][0],

                      'november 2020 value': Y_compare_unnormalized[8][0],
                      'november 2020 counterfactual': Y_compare_3d_preds[8][0],
                      'november 2020 fraction deviation': percent_error_3d[8][0],
                      'november 2020 95% CI (+/- %)': percent_error_95_CI_range[8][0],
                      'november 2020 95% CI low': Y_compare_3d_preds[8][0] - 1.96 * Y_compare_3d_preds_std[8][0],
                      'november 2020 95% CI high': Y_compare_3d_preds[8][0] + 1.96 * Y_compare_3d_preds_std[8][0],

                      'december 2020 value': Y_compare_unnormalized[9][0],
                      'december 2020 counterfactual': Y_compare_3d_preds[9][0],
                      'december 2020 fraction deviation': percent_error_3d[9][0],
                      'december 2020 95% CI (+/- %)': percent_error_95_CI_range[9][0],
                      'december 2020 95% CI low': Y_compare_3d_preds[9][0] - 1.96 * Y_compare_3d_preds_std[9][0],
                      'december 2020 95% CI high': Y_compare_3d_preds[9][0] + 1.96 * Y_compare_3d_preds_std[9][0]
                      }

    return pd.DataFrame(data=summary_3d_dict, index=[0])


def preprocess_data(X, Y, units_multiplier=1.0):
    # make new df so we can merge cooling and heating degree days into dataset
    dd_df = load_cooling_heating_degree_days()
    df_all = pd.DataFrame(data=X, columns=['X'])
    df_all['Y'] = Y
    df_all = df_all.merge(dd_df, left_on='X', right_on='Month')
    X_cdd = df_all['Cooling Degree days: U.S. (cooling degree days) cooling degree days']
    X_hdd = df_all['Heating Degree days: U.S. (heating degree days) heating degree days']
    X_cdd = X_cdd.values.reshape(X_cdd.values.size, 1)
    X_hdd = X_hdd.values.reshape(X_hdd.values.size, 1)

    X_float = np.array(X, dtype=float)
    X_min = np.min(X_float)
    X_max = np.max(X_float)
    Y = np.array(Y, dtype=float) * units_multiplier
    Y_min = np.min(Y)
    Y_max = np.max(Y)
    X_cdd_min = np.min(X_cdd)
    X_cdd_max = np.max(X_cdd)
    X_hdd_min = np.min(X_hdd)
    X_hdd_max = np.max(X_hdd)

    # normalize
    X_float = (X_float - X_min) / (X_max - X_min)
    Y = (Y - Y_min) / (Y_max - Y_min)
    X_cdd = (X_cdd - X_cdd_min) / (X_cdd_max - X_cdd_min)
    X_hdd = (X_hdd - X_hdd_min) / (X_hdd_max - X_hdd_min)

    # remove test months from the model!
    X_fit = X_float[:-num_months, :]
    X_fit_datetime = X[:-num_months, :]
    Y_fit = Y[:-num_months, :]
    X_cdd_fit = X_cdd[:-num_months, :]
    X_hdd_fit = X_hdd[:-num_months, :]

    X_float_compare = X_float[-num_months:, :]
    X_compare = X[-num_months:, :]
    Y_compare = Y[-num_months:, :]
    X_cdd_compare = X_cdd[-num_months:, :]
    X_hdd_compare = X_hdd[-num_months:, :]

    X_fit_datetime_reg = np.unique(X_fit_datetime)

    # undo normalization
    Y_compare_unnormalized = Y_compare * (Y_max - Y_min) + Y_min
    X_fit_reg, Y_fit_reg = clean_gp_input(X_fit, Y_fit, X_optional_0=X_cdd_fit, X_optional_1=X_hdd_fit)

    return X_fit_reg, Y_fit_reg, X_float_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, X_fit_datetime_reg, X_compare, Y_compare_unnormalized, Y_compare


def plot_c():
    eia_p = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'processed', 'eia923_monthly_c_and_c_over_e_and_e.p')
    eia923_df = pd.read_pickle(eia_p)
    final_em = get_eia923_c(eia923_df)

    X_923 = final_em['Date'].values
    Y_923 = final_em['Co2 Emissions (kg)'].values

    X_923 = X_923[buffer:]
    Y_923 = Y_923[buffer:]

    # process annualized data
    Y_923 = annualize_monthly_x_y(X_923, Y_923)

    # restructure
    X = X_923[np.newaxis].T
    Y = Y_923[np.newaxis].T

    X_fit_reg, Y_fit_reg, X_float_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, X_fit_datetime_reg, X_compare, Y_compare_unnormalized, Y_compare = preprocess_data(
        X, Y, units_multiplier=(1.0 / 1e3 / 1e6))

    # fit gp
    m = fit_3d_gp(X_fit_reg, Y_fit_reg)

    # run gp
    plot_std, plot_mean, plot_pred_std, plot_pred_mean, Y_fit_reg_unnormalized = run_3d_gp(m, X_fit_reg, X_float_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, Y_fit_reg)

    # plot final 3D GP results using stacked bars
    plot_3d_gp(X_fit_datetime_reg, X_compare, plot_std, plot_mean, plot_pred_std, plot_pred_mean,
               Y_fit_reg_unnormalized, Y_compare_unnormalized, title='Carbon Dioxide Emissions, C',
               ylabel='Carbon Dioxide Emissions (MMT, annualized)',
               savepath=os.path.join(output_path, '001_3D_c-gp-bars.pdf'))



    summary_3d_df = get_logging_output(m, X_float_compare, Y_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, Y_compare_unnormalized, feature='c')

    return summary_3d_df


def plot_c_over_e():
    eia_p = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'processed', 'eia923_monthly_c_and_c_over_e_and_e.p')
    eia923_df = pd.read_pickle(eia_p)
    final_fe = get_eia923_c_over_e(eia923_df)

    X_923 = final_fe['Date'].values
    Y_923 = final_fe['C/E (kg/MWh)'].values

    X_923 = X_923[buffer:]
    Y_923 = Y_923[buffer:]

    # restructure
    X = X_923[np.newaxis].T
    Y = Y_923[np.newaxis].T

    X_fit_reg, Y_fit_reg, X_float_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, X_fit_datetime_reg, X_compare, Y_compare_unnormalized, Y_compare = preprocess_data(
        X, Y, units_multiplier=(1.0))

    # fit gp
    m = fit_3d_gp(X_fit_reg, Y_fit_reg)

    # run gp
    plot_std, plot_mean, plot_pred_std, plot_pred_mean, Y_fit_reg_unnormalized = run_3d_gp(m, X_fit_reg, X_float_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, Y_fit_reg)

    # plot final 3D GP results using stacked bars
    plot_3d_gp(X_fit_datetime_reg, X_compare, plot_std, plot_mean, plot_pred_std, plot_pred_mean,
               Y_fit_reg_unnormalized, Y_compare_unnormalized, title='Carbon Intensity of Electricity Supply, C/E',
               ylabel='Carbon Intensity of Electricity Supply (kg/MWh)',
               savepath=os.path.join(output_path, '002_3D_c_over_e-gp-bars.pdf'))

    summary_3d_df = get_logging_output(m, X_float_compare, Y_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, Y_compare_unnormalized, feature='c_over_e')

    return summary_3d_df


def plot_e():
    eia_p = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'processed', 'eia923_monthly_c_and_c_over_e_and_e.p')
    eia923_df = pd.read_pickle(eia_p)
    final_gen = get_eia923_e(eia923_df)

    X_923 = final_gen['Date'].values
    Y_923 = final_gen['Net Generation (MWh)'].values

    X_923 = X_923[buffer:]
    Y_923 = Y_923[buffer:]

    # process annualized data
    Y_923 = annualize_monthly_x_y(X_923, Y_923)

    # restructure
    X = X_923[np.newaxis].T
    Y = Y_923[np.newaxis].T

    X_fit_reg, Y_fit_reg, X_float_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, X_fit_datetime_reg, X_compare, Y_compare_unnormalized, Y_compare = preprocess_data(
        X, Y, units_multiplier=(1.0 / 1e6))

    # fit gp
    m = fit_3d_gp(X_fit_reg, Y_fit_reg)

    # run gp
    plot_std, plot_mean, plot_pred_std, plot_pred_mean, Y_fit_reg_unnormalized = run_3d_gp(m, X_fit_reg, X_float_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, Y_fit_reg)

    # plot final 3D GP results using stacked bars
    plot_3d_gp(X_fit_datetime_reg, X_compare, plot_std, plot_mean, plot_pred_std, plot_pred_mean,
               Y_fit_reg_unnormalized, Y_compare_unnormalized, title='Electricity Generation, E',
               ylabel='Electricity Generation (TWh, annualized)',
               savepath=os.path.join(output_path, '0035_3D_e-gp-bars.pdf'),
               units_multiplier=1.0
               )

    summary_3d_df = get_logging_output(m, X_float_compare, Y_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, Y_compare_unnormalized, feature='e')

    return summary_3d_df


def plot_c_by_fuel(fuel_source):
    eia_p = os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'processed', 'eia923_monthly_c_by_fuel.p')
    eia923_df = pd.read_pickle(eia_p)
    dd_df = load_cooling_heating_degree_days()

    X_923 = eia923_df.index.values
    Y_923 = eia923_df[fuel_source].values

    X_923 = X_923[buffer:]
    Y_923 = Y_923[buffer:]

    # process annualized data
    Y_923 = annualize_monthly_x_y(X_923, Y_923)

    # restructure
    X = X_923[np.newaxis].T
    Y = Y_923[np.newaxis].T

    X_fit_reg, Y_fit_reg, X_float_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, X_fit_datetime_reg, X_compare, Y_compare_unnormalized, Y_compare = preprocess_data(
        X, Y)

    # fit gp
    m = fit_3d_gp(X_fit_reg, Y_fit_reg)

    # run gp
    plot_std, plot_mean, plot_pred_std, plot_pred_mean, Y_fit_reg_unnormalized = run_3d_gp(m, X_fit_reg, X_float_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, Y_fit_reg)

    # plot final 3D GP results using stacked bars
    plot_3d_gp(X_fit_datetime_reg, X_compare, plot_std, plot_mean, plot_pred_std, plot_pred_mean,
               Y_fit_reg_unnormalized, Y_compare_unnormalized,
               title='Carbon Dioxide Emissions from ' + fuel_source + ', C',
               ylabel='Carbon Dioxide Emissions (MMT, annualized)',
               savepath=os.path.join(output_path, 'c-gp-3d-gp-bars-' + fuel_source + '.pdf'))

    summary_3d_df = get_logging_output(m, X_float_compare, Y_compare, X_cdd_compare, X_hdd_compare, Y_max, Y_min, Y_compare_unnormalized, feature='c(' + fuel_source + ')')

    return summary_3d_df


# plot everything
summary_c_df = plot_c()
summary_e_df = plot_e()
summary_c_over_e_df = plot_c_over_e()
summary_c_coal_df = plot_c_by_fuel('Coal')
summary_c_oil_df = plot_c_by_fuel('Oil')
summary_c_gas_df = plot_c_by_fuel('Gas')

summary_all_df = pd.concat([summary_c_df, summary_e_df, summary_c_over_e_df, summary_c_coal_df, summary_c_oil_df, summary_c_gas_df])

# save results to CSV
summary_all_df.to_csv(os.path.join(output_path, 'zzz_summary_3d.csv'), index=False, header=True)
