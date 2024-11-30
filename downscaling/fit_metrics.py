import hydrobricks as hb
import numpy as np
import pandas as pd
import pymannkendall as mk
from extract_hydrological_variables import select_months


def mann_kendall_tests(meteo_df, file_paths):
    """
    Running a Mann-Kendall Trend test and a seasonal Mann-Kendall Trend on the
    calibrated parameters over several years.

    @param meteo_df (dataframe)
        The daily meteorological dataset, which also contains the
        calibrated downscaling parameters.
    @param file_paths (FilePaths)
        Object containing all file paths, used to give the file
        path and name of the results files.
    """

    meteo_df.index = pd.to_datetime(meteo_df.index)
    # Select the ones you want
    meteo_df = meteo_df[['$a$', '$b$', '$c$', '$M$']]
    yearly_meteo_df = meteo_df.groupby(pd.Grouper(freq='Y')).mean()

    # perform Mann-Kendall Trend Test
    with open(file_paths.mann_kendall_filename, 'w') as file:
        file.write(f"Parameter, Trend, h, p, z, Tau, s, var_s, slope, intercept\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(yearly_meteo_df['$a$'].values)
        file.write(f"a, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(yearly_meteo_df['$b$'].values)
        file.write(f"b, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(yearly_meteo_df['$c$'].values)
        file.write(f"c, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(yearly_meteo_df['$M$'].values)
        file.write(f"M, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")

    # perform seasonal Mann-Kendall Trend Test
    with open(file_paths.seasonal_mann_kendall_filename, 'w') as file:
        file.write(f"Parameter, Trend, h, p, z, Tau, s, var_s, slope, intercept\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.seasonal_test(meteo_df['$a$'].values, period=122)
        file.write(f"a, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.seasonal_test(meteo_df['$b$'].values, period=122)
        file.write(f"b, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.seasonal_test(meteo_df['$c$'].values, period=122)
        file.write(f"c, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.seasonal_test(meteo_df['$M$'].values, period=122)
        file.write(f"M, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")

def compute_reference_metric(all_bootstrapped_FDCs_dfs, observed_FDCs_df, metric):
    """
    Compute a reference for the provided metric (goodness of fit)
    by block bootstrapping the observed series n_evals times (100 times by default),
    evaluating the bootstrapped series using the provided metric and computing
    the mean of the results.

    @param all_bootstrapped_FDCs_dfs (dataframe)
        All bootstrapped flow duration curves as different columns.
    @param observed_FDCs_df (dataframe)
        The observed flow duration curves.
    @param metric (str)
        The abbreviation of the function as defined in HydroErr
        (https://hydroerr.readthedocs.io/en/stable/list_of_metrics.html)
        Examples: nse, kge_2012, ...

    @return The mean value of n_evals realization of the selected metric.
    """
    print(f"Compute reference {metric}...")
    # The length of the dataframe is the number of bootstrapped timeseries.
    n_evals = len(all_bootstrapped_FDCs_dfs.columns)

    # Evaluate all the bootstrapped discharge timeseries.
    metrics = np.empty(n_evals)
    i = 0
    while i < n_evals:
        value = hb.evaluate(all_bootstrapped_FDCs_dfs[str(i)].values, observed_FDCs_df['Discharge'].values, metric)
        metrics[i] = value
        i += 1

    ref_metric = np.mean(metrics)

    return ref_metric

def compute_metric(simulated_FDCs_df, cleaned_observed_FDCs_df, months, metric):
    """
    Compute the hydrological metric indicated as input on the two discharge datasets
    inputted.

    @param simulated_FDCs_df (dataframe)
        The simulated flow duration curves.
    @param cleaned_observed_FDCs_df (dataframe)
        The observed flow duration curves, already preprocessed.
    @param months (list)
        The months to which the analysis is restricted.
    @param metric (str)
        The abbreviation of the function as defined in HydroErr
        (https://hydroerr.readthedocs.io/en/stable/list_of_metrics.html)
        Examples: nse, kge_2012, ...

    @return (float) The value of the computed metric.
    """
    print(f"Compute {metric}...")

    # Formatting
    simulated_FDCs_df.reset_index(inplace=True)
    simulated_FDCs_df['year'] = pd.DatetimeIndex(simulated_FDCs_df['Date']).year
    simulated_FDCs_df = simulated_FDCs_df.set_index('Date')
    simulated_FDCs_df = select_months(simulated_FDCs_df, months)

    # Get rid of the last day of bissextile years (leap years) to always have years of 365 days.
    # We choose this day as it is already done in this way in the data gathered for Arolla.
    leap_days = simulated_FDCs_df[simulated_FDCs_df.index.strftime('%m-%d') == '02-29']
    leap_years = leap_days.year.unique()
    for year in leap_years:
        simulated_FDCs_df.drop(simulated_FDCs_df.loc[simulated_FDCs_df.index.strftime('%Y-%m-%d') == str(year) + '-12-31'].index, inplace=True)

    simulated_FDCs_df = simulated_FDCs_df.set_index('year')

    # Only select the intersection of the two datasets (mostly used when the simulated data comes from Hydrobricks)
    intersection_dates = simulated_FDCs_df.index.intersection(cleaned_observed_FDCs_df.index)
    simulated_FDCs_df = simulated_FDCs_df.loc[intersection_dates]
    cleaned_observed_FDCs_df = cleaned_observed_FDCs_df.loc[intersection_dates]

    metric = hb.evaluate(simulated_FDCs_df['Discharge'].values, cleaned_observed_FDCs_df['Discharge'].values, metric)

    return metric


def compute_r2(y_data, y_fit):
    """
    Computation of the coefficient of determination.

    @param y_data (array)
        The first dataset.
    @param y_fit (array)
        The second dataset.
    @return (float) The coefficient of determination.
    """
    # residual sum of squares
    ss_res = np.sum((y_data - y_fit) ** 2)
    # total sum of squares
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    # r-squared
    r2 = 1 - (ss_res / ss_tot)

    return r2

