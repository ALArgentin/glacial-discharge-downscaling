import hydrobricks as hb
import numpy as np
import pandas as pd
import pymannkendall as mk


def mann_kendall_tests(meteo_df, results):

    meteo_df.index = pd.to_datetime(meteo_df.index)
    # Select the ones you want
    meteo_df = meteo_df[['$a$', '$b$', '$c$', '$M$']]
    yearly_meteo_df = meteo_df.groupby(pd.Grouper(freq='Y')).mean()

    # perform Mann-Kendall Trend Test
    with open(f'{results}Mann_Kendall_test_results.txt', 'w') as file:
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(yearly_meteo_df['$a$'].values)
        file.write(f"a, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(yearly_meteo_df['$b$'].values)
        file.write(f"b, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(yearly_meteo_df['$c$'].values)
        file.write(f"c, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(yearly_meteo_df['$M$'].values)
        file.write(f"M, {trend}, {h}, {p}, {z}, {Tau}, {s}, {var_s}, {slope}, {intercept}\n")

    # perform seasonal Mann-Kendall Trend Test
    with open(f'{results}Seasonal_Mann_Kendall_test_results.txt', 'w') as file:
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

    Parameters
    ----------
    metric : str
        The abbreviation of the function as defined in HydroErr
        (https://hydroerr.readthedocs.io/en/stable/list_of_metrics.html)
        Examples: nse, kge_2012, ...
    n_evals : int
        Number of evaluations to perform (default: 100).

    Returns
    -------
    The mean value of n_evals realization of the selected metric.
    """
    print(f"Compute reference {metric}...")
    n_evals = len(all_bootstrapped_FDCs_dfs.columns)

    # Create all the bootstrapped discharges and evaluate them
    metrics = np.empty(n_evals)
    i = 0
    while i < n_evals:
        value = hb.evaluate(all_bootstrapped_FDCs_dfs[str(i)].values, observed_FDCs_df['Discharge'].values, metric)
        metrics[i] = value
        i += 1

    ref_metric = np.mean(metrics)

    return ref_metric

def compute_metric(simulated_FDCs_df, cleaned_observed_FDCs_df, months, metric):
    print(f"Compute {metric}...")

    # Formatting
    simulated_FDCs_df.reset_index(inplace=True)
    simulated_FDCs_df['year'] = pd.DatetimeIndex(simulated_FDCs_df['Date']).year
    simulated_FDCs_df = simulated_FDCs_df.set_index('Date')
    simulated_FDCs_df = fc.select_months(simulated_FDCs_df, months)

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
    # Computation of the coefficient of determination
    # residual sum of squares
    ss_res = np.sum((y_data - y_fit) ** 2)
    # total sum of squares
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    # r-squared
    r2 = 1 - (ss_res / ss_tot)

    return r2

