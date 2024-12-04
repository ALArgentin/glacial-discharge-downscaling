import numpy as np
import pandas as pd
from extract_hydrological_variables import select_months


def convert_to_hydrobricks_units(daily_mean_discharges, watershed_path,
                                 hydrobricks_files, catchment, 
                                 start_date='2010-01-01', end_date='2014-12-31'):
    """
    Converts discharge units between hydrological and Hydrobricks formats.

    This function handles the conversion of daily mean discharge data from
    one format to another, suitable for Hydrobricks usage. It converts
    discharges from depth-based units (mm/d) to volumetric flow rates (m³/s).

    @param daily_mean_discharges (str)
        Path to the CSV file containing daily mean discharges. Must include
        a 'Date' column and discharge data columns.
    @param watershed_path (str)
        Directory path containing watershed area data files.
    @param hydrobricks_files (str)
        Path or prefix for Hydrobricks output files.
    @param catchment (str)
        The specific catchment key for processing, used in file naming.
    @param start_date (str, optional)
        The start of the discharge date range.
    @param end_date (str, optional)
        The end of the discharge date range.

    @return (None)
        Writes the converted discharge data to the specified Hydrobricks
        output files.

    Notes:
    -----
    - Watershed area values are read from text files corresponding to
      each catchment or discharge key.
    - The conversion between units considers the following factors:
        - 1 m³/s = 86400 mm/d over the watershed area.
        - 1 mm/d = 1000 m³/s per watershed area.
    """

    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)
    times = pd.date_range(start_datetime, end_datetime, freq='D')
    means = pd.read_csv(daily_mean_discharges, header=0, na_values='', usecols=[0,1])
    means.set_index('Date', inplace=True)

    watershed_area = np.loadtxt(watershed_path, skiprows=1)
    m_to_mm = 1000
    persec_to_perday = 86400
    means = means * watershed_area / m_to_mm / persec_to_perday
    filename = hydrobricks_files
    means.columns = ['Discharge (m3/s)']
    means.to_csv(filename, date_format='%d/%m/%Y')

def retrieve_subdaily_discharge(df, day):
    """
    Retrieves sub-daily discharge data for a specific day from a dataframe.

    This function filters the input dataframe to include only rows corresponding
    to the specified day, based on the `df_str_index`. It also removes the first
    column of the filtered data.

    @param df (dataframe)
        The input dataframe containing discharge data. It is assumed to have
        a column structure where the first column will be dropped after filtering.
    @param day (str)
        A string representing the day for which to retrieve sub-daily discharge data.
        The format of the string should match the format of `df_str_index`.

    @return (dataframe)
        A dataframe containing the sub-daily discharge data for the specified day,
        with the first column removed.
    """

    # Select the day's data
    df_str_index = df.index.strftime('%Y-%m-%d')
    observed_subdaily_discharge = df[df_str_index == day]

    observed_subdaily_discharge = observed_subdaily_discharge.drop(observed_subdaily_discharge.columns[0], axis=1)

    return observed_subdaily_discharge


def get_statistics(observed_subdaily_discharge):
    """
    Computes the minimum, mean and maximum discharge out of the given discharge
    timeseries.

    @param observed_subdaily_discharge (dataframe)
        The observed discharge for a day.

    @return (float, float, float)
        The minimum, mean and maximum discharges of the day.
    """

    q_min = np.nanmin(observed_subdaily_discharge)
    q_mean = np.nanmean(observed_subdaily_discharge)
    q_max = np.nanmax(observed_subdaily_discharge)

    return q_min, q_mean, q_max

def block_boostrapping(observed_FDCs_df, months, n_evals=100):
    """
    Performs block bootstrapping of the flow duration curve (FDC) time series.

    This function implements a block bootstrapping technique on the observed
    flow duration curves (FDCs) for specified months, ensuring that each year's
    data is treated as a block. The function removes leap-year extra days to maintain
    consistency in yearly data (365 days per year). It returns both the original
    formatted dataset and a DataFrame containing bootstrapped FDC samples.

    @param observed_FDCs_df (pandas.DataFrame)
        A dataframe containing the observed flow duration curves (FDCs). It must
        have a datetime index, and the `Discharge` column should represent the flow data.
    @param months (list of int)
        A list of integers representing the months (1-12) to select for the analysis.
    @param n_evals (int, optional)
        The number of bootstrapped FDC samples to generate (default is 100).

    @return (pandas.DataFrame, pandas.DataFrame)
        A tuple containing:
        1. The original FDC dataset after processing (with leap days removed and
            formatted index).
        2. A dataframe containing bootstrapped FDC samples as additional columns
            (one column per sample).

    Notes:
    -----
    - The function drops the last day of leap years (February 29 and the associated
      December 31) to ensure consistency in annual data length (365 days).
    - If only one year is available in the dataset, the function exits with a
      warning message, as bootstrapping requires multiple years.
    """

    print(f"Block boostrapping...")
    # Formatting
    observed_FDCs_df.index.name = "date"
    observed_FDCs_df = select_months(observed_FDCs_df, months)
    observed_FDCs_df.reset_index(inplace=True)

    observed_FDCs_df['year'] = pd.DatetimeIndex(observed_FDCs_df['date']).year
    observed_FDCs_df = observed_FDCs_df.set_index('date')

    # Get rid of the last day of bissextile years (leap years) to always have years of 365 days.
    # We choose this day as it is already done in this way in the data gathered for Arolla.
    leap_days = observed_FDCs_df[observed_FDCs_df.index.strftime('%m-%d') == '02-29']
    leap_years = leap_days.year.unique()
    for year in leap_years:
        observed_FDCs_df.drop(observed_FDCs_df.loc[observed_FDCs_df.index.strftime('%Y-%m-%d') == str(year) + '-12-31'].index, inplace=True)

    index = observed_FDCs_df.index

    # Years available for bootstrapping
    years = observed_FDCs_df.year.unique()
    if len(years) == 1:
        print("Not possible computing reference metric on one year only.")
        return -1

    observed_FDCs_df = observed_FDCs_df.set_index('year')

    # Create the dataframe to store all bootstrapped discharges
    sampled_years = np.random.choice(years, size=years.size, replace=True)
    all_bootstrapped_FDCs_dfs = observed_FDCs_df.loc[sampled_years].copy()
    all_bootstrapped_FDCs_dfs.drop("Discharge", axis=1, inplace=True)
    all_bootstrapped_FDCs_dfs.index = index
    all_bootstrapped_FDCs_dfs.index.name = "Date"

    # Create all the bootstrapped discharges and evaluate them
    i = 0
    while i < n_evals:
        sampled_years = np.random.choice(years, size=years.size, replace=True)
        diff = sampled_years - years
        if not np.all(diff):
            continue
        bootstrapped_FDCs_df = observed_FDCs_df.loc[sampled_years].copy()
        all_bootstrapped_FDCs_dfs[str(i)] = bootstrapped_FDCs_df.values
        i += 1

    return observed_FDCs_df, all_bootstrapped_FDCs_dfs


def recreate_stacked_FDCs_from_observed_subdaily_discharge(
        subdaily_discharge, observed_FDCs_output_file):
    """
    Recreates a stacked flow duration curve (FDC) dataset from sub-daily discharge data.

    Processes sub-daily discharge data to create a time series of flow duration curves
    (FDCs). The input data is assumed to have 15-minute discharge intervals. The function
    stacks these intervals into a dataset and saves the processed FDC data to an output file.

    @param subdaily_discharge (str)
        Path to the CSV file containing the sub-daily discharge data. The file must have
        at least two columns: 'Date' (format: '%Y-%m-%d %H:%M:%S') and 'Discharge'.
    @param observed_FDCs_output_file (str)
        Path to the output CSV file where the processed stacked FDC dataset will be saved.

    @return (pandas.DataFrame)
        A DataFrame containing the stacked FDC dataset indexed by 15-minute intervals.

    Notes:
    -----
    - Assumes the input discharge data is recorded at 15-minute intervals (96 points).
    """

    # Observed discharge
    df = pd.read_csv(subdaily_discharge, header=0, na_values='', usecols=[0, 1],
                     parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    df.rename(columns={df.columns[1]: 'Discharge'}, inplace=True)
    index = df.index
    df['Date'] = df['Date'].dt.normalize()

    # Select the dates
    date1 = df['Date'].values[0]
    date2 = df['Date'].values[-1]

    # Create the observed FDCs
    FDCs_df = pd.DataFrame()
    for day in pd.date_range(start=date1, end=date2):
        observed_subdaily_discharge = df[df['Date'] == day]
        dfs = observed_subdaily_discharge.sort_values(by='Discharge', ascending=False)
        assert len(dfs) == 96 or len(dfs) == 0
        FDCs_df = pd.concat([FDCs_df, dfs])

    # Recreate the 15-min simulation intervals for the FDCs
    number_of_measures = len(FDCs_df["Date"]) # Without bisextile days
    number_of_days = len(FDCs_df["Date"]) / 96
    in_day_increment = list(range(96)) * int(number_of_days)

    # Add the 15-min intervals to the datetimes
    stacked_FDCs_time = [FDCs_df["Date"].iloc[i] + np.timedelta64(in_day_increment[i] * 15, 'm') for i in range(number_of_measures)]

    # Recreate the 15-min simulation intervals for the FDCs
    FDCs_df.index = stacked_FDCs_time
    FDCs_df.drop("Date", axis=1, inplace=True)
    FDCs_df.index.name = "Date"

    FDCs_df.to_csv(observed_FDCs_output_file)

    return FDCs_df

def bootstrapping_observed_FDCs(subdaily_discharge, observed_FDCs_output_file, months):
    """
    Recreates stacked FDCs from sub-daily discharge data, then applies block
    bootstrapping to generate multiple bootstrapped FDC datasets for analysis.

    @param subdaily_discharge (str)
        Path to the CSV file containing the sub-daily discharge data. The file must have
        at least two columns: 'Date' (format: '%Y-%m-%d %H:%M:%S') and 'Discharge'.
    @param observed_FDCs_output_file (str)
        Path to the output CSV file where the stacked observed FDC dataset will be saved.
    @param months (list of int)
        List of months to include in the analysis (e.g., [6, 7, 8, 9] for summer months).

    @return (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        A tuple containing:
        1. DataFrame containing the stacked observed FDC dataset.
        2. DataFrame of the cleaned observed FDC dataset after removing leap year
           inconsistencies and other formatting steps.
        3. DataFrame containing all bootstrapped FDC datasets.
    """
    print("Bootstrapping of observed FDCs...")

    observed_FDCs_df = recreate_stacked_FDCs_from_observed_subdaily_discharge(subdaily_discharge, observed_FDCs_output_file)
    cleaned_observed_FDCs_df, all_bootstrapped_FDCs_dfs = block_boostrapping(observed_FDCs_df, months)

    return observed_FDCs_df, cleaned_observed_FDCs_df, all_bootstrapped_FDCs_dfs
