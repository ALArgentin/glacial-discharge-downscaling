
import numpy as np
import pandas as pd

def convert_to_hydrobricks_units(daily_mean_discharges, watershed_path, area_txt, 
                                 hydrobricks_files, opposite_conversion=False, catchment=None):
    if opposite_conversion:
        start_date = '2010-01-01'
        end_date = '2014-12-31'
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        times = pd.date_range(start_datetime, end_datetime, freq='D')
        means = pd.read_csv(daily_mean_discharges, header=0, na_values='', usecols=[0,1])
        #means.insert(0, 'Date', times)
        means.set_index('Date', inplace=True)
    else:
        timestamp_format = '%Y-%m-%d'
        means = pd.read_csv(daily_mean_discharges + '.csv', header=0, na_values=np.nan, 
                            parse_dates=['Date'], date_format=timestamp_format, index_col=0)

    for key in means:
        
        if key != "BIrest":
            if opposite_conversion:
                filename = watershed_path + catchment + area_txt
                suffix = '_' + catchment + '.csv'
            else:
                filename = watershed_path + key + area_txt
                suffix = '_' + key + '.csv'
            watershed_area = np.loadtxt(filename, skiprows=1)
            # Create new pandas DataFrame.
            subdataframe = means[[key]]
            m_to_mm = 1000
            persec_to_perday = 86400
            if opposite_conversion:
                subdataframe = subdataframe * watershed_area / m_to_mm / persec_to_perday
                filename = hydrobricks_files
                subdataframe.columns = ['Discharge (m3/s)']
            else:
                subdataframe = subdataframe / watershed_area * m_to_mm * persec_to_perday
                filename = hydrobricks_files + suffix
                subdataframe.columns = ['Discharge (mm/d)']
            subdataframe.to_csv(filename, date_format='%d/%m/%Y')

def retrieve_subdaily_discharge(df, df_str_index, day):
    
    # Select the day's data
    observed_subdaily_discharge = df[df_str_index == day]
    
    observed_subdaily_discharge = observed_subdaily_discharge.drop(observed_subdaily_discharge.columns[0], axis=1)
    
    return observed_subdaily_discharge


def get_statistics(observed_subdaily_discharge):
    
    q_min = np.nanmin(observed_subdaily_discharge)
    q_mean = np.nanmean(observed_subdaily_discharge)
    q_max = np.nanmax(observed_subdaily_discharge)
    
    return q_min, q_mean, q_max

def block_boostrapping(observed_FDCs_df, months, n_evals=100):
    print(f"Block boostrapping...")
    # Formatting
    observed_FDCs_df.index.name = "date"
    observed_FDCs_df = fc.select_months(observed_FDCs_df, months)
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

def bootstrapping_observed_FDCs(subdaily_discharge, observed_FDCs_output_file, months):
    print("Bootstrapping of observed FDCs...")
    
    observed_FDCs_df = recreate_stacked_FDCs_from_observed_subdaily_discharge(subdaily_discharge, observed_FDCs_output_file)
    cleaned_observed_FDCs_df, all_bootstrapped_FDCs_dfs = block_boostrapping(observed_FDCs_df, months)
    
    return observed_FDCs_df, cleaned_observed_FDCs_df, all_bootstrapped_FDCs_dfs
