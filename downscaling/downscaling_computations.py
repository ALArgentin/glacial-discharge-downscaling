import csv
import sys
from datetime import datetime

import distribution_fitting as dsf
import extract_hydrological_variables as fc
import numpy as np
import pandas as pd
import pome_fitting as fit
import scipy.stats as stats
from preprocessing import get_statistics, retrieve_subdaily_discharge


def simulate_a_flow_duration_curve(params, q_min, q_max, M, day_meteo, function="Singh2014"):
    """
    Simulate a flow duration curve based on the given parameter sets
    and minimum and maximum discharges.
    
    @param params (list?)
        The set of parameters determined in the first calibration step.
    @param q_min (?)
        The minimum daily discharge.
    @param q_max (?)
        The maximum daily discharge.
    @param M (?)
        Parameter 'M' of the equation.
    @param day_meteo (dataframe)
        The daily meteorological dataset. Only used for the
        temporary new version. TO CHECK
    @param function (str)
        The function to use for the downscaling, between:
        - "Singh2014": the original function
        - "Sigmoid_d": the glacial function
        - "Sigmoid": the simplified glacial function
    
    @return y_fit (?) Return the simulated discharge.
    """
    # Extract the values of the params
    if np.isnan(params).any():
        a = b = c = d = np.nan
    else:
        if function == "Singh2014":
            a, b = params
        elif function == "Sigmoid_d":
            a, b, c, d = params
        elif function == "Sigmoid":
            a, b, c = params
        elif function == "Sigmoid_ext_var":
            var1 = day_meteo['Temperature'].values[0]
            var2 = day_meteo['Precipitation'].values[0]
            var3 = day_meteo['Radiation'].values[0]
            #a1, a2, a3, b1, b2, b3, c1, c2, c3 = params
            a1, b1, c1 = params
            a = a1 * var1 #+ a2 * var2 + a3 * var3
            b = b1 * var1 #+ b2 * var2 + b3 * var3
            c = c1 * var1 #+ c2 * var2 + c3 * var3

    # Generate fitted curve for plotting
    t_fit = np.linspace(0, 1, 96)
    if function == "Singh2014":
        y_fit = fit.discharge_time_equation_Singh2014(t_fit, a, b, q_min, q_max, M)
    elif function == "Sigmoid_d":
        y_fit = fit.discharge_time_equation_Sigmoid_d(t_fit, a, b, c, d, q_min, q_max, M)
    elif function == "Sigmoid" or function == "Sigmoid_ext_var":
        y_fit = fit.discharge_time_equation_Sigmoid(t_fit, a, b, c, q_min, q_max, M)

    # Return the simulated discharge
    return y_fit

def calibration_workflow(meteo_df, filename, function, dataframe_filename, 
                         file_paths, months):
    """
    Calibration workflow.
    
    @param meteo_df (dataframe)
        The daily meteorological dataset, which also contains the
        calibrated downscaling parameters.
    @param filename (?)
    
    @param function (str)
        The function to use for the downscaling, between:
        - "Singh2014": the original function
        - "Sigmoid_d": the glacial function
        - "Sigmoid": the simplified glacial function
    @param (?)
    @param file_paths (FilePaths)
        Object containing all file paths, used to give the file
        path and name of the results files.
    @param months (list)
        The months to which the analysis is restricted.
    """

    # Determined parameter lists
    a_array = []
    b_array = []
    c_array = []
    d_array = []
    M_array = []
    a1_array = []
    a2_array = []
    a3_array = []
    b1_array = []
    b2_array = []
    b3_array = []
    c1_array = []
    c2_array = []
    c3_array = []
    qmin_array = []
    qmax_array = []
    qmean_array = []
    h_array = []
    r2_array = []

    # In-between results needed for plotting
    x_data1_df = pd.DataFrame(columns=np.arange(96))
    y_data1_df = pd.DataFrame(columns=np.arange(96))
    x_fit1_df = pd.DataFrame(columns=np.arange(96))
    y_fit1_df = pd.DataFrame(columns=np.arange(96))
    r21_df = pd.DataFrame(columns=[0])
    x_data2_df = pd.DataFrame(columns=np.arange(96))
    y_data2_df = pd.DataFrame(columns=np.arange(96))
    t_fit2_df = pd.DataFrame(columns=np.arange(96))
    y_fit2_df = pd.DataFrame(columns=np.arange(96))
    r22_df = pd.DataFrame(columns=[0])

    # Open the discharge dataset and get all data points from the requested date
    df = pd.read_csv(filename, header=0, na_values='', usecols=[0, 1],
                     parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    df.rename(columns={df.columns[1]: 'Discharge'}, inplace=True)
    # Change the index format
    df.index = pd.to_datetime(df.iloc[:,0].values, format='%Y-%m-%d')
        
    meteo_df_str_index = meteo_df.index.strftime('%Y-%m-%d')
    
    daily_df = df.groupby(pd.Grouper(freq='D')).mean()
    daily_summer_df = fc.select_months(daily_df, months)

    start_t = datetime.now()
    
    time_df = df
    if function == "Sigmoid_ext_var":
        time_df = meteo_df

    date_range = np.unique(fc.select_months(time_df, months).index.strftime('%Y-%m-%d'))
    for i, day in enumerate(date_range):
        if day.endswith("-06-01"):
            print(f"Processing year {day}: {(datetime.now() - start_t).seconds} s spent")
        observed_subdaily_discharge = retrieve_subdaily_discharge(df, day=day)
        day_meteo = meteo_df[meteo_df_str_index == day]
        
        # CAREFUL, in this function I switched X_data and y_data in the sigmoid functions.... Consequently also switched for the rest.
        params, cov, x_data1, y_data1, x_fit1, y_fit1, r21 = fit.fit_a_and_b_to_discharge_probability_curve(observed_subdaily_discharge, day_meteo, function=function)
        x_data1_df.loc[day] = x_data1
        y_data1_df.loc[day] = y_data1
        if len(x_fit1) != 0: x_fit1_df.loc[day] = x_fit1
        if len(y_fit1) != 0: y_fit1_df.loc[day] = y_fit1
        if r21: r21_df.loc[day] = r21
        q_min, q_mean, q_max = get_statistics(observed_subdaily_discharge)
        M, var, x_data2, y_data2, t_fit2, y_fit2, r22 = fit.fit_m_to_flow_duration_curve(observed_subdaily_discharge, params, q_min, q_max, day_meteo, function=function)
        x_data2_df.loc[day] = x_data2
        y_data2_df.loc[day] = y_data2
        if len(t_fit2) != 0: t_fit2_df.loc[day] = t_fit2
        if len(y_fit2) != 0: y_fit2_df.loc[day] = y_fit2
        if r22: r22_df.loc[day] = r22
        simulated_subdaily_distribution = simulate_a_flow_duration_curve(params, q_min, q_max, M, day_meteo, function=function)

        # Extract the values of the params
        if np.isnan(params).any():
            a = b = c = d = np.nan
        else:
            if function == "Singh2014":
                a, b = params
            elif function == "Sigmoid_d":
                a, b, c, d = params
            elif function == "Sigmoid":
                a, b, c = params
            elif function == "Sigmoid_ext_var":
                var1 = day_meteo['Temperature'].values[0]
                var2 = day_meteo['Precipitation'].values[0]
                var3 = day_meteo['Radiation'].values[0]
                #a1, a2, a3, b1, b2, b3, c1, c2, c3 = params
                a1, b1, c1 = params
                a = a1 * var1 #+ a2 * var2 + a3 * var3
                b = b1 * var1 #+ b2 * var2 + b3 * var3
                c = c1 * var1 #+ c2 * var2 + c3 * var3

        if function == "Sigmoid_d":
            c_array.append(c)
            d_array.append(d)
        elif function == "Sigmoid":
            c_array.append(c)
        elif function == "Sigmoid_ext_var":
            c_array.append(c)
            a1_array.append(a1)
            #a2_array.append(a2)
            #a3_array.append(a3)
            b1_array.append(b1)
            #b2_array.append(b2)
            #b3_array.append(b3)
            c1_array.append(c1)
            #c2_array.append(c2)
            #c3_array.append(c3)
        h = stats.entropy(observed_subdaily_discharge)[0]

        a_array.append(a)
        b_array.append(b)
        M_array.append(M)
        qmin_array.append(q_min)
        qmax_array.append(q_max)
        qmean_array.append(q_mean)
        h_array.append(h)
    print("End of calibration.")

    meteo_df = meteo_df.reindex(daily_summer_df.index, fill_value=np.nan, method='nearest', tolerance='2min')
    # WORKS HERE
    
    data_to_assign = {
    "$a$": a_array,
    "$b$": b_array,
    "$M$": M_array,
    "$Q_{min}$": qmin_array,
    "$Q_{max}$": qmax_array,
    "$Q_{mean}$": qmean_array,
    "Entropy": h_array
    }
    if function == "Sigmoid_d":
        data_to_assign.update({"$c$": c_array, "$d$": d_array})
    elif function == "Sigmoid":
        data_to_assign.update({"$c$": c_array})
    elif function == "Sigmoid_ext_var":
        data_to_assign.update({
            "$c$": c_array,
            "$a1$": a1_array,
            "$b1$": b1_array,
            "$c1$": c1_array
        })
    
    #meteo_df = meteo_df.reindex(time_df.index)
    for key, value in data_to_assign.items():
        meteo_df.loc[:, key] = np.nan
        meteo_df.loc[:, key].loc[date_range] = value
        #print(bvuipö)
        #meteo_df.loc[time_df.index, key] = value
    #print(ioäbä)
    
    #meteo_df["Day of the year"] = daily_summer_df.index.dayofyear

 #   meteo_df.loc[time_df.index, "$a$"] = a_array
 #   meteo_df.loc[time_df.index, "$b$"] = b_array
 #   if function == "Sigmoid_d":
 #       meteo_df.loc[time_df.index, "$c$"] = c_array
 #       meteo_df.loc[time_df.index, "$d$"] = d_array
 #   elif function == "Sigmoid":
 #       meteo_df.loc[time_df.index, "$c$"] = c_array
 #   elif function == "Sigmoid_ext_var":
 #       meteo_df.loc[time_df.index, "$c$"] = c_array
 #       meteo_df.loc[time_df.index, "$a1$"] = a1_array
 #       #meteo_df.loc[time_df.index, "$a2$"] = a2_array
 #       #meteo_df.loc[time_df.index, "$a3$"] = a3_array
 #       meteo_df.loc[time_df.index, "$b1$"] = b1_array
 #       #meteo_df.loc[time_df.index, "$b2$"] = b2_array
 #       #meteo_df.loc[time_df.index, "$b3$"] = b3_array
 #       meteo_df.loc[time_df.index, "$c1$"] = c1_array
 #       #meteo_df.loc[time_df.index, "$c2$"] = c2_array
 #       #meteo_df.loc[time_df.index, "$c3$"] = c3_array
 #   meteo_df.loc[time_df.index, "$M$"] = M_array
 #   meteo_df.loc[time_df.index, "$Q_{min}$"] = qmin_array
 #   meteo_df.loc[time_df.index, "$Q_{max}$"] = qmax_array
 #   meteo_df.loc[time_df.index, "$Q_{mean}$"] = qmean_array
 #   meteo_df.loc[time_df.index, "Entropy"] = h_array
    meteo_df["Day of the year"] = daily_summer_df.index.dayofyear
    meteo_df.to_csv(dataframe_filename)

    x_data1_df.to_csv(file_paths.x_data1_filename)
    y_data1_df.to_csv(file_paths.y_data1_filename)
    x_fit1_df.to_csv(file_paths.x_fit1_filename)
    y_fit1_df.to_csv(file_paths.y_fit1_filename)
    r21_df.to_csv(file_paths.r21_filename)
    x_data2_df.to_csv(file_paths.x_data2_filename)
    y_data2_df.to_csv(file_paths.y_data2_filename)
    t_fit2_df.to_csv(file_paths.t_fit2_filename)
    y_fit2_df.to_csv(file_paths.y_fit2_filename)
    r22_df.to_csv(file_paths.r22_filename)



def apply_downscaling_to_daily_discharge(meteo_df, months, kde_dict, function,
                                         daily_discharge, qmin_regr, qmax_regr,
                                         FDC_output_file, path, subdaily_nb,
                                         modeled=False, criteria=None):
    print("Starting the downscaling of the discharge.")

    # Open the discharge dataset and get all mean daily discharge data
    if modeled:
        df = pd.read_csv(daily_discharge, header=0, na_values='', usecols=[0, 1],
                         parse_dates=['Date'], date_format='%d/%m/%Y', index_col=0)
    else:
        df = pd.read_csv(daily_discharge, header=0, na_values='', usecols=[0, 1],
                         parse_dates=['Date'], date_format='%d/%m/%Y', index_col=0)
    df.rename(columns={df.columns[0]: 'Discharge'}, inplace=True)
    df = fc.select_months(df, months)

    # Asserts
    if months == [6, 7, 8, 9]:
        years = df.index.year.unique()
        for y in years:
            assert len(df.loc[df.index.strftime('%Y') == str(y)]) == 122
    np.set_printoptions(threshold=sys.maxsize)

    # Get the days where the measured discharge is not available, and remove them,
    # as the computations after cannot handle NaNs.
    nan_indices = (df['Discharge'].isnull() == True).values
    qmean_distrib = df['Discharge'].to_frame().rename(columns={"Discharge": "$Q_{mean}$"})
    if criteria is None:
        pass
    else:
        print("With multiple regression!!")
        for c in criteria:
            qmean_distrib[c] = meteo_df[c]
            arr = (meteo_df[c].isnull() == True).values
            nan_indices = (nan_indices | arr)
    qmean_distrib = qmean_distrib.fillna(-99999)

    # Use the mean daily discharge data to infer the minimum and maximum daily discharges
    qmin_distrib = qmin_regr.predict(qmean_distrib)
    qmax_distrib = qmax_regr.predict(qmean_distrib)

    # Put the NaNs back in
    qmean_distrib[nan_indices] = np.nan
    qmin_distrib[nan_indices] = np.nan
    qmax_distrib[nan_indices] = np.nan

    # Take the distributions to sample them
    nb_days = len(df)
    if len(kde_dict) > 7:
        print("With weather option!!")
        state_of_days = meteo_df["Weather"].values
        print('state_of_days', state_of_days)
        a_distrib = dsf.sample_weather_distribs("$a$", kde_dict, state_of_days)
        b_distrib = dsf.sample_weather_distribs("$b$", kde_dict, state_of_days)
        c_distrib = dsf.sample_weather_distribs("$c$", kde_dict, state_of_days)
        M_distrib = dsf.sample_weather_distribs("$M$", kde_dict, state_of_days)
    else:
        a_distrib = kde_dict["$a$"].resample(nb_days)[0]
        b_distrib = kde_dict["$b$"].resample(nb_days)[0]
        c_distrib = kde_dict["$c$"].resample(nb_days)[0]
        M_distrib = kde_dict["$M$"].resample(nb_days)[0]

    all_y_fits = []
    for i, day in enumerate(df.index):
        print(i, day)
        q_min = qmin_distrib[i]
        q_max = qmax_distrib[i]
        print("Nb days:", nb_days)
        a = a_distrib[i]
        b = b_distrib[i]
        c = c_distrib[i]
        M = M_distrib[i]
        if q_min < 0:
            q_min = 0
        print("a", a, "b", b, "c", c, "q_min", q_min, "q_max", q_max)
    #    if not np.isnan(q_max) and not np.isnan(q_min):
    #        assert q_max >= q_min

        # Generate fitted curve for plotting
        t_fit = np.linspace(0, 1, subdaily_nb)
        if function == "Singh2014":
            y_fit = fit.discharge_time_equation_Singh2014(t_fit, a, b, q_min, q_max, M)
        elif function == "Sigmoid_d":
            y_fit = fit.discharge_time_equation_Sigmoid_d(t_fit, a, b, c, d, q_min, q_max, M)
        elif function == "Sigmoid":
            y_fit = fit.discharge_time_equation_Sigmoid(t_fit, a, b, c, q_min, q_max, M)

        print("Min, Max, ", np.min(y_fit), np.max(y_fit))

        all_y_fits.extend(y_fit)

    # Recreate the 15-min simulation intervals for the FDCs
    number_of_days = len(df.index)
    in_day_increment = list(range(96)) * int(number_of_days)

    # Add the 15-min intervals to the datetimes
    FDCs_time = [day + np.timedelta64(i * 15, 'm') for day in df.index for i in range(96)]
    FDCs_df = pd.DataFrame({'Date': FDCs_time, 'Discharge': all_y_fits})
    FDCs_df = FDCs_df.set_index('Date')

#    # Asserts (working but super long)
#    if months == [6, 7, 8, 9]:
#        years = FDCs_df.index.year.unique()
#        for y in years:
#            assert len(FDCs_df.loc[FDCs_df.index.strftime('%Y') == str(y)]) == 122 * 96

    pd.DataFrame(a_distrib).to_csv(path + "a_distrib.csv")
    pd.DataFrame(b_distrib).to_csv(path + "b_distrib.csv")
    pd.DataFrame(c_distrib).to_csv(path + "c_distrib.csv")
    pd.DataFrame(M_distrib).to_csv(path + "M_distrib.csv")

    FDCs_df.to_csv(FDC_output_file)

    return FDCs_df




