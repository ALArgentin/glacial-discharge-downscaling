import csv
import sys
from datetime import datetime

import distribution_fitting as dsf
import extract_hydrological_variables as fc
import numpy as np
import pandas as pd
import pome_fitting as fit
import rpy2.robjects as ro
import scipy.stats as stats
from preprocessing import get_statistics, retrieve_subdaily_discharge
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

class DownscalingModel():
    def __init__(self, file_paths, function="Singh2014", months=[6, 7, 8, 9], 
                 subdaily_intervals=96):
        """
        Initialize the DownscalingModel class with necessary file paths,
        downscaling function, and parameters.

        @param file_paths
            Object containing paths to input/output files.
        @param function (str)
            The function to use for the downscaling, between:
            - "Singh2014": the original function
            - "Sigmoid_d": the glacial function
            - "Sigmoid": the simplified glacial function
        @param subdaily_intervals
            Number of sub-daily intervals (e.g., 96 for 15-min intervals).
        """
        ## (FilePaths) Object containing all file paths, used to give the file path and name for
        ## input and output files.
        self.file_paths = file_paths
        
        
        self.function = function
        
        ## (int) Number of sub-daily intervals per day (e.g., 96 for 15-minute intervals).
        self.subdaily_intervals = subdaily_intervals
        
        ## (list of int) List of months to which the analysis is restricted (e.g., 
        ## [6, 7, 8, 9] for summer).
        self.months = months
        
        ## (pandas.DataFrame) Meteorological data containing relevant predictors
        ## (e.g., temperature, precipitation) for criteria-based downscaling, 
        ## and the calibration results.
        self.meteo_df = None

    def simulate_a_flow_duration_curve(self, params, q_min, q_max, M, day_meteo):
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
    
        @return y_fit (?) Return the simulated discharge.
        """
        # Extract the values of the params
        if np.isnan(params).any():
            a = b = c = d = np.nan
        else:
            if self.function == "Singh2014":
                a, b = params
            elif self.function == "Sigmoid_d":
                a, b, c, d = params
            elif self.function == "Sigmoid":
                a, b, c = params
            elif self.function == "Sigmoid_ext_var":
                var1 = day_meteo['Temperature'].values[0]
                var2 = day_meteo['Precipitation'].values[0]
                var3 = day_meteo['Radiation'].values[0]
                #a1, a2, a3, b1, b2, b3, c1, c2, c3 = params
                a1, b1, c1 = params
                a = a1 * var1 #+ a2 * var2 + a3 * var3
                b = b1 * var1 #+ b2 * var2 + b3 * var3
                c = c1 * var1 #+ c2 * var2 + c3 * var3
    
        # Generate fitted curve for plotting
        t_fit = np.linspace(0, 1, self.subdaily_intervals)
        if self.function == "Singh2014":
            y_fit = fit.discharge_time_equation_Singh2014(t_fit, a, b, q_min, q_max, M)
        elif self.function == "Sigmoid_d":
            y_fit = fit.discharge_time_equation_Sigmoid_d(t_fit, a, b, c, d, q_min, q_max, M)
        elif self.function == "Sigmoid" or function == "Sigmoid_ext_var":
            y_fit = fit.discharge_time_equation_Sigmoid(t_fit, a, b, c, q_min, q_max, M)
    
        # Return the simulated discharge
        return y_fit

    def calibration_workflow(self, meteo_df):
        """
        Calibration workflow.
    
        @param meteo_df (dataframe)
            The daily meteorological dataset, which also contains the
            calibrated downscaling parameters.
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
        x_data1_df = pd.DataFrame(columns=np.arange(self.subdaily_intervals))
        y_data1_df = pd.DataFrame(columns=np.arange(self.subdaily_intervals))
        x_fit1_df = pd.DataFrame(columns=np.arange(self.subdaily_intervals))
        y_fit1_df = pd.DataFrame(columns=np.arange(self.subdaily_intervals))
        r21_df = pd.DataFrame(columns=[0])
        x_data2_df = pd.DataFrame(columns=np.arange(self.subdaily_intervals))
        y_data2_df = pd.DataFrame(columns=np.arange(self.subdaily_intervals))
        t_fit2_df = pd.DataFrame(columns=np.arange(self.subdaily_intervals))
        y_fit2_df = pd.DataFrame(columns=np.arange(self.subdaily_intervals))
        r22_df = pd.DataFrame(columns=[0])
    
        # Open the discharge dataset and get all data points from the requested date
        df = pd.read_csv(self.file_paths.subdaily_discharge, header=0, na_values='', usecols=[0, 1],
                         parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
        df.rename(columns={df.columns[1]: 'Discharge'}, inplace=True)
        # Change the index format
        df.index = pd.to_datetime(df.iloc[:,0].values, format='%Y-%m-%d')
    
        meteo_df_str_index = meteo_df.index.strftime('%Y-%m-%d')
    
        daily_df = df.groupby(pd.Grouper(freq='D')).mean()
        daily_summer_df = fc.select_months(daily_df, self.months)
    
        start_t = datetime.now()
    
        time_df = df
        if self.function == "Sigmoid_ext_var":
            time_df = meteo_df
    
        date_range = np.unique(fc.select_months(time_df, self.months).index.strftime('%Y-%m-%d'))
        for i, day in enumerate(date_range):
            if day.endswith("-06-01"):
                print(f"Processing year {day}: {(datetime.now() - start_t).seconds} s spent")
            observed_subdaily_discharge = retrieve_subdaily_discharge(df, day=day)
            day_meteo = meteo_df[meteo_df_str_index == day]
    
            # CAREFUL, in this function I switched X_data and y_data in the sigmoid functions.... Consequently also switched for the rest.
            params, cov, x_data1, y_data1, x_fit1, y_fit1, r21 = fit.fit_a_and_b_to_discharge_probability_curve(
                observed_subdaily_discharge, day_meteo, function=self.function)
            x_data1_df.loc[day] = x_data1
            y_data1_df.loc[day] = y_data1
            if len(x_fit1) != 0: x_fit1_df.loc[day] = x_fit1
            if len(y_fit1) != 0: y_fit1_df.loc[day] = y_fit1
            if r21: r21_df.loc[day] = r21
            q_min, q_mean, q_max = get_statistics(observed_subdaily_discharge)
            M, var, x_data2, y_data2, t_fit2, y_fit2, r22 = fit.fit_m_to_flow_duration_curve(
                observed_subdaily_discharge, params, q_min, q_max, day_meteo, function=self.function)
            x_data2_df.loc[day] = x_data2
            y_data2_df.loc[day] = y_data2
            if len(t_fit2) != 0: t_fit2_df.loc[day] = t_fit2
            if len(y_fit2) != 0: y_fit2_df.loc[day] = y_fit2
            if r22: r22_df.loc[day] = r22
            simulated_subdaily_distribution = self.simulate_a_flow_duration_curve(params, q_min, q_max, M, day_meteo)
    
            # Extract the values of the params
            if np.isnan(params).any():
                a = b = c = d = np.nan
            else:
                if self.function == "Singh2014":
                    a, b = params
                elif self.function == "Sigmoid_d":
                    a, b, c, d = params
                elif self.function == "Sigmoid":
                    a, b, c = params
                elif self.function == "Sigmoid_ext_var":
                    var1 = day_meteo['Temperature'].values[0]
                    var2 = day_meteo['Precipitation'].values[0]
                    var3 = day_meteo['Radiation'].values[0]
                    #a1, a2, a3, b1, b2, b3, c1, c2, c3 = params
                    a1, b1, c1 = params
                    a = a1 * var1 #+ a2 * var2 + a3 * var3
                    b = b1 * var1 #+ b2 * var2 + b3 * var3
                    c = c1 * var1 #+ c2 * var2 + c3 * var3
    
            if self.function == "Sigmoid_d":
                c_array.append(c)
                d_array.append(d)
            elif self.function == "Sigmoid":
                c_array.append(c)
            elif self.function == "Sigmoid_ext_var":
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
        if self.function == "Sigmoid_d":
            data_to_assign.update({"$c$": c_array, "$d$": d_array})
        elif self.function == "Sigmoid":
            data_to_assign.update({"$c$": c_array})
        elif self.function == "Sigmoid_ext_var":
            data_to_assign.update({
                "$c$": c_array,
                "$a1$": a1_array,
                "$b1$": b1_array,
                "$c1$": c1_array,
                "$a2$": a2_array,
                "$b2$": b2_array,
                "$c2$": c2_array,
                "$a3$": a3_array,
                "$b3$": b3_array,
                "$c3$": c3_array
            })
    
        for key, value in data_to_assign.items():
            meteo_df.loc[:, key] = np.nan
            meteo_df.loc[:, key].loc[date_range] = value
    
        meteo_df["Day of the year"] = daily_summer_df.index.dayofyear
        meteo_df.to_csv(self.file_paths.dataframe_filename)
        self.meteo_df = meteo_df
    
        x_data1_df.to_csv(self.file_paths.x_data1_filename)
        y_data1_df.to_csv(self.file_paths.y_data1_filename)
        x_fit1_df.to_csv(self.file_paths.x_fit1_filename)
        y_fit1_df.to_csv(self.file_paths.y_fit1_filename)
        r21_df.to_csv(self.file_paths.r21_filename)
        x_data2_df.to_csv(self.file_paths.x_data2_filename)
        y_data2_df.to_csv(self.file_paths.y_data2_filename)
        t_fit2_df.to_csv(self.file_paths.t_fit2_filename)
        y_fit2_df.to_csv(self.file_paths.y_fit2_filename)
        r22_df.to_csv(self.file_paths.r22_filename)
        
    def apply_gam_min_max_downscaling(self, df):
        qmean_distrib = df['Discharge'].to_frame().rename(columns={"Discharge": "$Q_{mean}$"})
        
        # Rename columns in the DataFrame
        sel_data = self.meteo_df.copy()
        column_mapping = {
            'Glacier area percentage': 'GAP',
            'Precipitation': 'P',
            'Temperature': 'Tp',
            'Radiation': 'R',
            'Snow melt': 'SM',
            'Ice melt': 'IM',
            'All snow': 'AS',
            'Glacier snow': 'GS',
            '$a$': 'a',
            '$b$': 'b',
            '$M$': 'M',
            '$Q_{min}$': 'Qmin',
            '$Q_{max}$': 'Qmax',
            '$Q_{mean}$': 'Qmean',
            'Entropy': 'E',
            '$c$': 'c',
            'Weather': 'W'
        }
        sel_data.rename(columns=column_mapping, inplace=True)
        
        # Activate pandas to R DataFrame conversion
        pandas2ri.activate()
        
        # Load the fitted model for minimum discharge and use it for predictions
        fitted_model = ro.r(f'readRDS("{self.file_paths.gam_min_model}")')
        new_data_r = pandas2ri.py2rpy(sel_data)  # Convert new data to R dataframe
        predictions = ro.r('predict')(fitted_model, new_data_r)  # Make predictions
        min_distrib = np.array(predictions)  # Convert predictions back to Python
        
        # Load the fitted model for maximum discharge and use it for predictions
        fitted_model = ro.r(f'readRDS("{self.file_paths.gam_max_model}")')
        new_data_r = pandas2ri.py2rpy(sel_data)  # Convert new data to R dataframe
        predictions = ro.r('predict')(fitted_model, new_data_r)  # Make predictions
        max_distrib = np.array(predictions)  # Convert predictions back to Python
        
        return min_distrib, max_distrib

    def apply_linear_min_max_downscaling(self, df, qmin_regr, qmax_regr, 
                                         criteria=None):
        """
        Applies linear regression-based downscaling to estimate daily minimum 
        and maximum discharge values from mean daily discharge data.
    
        This function leverages regression models to compute the minimum and 
        maximum daily discharge values based on input mean daily discharge data 
        and additional meteorological predictors, if specified.
    
        @param df (pandas.DataFrame)
            A DataFrame containing mean daily discharge data with one column 
            labeled as 'Discharge'. Missing discharge data is handled by 
            imputing NaNs into the resulting distributions.
        @param qmin_regr (sklearn model)
            Regression model for estimating the minimum daily discharge.
        @param qmax_regr (sklearn model)
            Regression model for estimating the maximum daily discharge.
        @param criteria (list of str, optional)
            List of additional meteorological predictors to include in the 
            downscaling process. These predictors are extracted from `self.meteo_df`. 
            If None, no additional predictors are included. Default is None.
    
        @return (tuple of pandas.Series)
            A tuple containing two pandas Series:
            - `qmin_distrib`: Estimated minimum daily discharge values.
            - `qmax_distrib`: Estimated maximum daily discharge values.
    
        Notes:
        -----
        - The function assumes that `self.meteo_df` contains the required 
          meteorological predictors specified in `criteria`.
        """
        
        # Get the days where the measured discharge is not available, and remove them,
        # as the computations after cannot handle NaNs.
        nan_indices = (df['Discharge'].isnull() == True).values
        qmean_distrib = df['Discharge'].to_frame().rename(columns={"Discharge": "$Q_{mean}$"})
        if criteria is None:
            pass
        else:
            print("With multiple regression!!")
            for c in criteria:
                qmean_distrib[c] = self.meteo_df[c]
                arr = (self.meteo_df[c].isnull() == True).values
                nan_indices = (nan_indices | arr)
        qmean_distrib = qmean_distrib.fillna(-99999)
    
        # Use the mean daily discharge data to infer the minimum and maximum daily discharges
        qmin_distrib = qmin_regr.predict(qmean_distrib)
        qmax_distrib = qmax_regr.predict(qmean_distrib)
    
        # Put the NaNs back in
        qmean_distrib[nan_indices] = np.nan
        qmin_distrib[nan_indices] = np.nan
        qmax_distrib[nan_indices] = np.nan
        
        return qmin_distrib, qmax_distrib

    def select_other_parameters(self, df, kde_dict):
        # Take the distributions to sample them
        nb_days = len(df)
        if len(kde_dict) > 7: # With weather
            state_of_days = self.meteo_df["Weather"].values
            a_distrib = dsf.sample_weather_distribs("$a$", kde_dict, state_of_days)
            b_distrib = dsf.sample_weather_distribs("$b$", kde_dict, state_of_days)
            c_distrib = dsf.sample_weather_distribs("$c$", kde_dict, state_of_days)
            M_distrib = dsf.sample_weather_distribs("$M$", kde_dict, state_of_days)
        else: # Without weather
            a_distrib = kde_dict["$a$"].resample(nb_days, seed=42)[0]
            b_distrib = kde_dict["$b$"].resample(nb_days, seed=42)[0]
            c_distrib = kde_dict["$c$"].resample(nb_days, seed=42)[0]
            M_distrib = kde_dict["$M$"].resample(nb_days, seed=42)[0]

        return a_distrib, b_distrib, c_distrib, M_distrib

    def apply_downscaling_to_daily_discharge(self, kde_dict,
                                             qmin_regr, qmax_regr, FDC_output_file, 
                                             modeled=False, criteria=None, debug=False,
                                             min_max_gam=True):
        """
        Applies downscaling to daily discharge data to generate sub-daily discharge 
        flow durations.
    
        This function takes daily discharge data and applies statistical downscaling 
        based on regression models, distribution fitting, and specified criteria
        to compute sub-daily discharge intervals.
    
        @param kde_dict (dict)
            Dictionary containing kernel density estimation (KDE) distributions for 
            downscaling parameters (e.g., `$a$`, `$b$`, `$c$`, `$M$`).
        @param qmin_regr (sklearn model)
            Regression model for estimating the minimum daily discharge.
        @param qmax_regr (sklearn model)
            Regression model for estimating the maximum daily discharge.
        @param FDC_output_file (str)
            Path to the output CSV file where the generated flow duration curves 
            (FDCs) will be saved.
        @param modeled (bool, optional)
            Whether the input discharge data is modeled or observed. Default is False.
        @param criteria (list of str, optional)
            List of additional meteorological predictors to include in the downscaling 
            process. Default is None.
        @param debug (bool, optional)
            If True, enables additional assertions and debugging outputs. Default 
            is False.
        @param min_max_gam (bool, optional)
            If True, uses GAM to downscale the minimum and maximum daily discharge. 
            If False, uses linear regression. Default is True.
    
        @return (pandas.DataFrame)
            A DataFrame containing the downscaled sub-daily discharge data with 
            15-minute intervals.
    
        Notes:
        -----
        - Missing discharge or meteorological data is handled by imputing NaNs into 
          the generated distributions.
        - Distributions of downscaling parameters are saved as CSV files in the 
          specified path for debugging or further analysis.
        """
        print("Starting the downscaling of the discharge.")
    
        # Open the discharge dataset and get all mean daily discharge data
        if modeled:
            df = pd.read_csv(self.file_paths.simulated_daily_discharge_m3s, header=0, na_values='', usecols=[0, 1],
                             parse_dates=['Date'], date_format='%d/%m/%Y', index_col=0)
        else:
            df = pd.read_csv(self.file_paths.observed_daily_discharge, header=0, na_values='', usecols=[0, 1],
                             parse_dates=['Date'], date_format='%d/%m/%Y', index_col=0)
        df.rename(columns={df.columns[0]: 'Discharge'}, inplace=True)
        df = fc.select_months(df, self.months)
    
        # Asserts
        if debug:
            if self.months == [6, 7, 8, 9]:
                years = df.index.year.unique()
                for y in years:
                    assert len(df.loc[df.index.strftime('%Y') == str(y)]) == 122
        np.set_printoptions(threshold=sys.maxsize)
    
        if min_max_gam:
            qmin_distrib, qmax_distrib = self.apply_gam_min_max_downscaling(df)
        else:
            qmin_distrib, qmax_distrib = self.apply_linear_min_max_downscaling(df, qmin_regr, qmax_regr, criteria)
        a_distrib, b_distrib, c_distrib, M_distrib = self.select_other_parameters(df, kde_dict)
    
        # Generate fitted curve for plotting
        all_y_fits = []
        for i, day in enumerate(df.index):
            q_min = qmin_distrib[i]
            q_max = qmax_distrib[i]
            a = a_distrib[i]
            b = b_distrib[i]
            c = c_distrib[i]
            M = M_distrib[i]
            if q_min < 0:
                q_min = 0
        #    if not np.isnan(q_max) and not np.isnan(q_min):
        #        assert q_max >= q_min
    
            t_fit = np.linspace(0, 1, self.subdaily_intervals)
            if self.function == "Singh2014":
                y_fit = fit.discharge_time_equation_Singh2014(t_fit, a, b, q_min, q_max, M)
            elif self.function == "Sigmoid_d":
                y_fit = fit.discharge_time_equation_Sigmoid_d(t_fit, a, b, c, d, q_min, q_max, M)
            elif self.function == "Sigmoid":
                y_fit = fit.discharge_time_equation_Sigmoid(t_fit, a, b, c, q_min, q_max, M)
    
            print("Min, Max, ", np.min(y_fit), np.max(y_fit))
    
            all_y_fits.extend(y_fit)
    
        # Recreate the 15-min simulation intervals for the FDCs
        number_of_days = len(df.index)
        in_day_increment = list(range(self.subdaily_intervals)) * int(number_of_days)
    
        # Add the 15-min intervals to the datetimes
        FDCs_time = [day + np.timedelta64(i * 15, 'm') for day in df.index for i in range(self.subdaily_intervals)]
        FDCs_df = pd.DataFrame({'Date': FDCs_time, 'Discharge': all_y_fits})
        FDCs_df = FDCs_df.set_index('Date')
    
        if debug:
            # Asserts (working but super long)
            if self.months == [6, 7, 8, 9]:
                years = FDCs_df.index.year.unique()
                for y in years:
                    assert len(FDCs_df.loc[FDCs_df.index.strftime('%Y') == str(y)]) == 122 * self.subdaily_intervals
    
        pd.DataFrame(a_distrib).to_csv(self.file_paths.a_distrib_filename)
        pd.DataFrame(b_distrib).to_csv(self.file_paths.b_distrib_filename)
        pd.DataFrame(c_distrib).to_csv(self.file_paths.c_distrib_filename)
        pd.DataFrame(M_distrib).to_csv(self.file_paths.M_distrib_filename)
    
        FDCs_df.to_csv(FDC_output_file)
    
        return FDCs_df
    
    def load_calibrated_results(self):
        self.meteo_df = pd.read_csv(self.file_paths.dataframe_filename, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d')
        self.meteo_df = self.meteo_df.drop(self.meteo_df.columns[0], axis=1)
        self.meteo_df.index = pd.to_datetime(self.meteo_df.index)

    def discard_calibrated_parameter_outliers(self):    
        # Put rows with too high or low a, b, c values to NaNs
        # to get more readable plots & KDEs.
        print("SHOULD I TRY INSTEAD TO REDEFINE a & b ACCORDING TO THE SENSITIVITY ANALYSIS?")
        self.meteo_df.loc[self.meteo_df["$a$"] < -50, :] = np.nan
        self.meteo_df.loc[self.meteo_df["$a$"] > 150, :] = np.nan
        self.meteo_df.loc[self.meteo_df["$b$"] < -10, :] = np.nan
        self.meteo_df.loc[self.meteo_df["$b$"] > 10, :] = np.nan


