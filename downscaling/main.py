import distribution_fitting as dsf
import downscaling_computations as dc
import fit_metrics as fm
import min_max_fitting as mmf
import numpy as np
import pandas as pd
from extract_hydrological_variables import get_meteorological_hydrological_data
from preprocessing import (bootstrapping_observed_FDCs,
                           convert_to_hydrobricks_units)


class FilePaths:
    def __init__(self, path, study_area, catchment, months, function):
        self.path = path
        self.results = f"{path}Outputs/{study_area}/{catchment}/"
        self.months_str = '_'.join(str(m) for m in months)

        self.subdaily_discharge = None
        self.observed_daily_discharge = None
        self.hydro_units_file = None
        self.forcing_file = None
        self.results_file = None
        self.dataframe_filename = None

        ## Path to the Mann-Kendall test results file.
        self.mann_kendall_filename = None
        ## Path to the Seasonal Mann-Kendall test results file.
        self.seasonal_mann_kendall_filename = None
        
        ## @name First Calibration Step Files
        #  Group of attributes for first calibration step files.
        #  @{
        # Observed dataset (x_data1, y_data1), simulated dataset 
        # (x_fit1, y_fit1) and correlation coefficient (r21).
        self.x_data1_filename = None
        self.y_data1_filename = None
        self.x_fit1_filename = None
        self.y_fit1_filename = None
        self.r21_filename = None
        # @}

        ## @name Second Calibration Step Files
        #  Group of attributes for second calibration step files.
        #  @{
        # Observed dataset (x_data2, y_data2), simulated dataset 
        # (x_fit2, y_fit2) and correlation coefficient (r22).
        self.x_data2_filename = None
        self.y_data2_filename = None
        self.t_fit2_filename = None
        self.y_fit2_filename = None
        self.r22_filename = None
        # @}
        
        self.functions_filename = None
        self.linear_regr_filename = None
        
        self.observed_daily_discharge_FDC_output_file = None
        self.simulated_daily_discharge_FDC_output_file = None
        self.observed_FDCs_output_file = None
        self.weather_observed_FDCs_output_file = None
        self.multi_weather_observed_FDCs_output_file = None

        self._set_input_file_paths(catchment, months, function)
        self._set_output_file_paths(months, function)

    def _set_input_file_paths(self, catchment, months, function):
        self.subdaily_discharge = f"{self.path}Outputs/ObservedDischarges/Arolla_15min_discharge_all_corrected_{catchment}.csv"
        self.observed_daily_discharge = f"{self.path}Outputs/ObservedDischarges/Arolla_daily_mean_discharge_{catchment}.csv"
        self.hydro_units_file = f"{self.results}hydro_units.csv"
        self.forcing_file = f"{self.results}/forcing.nc"
        self.results_file = f"{self.results}results.nc"
        self.dataframe_filename = f"{self.results}meteo_df_{function}_{self.months_str}.csv"
        
        self.functions_filename = f"{self.results}pdf_functions_{function}_{self.months_str}.csv"
        self.linear_regr_filename = f"{self.results}linear_regr_{self.months_str}.csv"
        
        self.simulated_daily_discharge = f"{self.path}Outputs/Arolla/{catchment}/best_fit_simulated_discharge_SCEUA_melt:temperature_index_nse.csv"
        self.simulated_daily_discharge_m3s = f"{self.path}Outputs/Arolla/{catchment}/best_fit_simulated_discharge_SCEUA_melt:temperature_index_nse_m3s.csv"

    def _set_output_file_paths(self, months, function):
        """
        @brief Sets the file paths for output files used in the analysis.

        This method generates and assigns file paths for various output files
        based on the input `function` and the string representation of `months`.

        @param months (list of int)
            A list of integers representing the months used in the analysis.
        @param function (str)
            The function to use for the downscaling, between:
            - "Singh2014": the original function
            - "Sigmoid_d": the glacial function
            - "Sigmoid": the simplified glacial function
        """
        self.mann_kendall_filename = f'{self.results}Mann_Kendall_test_results.txt'
        self.seasonal_mann_kendall_filename = f'{self.results}Seasonal_Mann_Kendall_test_results.txt'

        self.x_data1_filename = f"{self.results}x_data1_df_{function}_{self.months_str}.csv"
        self.y_data1_filename = f"{self.results}y_data1_df_{function}_{self.months_str}.csv"
        self.x_fit1_filename = f"{self.results}x_fit1_df_{function}_{self.months_str}.csv"
        self.y_fit1_filename = f"{self.results}y_fit1_df_{function}_{self.months_str}.csv"
        self.r21_filename = f"{self.results}r21_df_{function}_{self.months_str}.csv"

        self.x_data2_filename = f"{self.results}x_data2_df_{function}_{self.months_str}.csv"
        self.y_data2_filename = f"{self.results}y_data2_df_{function}_{self.months_str}.csv"
        self.t_fit2_filename = f"{self.results}t_fit2_df_{function}_{self.months_str}.csv"
        self.y_fit2_filename = f"{self.results}y_fit2_df_{function}_{self.months_str}.csv"
        self.r22_filename = f"{self.results}r22_df_{function}_{self.months_str}.csv"
        
        self.all_bootstrapped_FDCs_output_file = f"{self.path}Outputs/Arolla/bootstrapped_FDCs_example.csv"
        self.observed_daily_discharge_FDC_output_file = f"{self.results}Arolla_15min_FDCs_{catchment}_from_observed_daily_discharge.csv"
        self.simulated_daily_discharge_FDC_output_file = f"{self.results}Arolla_15min_FDCs_{catchment}_from_hydrobricks_daily_discharge.csv"
        self.observed_FDCs_output_file = f"{self.results}Arolla_15min_FDCs_{catchment}_from_observed_15min_discharge.csv"
        self.weather_observed_FDCs_output_file = f"{self.results}Arolla_15min_FDCs_{catchment}_from_observed_daily_discharge_plus_weather.csv"
        self.multi_weather_observed_FDCs_output_file = f"{self.results}Arolla_15min_FDCs_{catchment}_from_observed_daily_discharge_plus_weather_plus_multiregr.csv"

##########################################################################################################
##########################################################################################################
##########################################################################################################

path = "/home/anne-laure/Documents/Datasets/"
catchments = ['BI', 'HGDA', 'TN', 'PI', 'BS', 'VU', 'DB'] # Ordered by area.
function = "Sigmoid" #"Singh2014" "Sigmoid" "Sigmoid_d" "Sigmoid_ext_var"
months = [6, 7, 8, 9]
calibrate = True


if calibrate:
    for catchment in catchments:
        print(f"Now doing catchment {catchment}.")
        fp = FilePaths(path, "Arolla", catchment, months, function)
        meteo_df = get_meteorological_hydrological_data(fp.forcing_file, fp.results_file, fp.hydro_units_file, months,
                                                        melt_model='degree_day', with_debris=False)
        dc.calibration_workflow(meteo_df, fp.subdaily_discharge, function, fp.dataframe_filename, fp, months)

# List of catchment and metric names
metrics = ["catchments", "ref_nses", "ref_kges", "odd_nses", "odd_kges", "wodd_nses", "wodd_kges", "mwodd_nses", "mwodd_kges", "sdd_nses", "sdd_kges"]
metric_dict = {metric: [] for metric in metrics}

# Add all the data to the same dataframe
watershed_path = f'{path}Swiss_discharge/Arolla_discharge/Watersheds_on_dhm25/'
area_txt = '_UpslopeArea_EPSG21781.txt'

for catchment in catchments:
    print(f"Now doing catchment {catchment}.")
    
    fp = FilePaths(path, "Arolla", catchment, months, function)

    # Convert the simulated discharge of Hydrobricks from [mm/day] to [m3/s].
    convert_to_hydrobricks_units(fp.simulated_daily_discharge, watershed_path, area_txt,
                                 fp.simulated_daily_discharge_m3s, opposite_conversion=True,
                                 catchment=catchment)

    meteo_df = pd.read_csv(fp.dataframe_filename, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d')
    meteo_df = meteo_df.drop(meteo_df.columns[0], axis=1)
    meteo_df.index = pd.to_datetime(meteo_df.index)

    fm.mann_kendall_tests(meteo_df, fp)

    # Put rows with too high or low a, b, c values to NaNs
    # to get more readable plots & KDEs.
    print("SHOULD I TRY INSTEAD TO REDEFINE a & b ACCORDING TO THE SENSITIVITY ANALYSIS?")
    meteo_df.loc[meteo_df["$a$"] < -50, :] = np.nan
    meteo_df.loc[meteo_df["$a$"] > 150, :] = np.nan
    meteo_df.loc[meteo_df["$b$"] < -10, :] = np.nan
    meteo_df.loc[meteo_df["$b$"] > 10, :] = np.nan

    dsf.find_and_save_best_pdf_functions(meteo_df, fp.functions_filename, function)
    weather_kde_dict = dsf.KDE_computations(meteo_df, function, weather=True)
    kde_dict = dsf.KDE_computations(meteo_df, function, weather=False)
    qmin_regr, qmax_regr, qmin_multi_regr, qmax_multi_regr = mmf.extract_discharge_relation_to_daily_mean(meteo_df, fp.linear_regr_filename, criteria=['Ice melt', 'Snow melt'])
    observed_daily_discharge_FDCs_df = dc.apply_downscaling_to_daily_discharge(meteo_df, months, kde_dict, function, fp.observed_daily_discharge,
                                                                               qmin_regr, qmax_regr, fp.observed_daily_discharge_FDC_output_file,
                                                                               fp.results, subdaily_nb=96)
    weather_observed_daily_discharge_FDCs_df = dc.apply_downscaling_to_daily_discharge(meteo_df, months, weather_kde_dict, function, fp.observed_daily_discharge,
                                                                                       qmin_regr, qmax_regr, fp.weather_observed_FDCs_output_file,
                                                                                       fp.results, subdaily_nb=96)
    multi_weather_observed_daily_discharge_FDCs_df = dc.apply_downscaling_to_daily_discharge(meteo_df, months, weather_kde_dict, function, fp.observed_daily_discharge,
                                                                                             qmin_multi_regr, qmax_multi_regr, fp.multi_weather_observed_FDCs_output_file,
                                                                                             fp.results, subdaily_nb=96, criteria=['Ice melt', 'Snow melt'])
    simulated_daily_discharge_FDCs_df = dc.apply_downscaling_to_daily_discharge(meteo_df, months, kde_dict, function, fp.simulated_daily_discharge_m3s,
                                                                                qmin_regr, qmax_regr, fp.simulated_daily_discharge_FDC_output_file,
                                                                                fp.results, subdaily_nb=96, modeled=True)

    observed_FDCs_df, cleaned_observed_FDCs_df, all_bootstrapped_FDCs_dfs = bootstrapping_observed_FDCs(subdaily_discharge, fp.observed_FDCs_output_file, months)
    all_bootstrapped_FDCs_dfs.to_csv(fp.all_bootstrapped_FDCs_output_file)
    ref_nse = fm.compute_reference_metric(all_bootstrapped_FDCs_dfs, cleaned_observed_FDCs_df, 'nse')
    ref_kge = fm.compute_reference_metric(all_bootstrapped_FDCs_dfs, cleaned_observed_FDCs_df, 'kge_2012')

    odd_nse = fm.compute_metric(observed_daily_discharge_FDCs_df, cleaned_observed_FDCs_df, months, 'nse')
    odd_kge = fm.compute_metric(observed_daily_discharge_FDCs_df, cleaned_observed_FDCs_df, months, 'kge_2012')
    wodd_nse = fm.compute_metric(weather_observed_daily_discharge_FDCs_df, cleaned_observed_FDCs_df, months, 'nse')
    wodd_kge = fm.compute_metric(weather_observed_daily_discharge_FDCs_df, cleaned_observed_FDCs_df, months, 'kge_2012')
    mwodd_nse = fm.compute_metric(multi_weather_observed_daily_discharge_FDCs_df, cleaned_observed_FDCs_df, months, 'nse')
    mwodd_kge = fm.compute_metric(multi_weather_observed_daily_discharge_FDCs_df, cleaned_observed_FDCs_df, months, 'kge_2012')
    sdd_nse = fm.compute_metric(simulated_daily_discharge_FDCs_df, cleaned_observed_FDCs_df, months, 'nse')
    sdd_kge = fm.compute_metric(simulated_daily_discharge_FDCs_df, cleaned_observed_FDCs_df, months, 'kge_2012')

    # Append the metrics for the current catchment as a row in the DataFrame
    metric_dict["catchments"].append(catchment)
    metric_dict["ref_nses"].append(ref_nse)
    metric_dict["ref_kges"].append(ref_kge)
    metric_dict["odd_nses"].append(odd_nse)
    metric_dict["odd_kges"].append(odd_kge)
    metric_dict["wodd_nses"].append(wodd_nse)
    metric_dict["wodd_kges"].append(wodd_kge)
    metric_dict["mwodd_nses"].append(mwodd_nse)
    metric_dict["mwodd_kges"].append(mwodd_kge)
    metric_dict["sdd_nses"].append(sdd_nse)
    metric_dict["sdd_kges"].append(sdd_kge)

    #bootstrapped_FDCs, ref_nse, nse = bootstrapping_observed_FDCs(observed_daily_discharge_FDCs_df, fp.subdaily_discharge, fp.observed_FDCs_output_file, months, 'nse')
    #ref_nses1.append(ref_nse)
    #nses1.append(nse)
    #_, w_ref_nse, w_nse = bootstrapping_observed_FDCs(weather_observed_daily_discharge_FDCs_df, fp.subdaily_discharge, fp.observed_FDCs_output_file, months, 'nse')
    #w_ref_nses1.append(w_ref_nse)
    #w_nses1.append(w_nse)
    #_, ref_nse, nse = bootstrapping_observed_FDCs(simulated_daily_discharge_FDCs_df, fp.subdaily_discharge, fp.observed_FDCs_output_file, months, 'nse')
    #ref_nses2.append(ref_nse)
    #nses2.append(nse)
    #_, ref_kge_2012, kge_2012 = bootstrapping_observed_FDCs(observed_daily_discharge_FDCs_df, fp.subdaily_discharge, fp.observed_FDCs_output_file, months, 'kge_2012')
    #ref_kge_2012s1.append(ref_kge_2012)
    #kge_2012s1.append(kge_2012)
    #_, ref_kge_2012, kge_2012 = bootstrapping_observed_FDCs(simulated_daily_discharge_FDCs_df, fp.subdaily_discharge, fp.observed_FDCs_output_file, months, 'kge_2012')
    #ref_kge_2012s2.append(ref_kge_2012)
    #kge_2012s2.append(kge_2012)

metrics_df = pd.DataFrame(metric_dict)
metrics_df.to_csv(f"{path}Outputs/Arolla/downscaling_metrics.csv")

print("Finished")


