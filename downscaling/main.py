import distribution_fitting as dsf
import downscaling_computations as dc
import fit_metrics as fm
import min_max_fitting as mmf
import numpy as np
import pandas as pd
from extract_hydrological_variables import get_meteorological_hydrological_data
from preprocessing import (bootstrapping_observed_FDCs,
                           convert_to_hydrobricks_units)



path = "/home/anne-laure/Documents/Datasets/"
catchments = ['BI', 'HGDA', 'TN', 'PI', 'BS', 'VU', 'DB'] # Ordered by area.
function = "Sigmoid" #"Singh2014" "Sigmoid" "Sigmoid_d" "Sigmoid_ext_var"
months = [6, 7, 8, 9]
calibrate = True


if calibrate:
    for catchment in catchments:
        print(f"Now doing catchment {catchment}.")
        fp = FilePaths(path, "Arolla", catchment, months, function)
        meteo_df = get_meteorological_hydrological_data(fp, months, melt_model='degree_day', with_debris=False)
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
    observed_daily_discharge_FDCs_df = dc.apply_downscaling_to_daily_discharge(meteo_df, months, kde_dict, function,
                                                                               qmin_regr, qmax_regr, fp.observed_daily_discharge_FDC_output_file,
                                                                               fp, subdaily_nb=96)
    weather_observed_daily_discharge_FDCs_df = dc.apply_downscaling_to_daily_discharge(meteo_df, months, weather_kde_dict, function,
                                                                                       qmin_regr, qmax_regr, fp.weather_observed_FDCs_output_file,
                                                                                       fp, subdaily_nb=96)
    multi_weather_observed_daily_discharge_FDCs_df = dc.apply_downscaling_to_daily_discharge(meteo_df, months, weather_kde_dict, function,
                                                                                             qmin_multi_regr, qmax_multi_regr, fp.multi_weather_observed_FDCs_output_file,
                                                                                             fp, subdaily_nb=96, criteria=['Ice melt', 'Snow melt'])
    simulated_daily_discharge_FDCs_df = dc.apply_downscaling_to_daily_discharge(meteo_df, months, kde_dict, function,
                                                                                qmin_regr, qmax_regr, fp.simulated_daily_discharge_FDC_output_file,
                                                                                fp, subdaily_nb=96, modeled=True)

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


