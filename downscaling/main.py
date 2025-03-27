import distribution_fitting as dsf
import downscaling_computations as dc
import fit_metrics as fm
import min_max_fitting as mmf
import numpy as np
import pandas as pd
from extract_hydrological_variables import get_meteorological_hydrological_data
from file_paths import FilePaths
from preprocessing import (bootstrapping_observed_FDCs,
                           convert_to_hydrobricks_units)
from visualization.plot_downscaling_computations import *


path = "/home/anne-laure/Documents/Datasets/"
catchments = ['BI', 'HGDA', 'TN', 'PI', 'BS', 'VU', 'DB'] # Ordered by area.
function = "Sigmoid" #"Singh2014" "Sigmoid" "Sigmoid_d" "Sigmoid_ext_var"
months = [6, 7, 8, 9]
calibrate = True


if False:
    for catchment in catchments:
        fp = FilePaths(path, "Arolla", catchment, months, function)
        model = dc.DownscalingModel(fp, function, months, subdaily_intervals=96)
        if calibrate:
            print(f"Now doing catchment {catchment}.")
            meteo_df = get_meteorological_hydrological_data(fp, months, melt_model='degree_day', with_debris=False)
            model.calibration_workflow(meteo_df)
        else:
            model.load_calibrated_results()
    
        # Convert the simulated discharge of Hydrobricks from [mm/day] to [m3/s].
        convert_to_hydrobricks_units(fp.simulated_daily_discharge, fp.watershed_path,
                                     fp.simulated_daily_discharge_m3s, catchment)
    
        fm.mann_kendall_tests(model.meteo_df, fp)
        
        model.discard_calibrated_parameter_outliers()
    
        dsf.find_and_save_best_pdf_functions(model.meteo_df, fp.functions_filename, function)
        weather_kde_dict = dsf.KDE_computations(model.meteo_df, function, weather=True)
        kde_dict = dsf.KDE_computations(model.meteo_df, function, weather=False)
        
        mmf.gam_on_discharge(model.meteo_df, catchment, fp)
        mmf.gam_on_discharge(model.meteo_df, catchment, fp, '2009-06-01', '2014-09-30', excluded_periods) # BUGS for VU!!!
        qmin_regr, qmax_regr, qmin_multi_regr, qmax_multi_regr = mmf.extract_discharge_relation_to_daily_mean(model.meteo_df, fp.linear_regr_filename, criteria=['Ice melt', 'Snow melt'])
        
        observed_daily_discharge_FDCs_df = model.apply_downscaling_to_daily_discharge(kde_dict, qmin_regr, qmax_regr, fp.observed_daily_discharge_FDCs)
        weather_observed_daily_discharge_FDCs_df = model.apply_downscaling_to_daily_discharge(weather_kde_dict, qmin_regr, qmax_regr, fp.weather_observed_daily_discharge_FDCs)
        multi_weather_observed_daily_discharge_FDCs_df = model.apply_downscaling_to_daily_discharge(weather_kde_dict, qmin_multi_regr, qmax_multi_regr, fp.multi_weather_observed_daily_discharge_FDCs,
                                                                                                 criteria=['Ice melt', 'Snow melt'])
        simulated_daily_discharge_FDCs_df = model.apply_downscaling_to_daily_discharge(kde_dict, qmin_regr, qmax_regr, fp.simulated_daily_discharge_FDCs, modeled=True)
    
        observed_FDCs_df, cleaned_observed_FDCs_df, all_bootstrapped_FDCs_dfs = bootstrapping_observed_FDCs(fp.subdaily_discharge, fp.observed_15min_discharge_FDCs, months)
        all_bootstrapped_FDCs_dfs.to_csv(fp.all_bootstrapped_discharge_FDCs)
        
        metrics = ['rmse', 'ned', 'nrmse_range', 'nrmse_iqr', 'nrmse_mean', 'norm_max_dist']
        refs = fm.compute_reference_metric(all_bootstrapped_FDCs_dfs, cleaned_observed_FDCs_df, metrics)
    
        # Compute all the metrics
        # For the regression results
        or_ = fm.compute_metric(FDCs_qmean_observed_regr_df, cleaned_observed_FDCs_df, months, metrics)
        orw = fm.compute_metric(FDCs_qmean_observed_regr_weather_df, cleaned_observed_FDCs_df, months, metrics)
        omw = fm.compute_metric(FDCs_qmean_observed_multiregr_weather_df, cleaned_observed_FDCs_df, months, metrics)
        sr_ = fm.compute_metric(FDCs_qmean_simulated_regr_df, cleaned_observed_FDCs_df, months, metrics)
        srw = fm.compute_metric(FDCs_qmean_simulated_regr_weather_df, cleaned_observed_FDCs_df, months, metrics)
    
        # For the GAM results
        og_ = fm.compute_metric(FDCs_qmean_observed_gam_df, cleaned_observed_FDCs_df, months, metrics)
        ogw = fm.compute_metric(FDCs_qmean_observed_gam_weather_df, cleaned_observed_FDCs_df, months, metrics)
        sg_ = fm.compute_metric(FDCs_qmean_simulated_gam_df, cleaned_observed_FDCs_df, months, metrics)
        sgw = fm.compute_metric(FDCs_qmean_simulated_gam_weather_df, cleaned_observed_FDCs_df, months, metrics)
    
        # Save the computed metrics for the current catchment as a file
        metric_dict = {}
        metric_dict["metrics"] = metrics
        metric_dict["refs"] = refs
        metric_dict["or_"] = or_
        metric_dict["orw"] = orw
        metric_dict["omw"] = omw
        metric_dict["sr_"] = sr_
        metric_dict["srw"] = srw
        metric_dict["og_"] = og_
        metric_dict["ogw"] = ogw
        metric_dict["sg_"] = sg_
        metric_dict["sgw"] = sgw
        metrics_df = pd.DataFrame(metric_dict)
        metrics_df.to_csv(fp.downscaling_metrics)

if True:
    
    ###########################################################################################################
    months_str = '_'.join(str(m) for m in months)
    figures = f"{path}OutputFigures/"
    
    catchments = ['BI', 'HGDA', 'TN', 'PI', 'BS', 'VU', 'DB'] # Ordered by area.
    plot_yearly_discharges('Arolla_15min_discharge_all_corrected_', catchments, path, f'{figures}Arolla_yearly_discharges')
    plot_yearly_discharges('Ferpecle_discharge_all_corrected_', ['BR', 'BL', 'RR', 'MA', 'RO'], path, f'{figures}Ferpecle_yearly_discharges')
    plot_yearly_discharges('Mattertal_discharge_all_corrected_', ['AR', 'FU', 'TR'], path, f'{figures}Mattertal_yearly_discharges')
    
    plot_comparison_of_catchments_distributions(catchments, path, function, months_str, f'{figures}comparison_histograms_{function}_{months_str}')
    plot_comparison_of_catchments_distributions_depending_on_weather(catchments, path, function, months_str, f'{figures}comparison_KDE_weather_{function}_{months_str}')
    
    plot_coefficients_of_determination(catchments, path, function, months_str, f'{figures}coefficients_of_determination_{function}_{months_str}')
    
    ##################################### Arolla ##############################################################
    
    catchment = "BI"
    fp = FilePaths(path, "Arolla", catchment, months, function)
    radiation = False
    
    ###########################################################################################################
    
    meteo_df = pd.read_csv(fp.dataframe_filename, index_col=0)
    
    # Drop rows with too high or low a, b, c values to get more readable plots
    meteo_df.drop(meteo_df[meteo_df["$a$"] < -50].index, inplace=True)
    meteo_df.drop(meteo_df[meteo_df["$a$"] > 150].index, inplace=True)
    meteo_df.drop(meteo_df[meteo_df["$b$"] < -10].index, inplace=True)
    meteo_df.drop(meteo_df[meteo_df["$b$"] > 10].index, inplace=True)
    
    
    linear_regr_filename = f"{fp.results}linear_regr_{months_str}.csv"
    plot_q_mean_q_min_q_max_regressions(meteo_df, linear_regr_filename,
                                        f'{figures}Qmean_Qmin_Qmax_regressions_{catchment}_{months_str}')
    
    plot_FDCs_together(fp.observed_15min_discharge_FDCs, fp.observed_daily_discharge_FDCs, 
                       f'{figures}FDCs_{function}_{catchment}_{months_str}.pdf')
    plot_downscaling_improvement(fp.observed_15min_discharge_FDCs, 
                                 fp.observed_daily_discharge_FDCs, 
                                 fp.weather_observed_daily_discharge_FDCs,
                                 fp.multi_weather_observed_daily_discharge_FDCs,
                                 fp.simulated_daily_discharge_FDCs,
                                 fp.all_bootstrapped_discharge_FDCs, 
                                 fp.downscaling_metrics, catchment,
                                 f'{figures}Improvement_FDCs_{function}_{catchment}_{months_str}.png')
    plot_sampled_distributions(meteo_df, fp.results, function, 
                               f'{figures}all_sampled_histograms_{function}_{catchment}_{months_str}')
    print(nboä)
    
    variables = {"$a$": np.array([5, 10, 25, 50, 75]), "$b$": np.array([0, 0.4, 0.8, 1.2, 1.8]),
                 "$c$": np.linspace(-1, 1, 5),
                 "$M$": np.linspace(0.1, 5.1, 5),
                 "$Q_{min}$": np.linspace(0, 3, 5), "$Q_{max}$": np.linspace(2, 10, 5)}
    plot_function_sensitivity(f'{figures}function_sensitivity_high_threshold_{function}_{catchment}_{months_str}.pdf',
                              variables, function="Sigmoid")
    variables = {"$a$": np.array([0, -1.8, -5, -15, -25]), "$b$": np.array([0, -0.1, -0.2, -0.4, -0.8]), #np.linspace(-20, 0, 5)
                 "$c$": np.linspace(-2, -1, 5),
                 "$M$": np.linspace(-0.1, -5.1, 5),
                 "$Q_{min}$": np.linspace(0, 3, 5), "$Q_{max}$": np.linspace(2, 10, 5)}
    plot_function_sensitivity(f'{figures}function_sensitivity_low_threshold_{function}_{catchment}_{months_str}.pdf',
                              variables, function="Sigmoid")
    
    ################################## Discharge variability plot #############################################
    plot_discharge_variability(['BI', 'HGDA', 'TN', 'PI', 'BS', 'VU'], months_str, function,
                               f"{figures}dicharge_variability_{function}_{catchment}_{months_str}.pdf")
    
    #################################### Pairplots
    pairplot(meteo_df, f"{figures}pairplot_{function}_{catchment}_{months_str}.pdf")
    
    retrieve_subdaily_discharge(fp.subdaily_discharge, figures=figures)
    figure3(fp.subdaily_discharge, fp.results, months_str, figures=figures)
    figure4(fp.subdaily_discharge, fp.results, months_str, figures=figures)
    
    x_data1_df = pd.read_csv(fp.x_data1_filename)
    y_data1_df = pd.read_csv(fp.y_data1_filename)
    x_fit1_df = pd.read_csv(fp.x_fit1_filename)
    y_fit1_df = pd.read_csv(fp.y_fit1_filename)
    r21_df = pd.read_csv(fp.r21_filename)
    x_data2_df = pd.read_csv(fp.x_data2_filename)
    y_data2_df = pd.read_csv(fp.y_data2_filename)
    t_fit2_df = pd.read_csv(fp.t_fit2_filename)
    y_fit2_df = pd.read_csv(fp.y_fit2_filename)
    r22_df = pd.read_csv(fp.r22_filename)
    x_data1_df.columns.values[0] = "Date"
    y_data1_df.columns.values[0] = "Date"
    x_fit1_df.columns.values[0] = "Date"
    y_fit1_df.columns.values[0] = "Date"
    r21_df.columns.values[0] = "Date"
    x_data2_df.columns.values[0] = "Date"
    y_data2_df.columns.values[0] = "Date"
    t_fit2_df.columns.values[0] = "Date"
    y_fit2_df.columns.values[0] = "Date"
    r22_df.columns.values[0] = "Date"
    
    fit_a_and_b_to_discharge_probability_curve(meteo_df, x_data1_df, y_data1_df, x_fit1_df, y_fit1_df, r21_df,
                                               function=function, figures=figures)
    fit_m_to_flow_duration_curve(meteo_df, x_data2_df, y_data2_df, t_fit2_df, y_fit2_df, r22_df, function=function, figures=figures)
    
    #################################### Histograms
    combined_histograms(meteo_df, figures, f'{figures}all_histograms_{function}_{catchment}_{months_str}',
                        function=function)
    
    #################################### PDF fits on top of histograms
    combined_histograms(meteo_df, figures, f'{figures}all_histograms_{function}_{catchment}_{months_str}',
                        function=function, pdfs=fp.functions_filename, kde=True)
    
    for catch in catchments:
        subdaily_discharge = f"{path}Outputs/ObservedDischarges/Arolla_15min_discharge_all_corrected_{catch}.csv"
        Mutzner2015_plot(subdaily_discharge, months, f'{figures}detrended_discharge_Mutzner2015_{catch}_{months_str}.png')
    
    print(vhiöv)
    
    #plot_influence(precip_m, a_array, "Precipitation", "a")
    #plot_influence(precip_m, b_array, "Precipitation", "b")
    #plot_influence(precip_m, c_array, "Precipitation", "c")
    #plot_influence(precip_m, d_array, "Precipitation", "d")
    #plot_influence(precip_m, M_array, "Precipitation", "M")
    #plot_influence(precip_m, qmin_array, "Precipitation", "Qmin")
    #plot_influence(precip_m, qmax_array, "Precipitation", "Qmax")
    
    #plot_influence(temper_m, a_array, "Temperature", "a")
    #plot_influence(temper_m, b_array, "Temperature", "b")
    #plot_influence(temper_m, c_array, "Temperature", "c")
    #plot_influence(temper_m, d_array, "Temperature", "d")
    #plot_influence(temper_m, M_array, "Temperature", "M")
    #plot_influence(temper_m, qmin_array, "Temperature", "Qmin")
    #plot_influence(temper_m, qmax_array, "Temperature", "Qmax")
    
    #sns.heatmap(meteo_df, annot=True)
    
    ################################## Recession flows and subdaily plots #####################################
    
    min_daily_discharge_data, \
    max_discharge_between_min_df, \
    recession_flows = find_minima_and_maxima(fp.subdaily_discharge, start='2010-06-01', end='2010-06-07')

print("Finished")


