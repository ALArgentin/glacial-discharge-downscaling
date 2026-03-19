import sys
from pathlib import Path
# Add the parent folder of 'downscaling' to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from fit_metrics import compute_KolmogorovSmirnov, compute_Wasserstein, compute_r2, compute_metric
from pome_fitting import pome
import warnings


path = "/home/aargentin/Documents/Datasets/"
catchments = ['BI', 'HGDA', 'TN', 'PI', 'BS', 'VU', 'DB'] # Ordered by area.
function = "Sigmoid" #"Singh2014" "Sigmoid" "Sigmoid_d" "Sigmoid_ext_var"
months = [6, 7, 8, 9]
weather_list = ['Freezing', 'Melting', 'Raining', 'Snowing']
calibrate = False


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
            
        # Post-processing of calibrated parameters
        model.discard_calibrated_parameter_outliers(["$a$", "$b$"], 0.05, ["$a$", "$b$", "$c$", "$M$"])
        model.classify_according_to_weather(weather_list)
        model.meteo_df.to_csv(fp.dataframe_constrained_filename)
        
        # Convert the simulated discharge of Hydrobricks from [mm/day] to [m3/s].
        convert_to_hydrobricks_units(fp.simulated_daily_discharge, fp.watershed_path,
                                     fp.simulated_daily_discharge_m3s, catchment)
    
        fm.mann_kendall_tests(model.meteo_df, fp)
    
        dsf.find_and_save_best_pdf_functions(model.meteo_df, fp.functions_filename, function)
        weather_kde_dict = dsf.KDE_computations(model.meteo_df, function, weather_list)
        kde_dict = dsf.KDE_computations(model.meteo_df, function, weather_list=None)
        
        # Fit 1) a GAM, 2) a regression.
        excluded_periods = []
        if catchment == 'VU':
            excluded_periods = [['2012-09-20', '2012-09-28']]
        mmf.gam_on_discharge(model.meteo_df, catchment, fp, '2009-06-01', '2014-09-30', excluded_periods) # BUGS for VU!!!
        qmin_regr, qmax_regr, qmin_multi_regr, qmax_multi_regr = mmf.extract_discharge_relation_to_daily_mean(model.meteo_df, fp.linear_regr_filename, criteria=['Ice melt', 'Snow melt'])
        
        independent = False
        # Downscale a daily discharge in different ways
        # First, with the regressions
        FDCs_qmean_observed_regr_df = model.apply_downscaling_to_daily_discharge(kde_dict, independent, False, qmin_regr, qmax_regr, fp.FDCs_qmean_observed_regr)
        FDCs_qmean_observed_regr_weather_df = model.apply_downscaling_to_daily_discharge(weather_kde_dict, independent, True, qmin_regr, qmax_regr, fp.FDCs_qmean_observed_regr_weather)
        FDCs_qmean_observed_multiregr_weather_df = model.apply_downscaling_to_daily_discharge(weather_kde_dict, independent, True, qmin_multi_regr, qmax_multi_regr, fp.FDCs_qmean_observed_multiregr_weather,
                                                                                              criteria=['Ice melt', 'Snow melt'])
        FDCs_qmean_simulated_regr_df = model.apply_downscaling_to_daily_discharge(kde_dict, independent, False, qmin_regr, qmax_regr, fp.FDCs_qmean_simulated_regr, modeled=True)
        FDCs_qmean_simulated_regr_weather_df = model.apply_downscaling_to_daily_discharge(weather_kde_dict, independent, True, qmin_regr, qmax_regr, fp.FDCs_qmean_simulated_regr_weather, modeled=True)
        # Second, with the GAMs
        FDCs_qmean_observed_gam_df = model.apply_downscaling_to_daily_discharge(kde_dict, independent, False, None, None, fp.FDCs_qmean_observed_gam)
        FDCs_qmean_observed_gam_weather_df = model.apply_downscaling_to_daily_discharge(weather_kde_dict, independent, True, None, None, fp.FDCs_qmean_observed_gam_weather)
        FDCs_qmean_simulated_gam_df = model.apply_downscaling_to_daily_discharge(kde_dict, independent, False, None, None, fp.FDCs_qmean_simulated_gam, modeled=True)
        FDCs_qmean_simulated_gam_weather_df = model.apply_downscaling_to_daily_discharge(weather_kde_dict, independent, True, None, None, fp.FDCs_qmean_simulated_gam_weather, modeled=True)
    
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

    ################################# ADD SOME FIGURES FOR THE REVIEWERS ######################################

    # Do the transferability test between BI and HGDA asked by Reviewer 2
    if True: 
        weather_list = ['Freezing', 'Melting', 'Raining', 'Snowing']
        independent=False
        # Get the BI parameters
        BI_fp = FilePaths(path, "Arolla", "BI", months, function)
        BI_model = dc.DownscalingModel(BI_fp, function, months, subdaily_intervals=96)
        BI_model.load_calibrated_results(BI_fp.dataframe_constrained_filename)
        BI_weather_kde_dict = dsf.KDE_computations(BI_model.meteo_df, function, weather_list)
        # Get the HGDA parameters
        HGDA_fp = FilePaths(path, "Arolla", "HGDA", months, function)
        HGDA_model = dc.DownscalingModel(HGDA_fp, function, months, subdaily_intervals=96)
        HGDA_model.load_calibrated_results(HGDA_fp.dataframe_constrained_filename)
        HGDA_weather_kde_dict = dsf.KDE_computations(HGDA_model.meteo_df, function, weather_list)
        # Apply them to HGDA
        HGDA_FDCs_qmean_observed_gam_weather_df = HGDA_model.apply_downscaling_to_daily_discharge(HGDA_weather_kde_dict, independent, True, None, None, 
                                                                                                  f"{HGDA_fp.results}Transferability_HGDA2HGDA.csv")
        
        HGDA_model.meteo_df["$a$"] = BI_model.meteo_df["$a$"]
        HGDA_model.meteo_df["$b$"] = BI_model.meteo_df["$b$"]
        HGDA_model.meteo_df["$c$"] = BI_model.meteo_df["$c$"]
        HGDA_model.meteo_df["$M$"] = BI_model.meteo_df["$M$"]

        BI_FDCs_qmean_observed_gam_weather_df = HGDA_model.apply_downscaling_to_daily_discharge(BI_weather_kde_dict, independent, True, None, None, 
                                                                                                f"{BI_fp.results}Transferability_BI2HGDA.csv")
        HGDA_comparison_data = HGDA_fp.observed_15min_discharge_FDCs

        FDCs_ref_df = pd.read_csv(HGDA_fp.observed_15min_discharge_FDCs, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
        plt.figure(figsize=(200, 3))
        start_date = '2012-01-01'
        end_date   = '2013-01-01'
        plt.plot(HGDA_FDCs_qmean_observed_gam_weather_df.loc[start_date:end_date].index, 
                HGDA_FDCs_qmean_observed_gam_weather_df.loc[start_date:end_date]['Discharge'], 
                linestyle='-', color='blue', label='Discharge')
        plt.plot(FDCs_ref_df.loc[start_date:end_date].index, 
                FDCs_ref_df.loc[start_date:end_date]['Discharge'], 
                linestyle='-', color='red', label='Discharge')
        plt.xlabel('Date')
        plt.ylabel('Discharge')
        plt.legend()
        plt.tight_layout()
        plt.savefig("plot.png")

        metrics = ['rmse', 'ned', 'nrmse_range', 'nrmse_iqr', 'nrmse_mean', 'norm_max_dist']
        HGDA_metric_values = compute_metric(HGDA_FDCs_qmean_observed_gam_weather_df, FDCs_ref_df, months, metrics, True, True)
        BI_metric_values = compute_metric(BI_FDCs_qmean_observed_gam_weather_df, FDCs_ref_df, months, metrics, True, True)
        print("HGDA metrics:", HGDA_metric_values, "BI metrics:", BI_metric_values)
        plot_downscaling_transferability(HGDA_comparison_data, f"{HGDA_fp.results}Transferability_HGDA2HGDA.csv", 
                                        f"{BI_fp.results}Transferability_BI2HGDA.csv", f"{path}OutputFigures/Transferability.png")
     

    catchment = "BI"
    fp = FilePaths(path, "Arolla", catchment, months, function)
    months_str = '_'.join(str(m) for m in months)
    figures = f"{path}OutputFigures/"
    figure3(path, "Arolla", "BI", months, function, figures=figures)
    precipitation("/home/aargentin/Documents/Datasets/IDAWEB_MeteoSwiss/Arolla/order_124535_data.txt", figures)
    figure4(path, "Arolla", "BI", months, function, figures=figures)

    KS_distances(catchments, path, months, function, figures=figures)
    wasserstein_distances(catchments, path, months, function, figures=figures)
    
    variables = {"$a$": np.array([5, 10, 25, 50, 75]), "$b$": np.array([0, 0.4, 0.8, 1.2, 1.8]),
                 "$c$": np.linspace(-1, 1, 5),
                 "$M$": np.linspace(0.1, 5.1, 5),
                 #"$Q_{min}$": np.linspace(0, 3, 5), "$Q_{max}$": np.linspace(2, 10, 5)
                 }
    plot_function_sensitivity(f'{figures}function_sensitivity_high_threshold_{function}_{catchment}_{months_str}.pdf',
                              variables, function="Sigmoid")
    variables = {"$a$": np.array([0, -1.8, -5, -15, -25]), "$b$": np.array([0, -0.1, -0.2, -0.4, -0.8]), #np.linspace(-20, 0, 5)
                 "$c$": np.linspace(-2, -1, 5),
                 "$M$": np.linspace(-0.1, -5.1, 5),
                 #"$Q_{min}$": np.linspace(0, 3, 5), "$Q_{max}$": np.linspace(2, 10, 5)
                 }
    plot_function_sensitivity(f'{figures}function_sensitivity_low_threshold_{function}_{catchment}_{months_str}.pdf',
                              variables, function="Sigmoid")


    # Do the KS computations asked by Reviewer 2
    if True: 

        start_t = datetime.now()
        for catchment in ["HGDA"]:
            fp = FilePaths(path, "Arolla", catchment, months, function)
            meteo_df = pd.read_csv(fp.dataframe_filename, index_col=0)
            
            x_data_df = pd.read_csv(fp.x_data1_filename, index_col=0, parse_dates=True)
            y_fit_df = pd.read_csv(fp.y_fit1_filename, index_col=0, parse_dates=True)

            # Get the date strings
            date_range = x_data_df.index.astype(str)

            ks_pvalues = np.zeros(len(date_range))
            ks_stats = np.zeros(len(date_range))
            ks_stats2 = np.zeros(len(date_range))
            wasserstein_dist = np.zeros(len(date_range))
            r2 = np.zeros(len(date_range))
            for i, day in enumerate(date_range):
                if day.endswith("-06-01"):
                    print(f"Processing year {day}: {(datetime.now() - start_t).seconds} s spent")
                
                # Extract the row as arrays of 96 values (ignore date index)
                x_data = x_data_df.loc[day].to_numpy(dtype=float)
                y_fit = y_fit_df.loc[day].to_numpy(dtype=float)

                # Skip invalid rows
                if np.any(np.isnan(x_data)) or np.any(np.isnan(y_fit)):
                    print("Skipping day due to NaNs:", day)
                    ks_pvalues[i] = ks_stats[i] = ks_stats2[i] = wasserstein_dist[i] = r2[i] = np.nan
                    continue

                try:
                    ks_pvalues[i], ks_stats[i], ks_stats2[i] = compute_KolmogorovSmirnov(x_data, y_fit)
                    wasserstein_dist[i] = compute_Wasserstein(x_data, y_fit)
                    r2[i] = compute_r2(x_data, y_fit)
                except AssertionError:
                    ks_pvalues[i], ks_stats[i], ks_stats2[i], r2[i] = np.nan, np.nan, np.nan, np.nan  # skip if y_fit is not sorted

            ks_stats_df = pd.DataFrame({"Date": date_range, "KS_pvalue": ks_pvalues, "KS_stat": ks_stats, "KS_stat2": ks_stats2, 
                                        "Wasserstein_dist": wasserstein_dist, "r2": r2, "a": meteo_df['$a$'], "b": meteo_df['$b$']})
            if function != "Singh2014":
                ks_stats_df["c"] = meteo_df['$c$']
            ks_stats_df.to_csv(fp.results + f"KS_{function}.txt", index=False)

                
        flushing_dates = ['2009-05-26', '2010-07-17', '2011-07-12', '2011-07-13', '2011-08-22', 
                          '2011-08-24', '2011-08-25', '2011-08-26', '2011-08-27', '2011-10-02', 
                          '2011-10-18', '2011-11-06', '2012-07-02', '2012-10-22', '2013-05-07', 
                          '2013-06-20', '2013-06-21', '2013-07-29', '2013-08-07', '2013-08-08', 
                          '2013-10-08']
        vu_missing_dates = [('2011-08-26', '2011-12-31')]

        start_t = datetime.now()
        for catchment in catchments:
            print(f"Now processing catchment {catchment}.")
            fp = FilePaths(path, "Arolla", catchment, months, function)
            meteo_df = pd.read_csv(fp.dataframe_filename, index_col=0)
            
            y_data_df = pd.read_csv(fp.y_data2_filename, index_col=0, parse_dates=True)
            y_fit_df = pd.read_csv(fp.y_fit2_filename, index_col=0, parse_dates=True)

            # Get the date strings
            date_range = y_data_df.index.astype(str)

            ks_pvalues = np.zeros(len(date_range))
            ks_stats = np.zeros(len(date_range))
            ks_stats2 = np.zeros(len(date_range))
            wasserstein_dist = np.zeros(len(date_range))
            r2 = np.zeros(len(date_range))
            for i, day in enumerate(date_range):
                if day.endswith("-06-01"):
                    print(f"Processing year {day}: {(datetime.now() - start_t).seconds} s spent")

                # check if day falls in any flushing range
                is_vu_missing = any(pd.Timestamp(start) <= pd.Timestamp(day) <= pd.Timestamp(end) 
                                for start, end in vu_missing_dates)

                if day in flushing_dates or is_vu_missing:
                    print(f"{day} is a flushing day!")
                    y_data = np.nan * np.ones(96)
                    y_fit = np.nan * np.ones(96)
                else:
                    # Extract the row as arrays of 96 values (ignore date index)
                    y_data = y_data_df.loc[day].to_numpy(dtype=float)
                    y_fit = y_fit_df.loc[day].to_numpy(dtype=float)

                # Skip invalid rows
                if np.any(np.isnan(y_data)) or np.any(np.isnan(y_fit)):
                    ks_pvalues[i] = ks_stats[i] = ks_stats2[i] = wasserstein_dist[i] = r2[i] = np.nan
                    continue

                try:
                    ks_pvalues[i], ks_stats[i], ks_stats2[i] = compute_KolmogorovSmirnov(y_data, y_fit)
                    wasserstein_dist[i] = compute_Wasserstein(y_data, y_fit)
                    r2[i] = compute_r2(y_data, y_fit)
                except AssertionError:
                    ks_pvalues[i], ks_stats[i], ks_stats2[i], r2[i] = np.nan, np.nan, np.nan, np.nan  # skip if y_fit is not sorted
            ks_stats_df = pd.DataFrame({"Date": date_range, "KS_pvalue": ks_pvalues, "KS_stat": ks_stats, "KS_stat2": ks_stats2, 
                                        "Wasserstein_dist": wasserstein_dist, "r2": r2, "a": meteo_df['$a$'], "b": meteo_df['$b$']})
            if function != "Singh2014":
                ks_stats_df["c"] = meteo_df['$c$']
            ks_stats_df.to_csv(fp.results + f"KS_{function}_bis.txt", index=False)




    print(vhlvyuol)
    
    ###########################################################################################################
    months_str = '_'.join(str(m) for m in months)
    figures = f"{path}OutputFigures/"
    catchments = ['BI', 'HGDA', 'TN', 'PI', 'BS', 'VU', 'DB'] # Ordered by area.
    plot_comparison_of_catchments_distributions(catchments, path, function, months_str, f'{figures}comparison_histograms_{function}_{months_str}')
    plot_comparison_of_catchments_distributions_depending_on_weather(catchments, path, function, months_str, f'{figures}comparison_KDE_weather_{function}_{months_str}')
    
    ################################## Discharge variability plot #############################################
    meteo_station = "/home/aargentin/Documents/Datasets/IDAWEB_MeteoSwiss/Temperature/"
    plot_discharge_variability(['BI', 'HGDA', 'TN', 'PI', 'BS', 'VU', 'DB'], months_str, function, meteo_station,
                               f"{figures}dicharge_variability_{function}_{months_str}")
    plot_discharge_variability(['BI', 'HGDA', 'TN', 'PI', 'BS', 'VU', 'DB'], months_str, function, meteo_station,
                               f"{figures}dicharge_variability_{function}_{months_str}", yearly=False)
    
    plot_composite_FDCs_by_category(catchments, path, "Arolla", months, function, "Radiation",
                                    f'{figures}composite_FDCs_by_category_{months_str}_Radiation.png')
    plot_composite_FDCs_by_category(catchments, path, "Arolla", months, function, "Temperature",
                                    f'{figures}composite_FDCs_by_category_{months_str}_Temperature.png')
    plot_composite_FDCs_by_category(catchments, path, "Arolla", months, function, "Precipitation",
                                    f'{figures}composite_FDCs_by_category_{months_str}_Precipitation.png')
    
    plot_Qmin_qmax_figure(f'{figures}Qmin_Qmax_figure_{months_str}', catchments, path, months_str, pdfs=True, function="Sigmoid")
    
    plot_yearly_discharges('Arolla_15min_discharge_all_corrected_', catchments, path, f'{figures}Arolla_yearly_discharges')
    plot_yearly_discharges('Ferpecle_discharge_all_corrected_', ['BR', 'BL', 'RR', 'MA', 'RO'], path, f'{figures}Ferpecle_yearly_discharges')
    plot_yearly_discharges('Mattertal_discharge_all_corrected_', ['AR', 'FU', 'TR'], path, f'{figures}Mattertal_yearly_discharges')
    
    plot_coefficients_of_determination(catchments, path, function, months_str, f'{figures}coefficients_of_determination_{function}_{months_str}')
    
    ##################################### Arolla ##############################################################
    
    for catchment in catchments:
        fp = FilePaths(path, "Arolla", catchment, months, function)
        
        ##########################################################################################################
        
        meteo_df = pd.read_csv(fp.dataframe_constrained_filename, index_col=0)
        
        ################################################# Other plot #############################################
        linear_regr_filename = f"{fp.results}linear_regr_{months_str}.csv"
        influences = ['Precipitation', 'Temperature', 'Snow melt', 'Ice melt', 'All snow',
                      'Glacier snow', '$a$', '$b$', '$c$', '$M$', '$Q_{min}$', '$Q_{max}$', '$Q_{mean}$',
                      'Entropy', 'Day of the year', 'Glacier area percentage', 'Radiation']
        for influence in influences:
            plot_q_mean_q_min_q_max_regressions(meteo_df, linear_regr_filename, 
                                                f'{figures}Qmean_Qmin_Qmax_regressions_{catchment}_{months_str}',
                                                influence_label=influence)
        
        plot_FDCs_together(fp.observed_15min_discharge_FDCs, fp.FDCs_qmean_observed_regr, 
                           f'{figures}FDCs_{function}_{catchment}_{months_str}.pdf')
        plot_downscaling_improvement(fp.observed_15min_discharge_FDCs, 
                                     fp.FDCs_qmean_observed_regr, 'Measured $Q_{mean}$', 'or_',
                                     fp.FDCs_qmean_observed_regr_weather, 'Measured $Q_{mean}$ with weather constraint', 'orw',
                                     fp.FDCs_qmean_observed_multiregr_weather, 'Measured $Q_{mean}$ with weather and multiregr', 'omw',
                                     fp.FDCs_qmean_simulated_regr, 'Simulated $Q_{mean}$', 'sr_',
                                     fp.all_bootstrapped_discharge_FDCs, 'refs',
                                     fp.downscaling_metrics, catchment,
                                     f'{figures}Improvement_FDCs_{function}_{catchment}_{months_str}.png')
        plot_downscaling_improvement(fp.observed_15min_discharge_FDCs, 
                                     fp.FDCs_qmean_observed_gam, 'Measured $Q_{mean}$', 'og_',
                                     fp.FDCs_qmean_observed_gam_weather, 'Measured $Q_{mean}$ with weather constraint', 'ogw',
                                     fp.FDCs_qmean_simulated_gam, 'Simulated $Q_{mean}$', 'sg_',
                                     fp.FDCs_qmean_simulated_gam_weather, 'Simulated $Q_{mean}$ with weather constraint', 'sgw',
                                     fp.all_bootstrapped_discharge_FDCs, 'refs',
                                     fp.downscaling_metrics, catchment,
                                     f'{figures}Improvement_FDCs_{function}_{catchment}_{months_str}_GAMs.png')
    
    figure3(path, "Arolla", "BI", months, function, figures=figures)
    figure4(path, "Arolla", "BI", months, function, figures=figures)
        
    catchment = "BI"
    fp = FilePaths(path, "Arolla", catchment, months, function)
    
    plot_sampled_distributions(meteo_df, fp.results, function, 
                               f'{figures}all_sampled_histograms_{function}_{catchment}_{months_str}')
    
    variables = {"$a$": np.array([5, 10, 25, 50, 75]), "$b$": np.array([0, 0.4, 0.8, 1.2, 1.8]),
                 "$c$": np.linspace(-1, 1, 5),
                 "$M$": np.linspace(0.1, 5.1, 5),
                 #"$Q_{min}$": np.linspace(0, 3, 5), "$Q_{max}$": np.linspace(2, 10, 5)
                 }
    plot_function_sensitivity(f'{figures}function_sensitivity_high_threshold_{function}_{catchment}_{months_str}.pdf',
                              variables, function="Sigmoid")
    variables = {"$a$": np.array([0, -1.8, -5, -15, -25]), "$b$": np.array([0, -0.1, -0.2, -0.4, -0.8]), #np.linspace(-20, 0, 5)
                 "$c$": np.linspace(-2, -1, 5),
                 "$M$": np.linspace(-0.1, -5.1, 5),
                 #"$Q_{min}$": np.linspace(0, 3, 5), "$Q_{max}$": np.linspace(2, 10, 5)
                 }
    plot_function_sensitivity(f'{figures}function_sensitivity_low_threshold_{function}_{catchment}_{months_str}.pdf',
                              variables, function="Sigmoid")
    
    #################################### Pairplots
    pairplot(meteo_df, f"{figures}pairplot_{function}_{catchment}_{months_str}.pdf")
    
    retrieve_subdaily_discharge(fp.subdaily_discharge, figures=figures)
    
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


