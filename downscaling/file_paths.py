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
        self.dataframe_constrained_filename = None
        
        self.watershed_path = None
        self.area_txt = None

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
        
        ## @name Distributions of Downscaling Parameters Files
        #  Group of attributes for distributions of downscaling parameter files.
        #  @{
        # These distributions are the distributions obtained from sampling
        # the original distributions randomly and are used to compare with
        # the original distributions and ensure the accuracy of the method.
        self.a_distrib_filename = None
        self.b_distrib_filename = None
        self.c_distrib_filename = None
        self.M_distrib_filename = None
        # @}
        
        self.functions_filename = None
        self.linear_regr_filename = None

        self.observed_15min_discharge_FDCs = None
        self.observed_daily_discharge_FDCs = None
        self.simulated_daily_discharge_FDCs = None
        self.weather_observed_daily_discharge_FDCs = None
        self.multi_weather_observed_daily_discharge_FDCs = None
        self.downscaling_metrics = None
        
        ## @name Generative Additive Model Files
        #  Group of attributes for generative additive model files.
        #  @{
        # The 'gam_diagnostics' file saves the pdf from the 'gam.check(res)'
        # command, while the 'gam_summary' file saves the text output of all
        # assessment commands and the 'gam_model' file saves the GAM model
        # itself for reuse in the future. Once for minimum-mean discharge
        # relationship, once for maximum-mean.
        self.gam_min_diagnostics = None
        self.gam_min_summary = None
        self.gam_min_model = None
        self.gam_max_diagnostics = None
        self.gam_max_summary = None
        self.gam_max_model = None
        # @}

        self._set_input_file_paths(catchment, months, function, study_area)
        self._set_output_file_paths(months, function, study_area)

    def _set_input_file_paths(self, catchment, months, function, study_area):
        if study_area == "Arolla":
            area = "Arolla_15min"
            self.watershed_path = f'{self.path}Swiss_discharge/Arolla_discharge/Watersheds_on_dhm25/{catchment}_UpslopeArea_EPSG21781.txt'
        elif study_area == "Solda":
            area = "Solda"
            self.watershed_path = f'{self.path}Italy_discharge/Watersheds/{catchment}_UpslopeArea_EPSG25832.txt'
        else:
            self.watershed_path = None
            assert self.watershed_path
        self.subdaily_discharge = f"{self.path}Outputs/ObservedDischarges/{area}_discharge_all_corrected_{catchment}.csv"
        self.observed_daily_discharge = f"{self.path}Outputs/ObservedDischarges/{study_area}_daily_mean_discharge_{catchment}.csv"
        
        self.hydro_units_file = f"{self.results}hydro_units.csv"
        self.forcing_file = f"{self.results}forcing.nc"
        self.results_file = f"{self.results}results.nc"
        self.dataframe_filename = f"{self.results}meteo_df_{function}_{self.months_str}.csv"
        self.dataframe_constrained_filename = f"{self.results}meteo_df_{function}_{self.months_str}_constrained.csv"
        
        self.functions_filename = f"{self.results}pdf_functions_{function}_{self.months_str}.csv"
        self.linear_regr_filename = f"{self.results}linear_regr_{self.months_str}.csv"

        self.simulated_daily_discharge = f"{self.results}best_fit_simulated_discharge_SCEUA_melt:temperature_index_kge_2012.csv"
        self.simulated_daily_discharge_m3s = f"{self.results}best_fit_simulated_discharge_SCEUA_melt:temperature_index_kge_2012_m3s.csv"

    def _set_output_file_paths(self, months, function, study_area):
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
        
        if study_area == "Arolla":
            area = "Arolla_15min"
        else:
            area = study_area

        self.all_bootstrapped_discharge_FDCs = f"{self.results}bootstrapped_FDCs.csv"
        self.observed_15min_discharge_FDCs = f"{self.results}{area}_FDCs_from_observed_15min_discharge.csv"
        self.FDCs_qmean_observed_regr = f"{self.results}{area}_FDCs_from_observed_daily_discharge.csv"
        self.FDCs_qmean_observed_regr_weather = f"{self.results}{area}_FDCs_from_observed_daily_discharge_plus_weather.csv"
        self.FDCs_qmean_observed_multiregr_weather = f"{self.results}{area}_FDCs_from_observed_daily_discharge_plus_weather_plus_multiregr.csv"
        self.FDCs_qmean_simulated_regr = f"{self.results}{area}_FDCs_from_hydrobricks_daily_discharge.csv"
        self.FDCs_qmean_simulated_regr_weather = f"{self.results}{area}_FDCs_from_hydrobricks_daily_discharge_plus_weather.csv"
        self.FDCs_qmean_observed_gam = f"{self.results}{area}_FDCs_from_observed_daily_discharge_GAMs.csv"
        self.FDCs_qmean_observed_gam_weather = f"{self.results}{area}_FDCs_from_observed_daily_discharge_plus_weather_GAMs.csv"
        self.FDCs_qmean_simulated_gam = f"{self.results}{area}_FDCs_from_hydrobricks_daily_discharge_GAMs.csv"
        self.FDCs_qmean_simulated_gam_weather = f"{self.results}{area}_FDCs_from_hydrobricks_daily_discharge_GAMs_plus_weather.csv"
        self.downscaling_metrics = f"{self.results}downscaling_metrics.csv"
        
        self.a_distrib_filename = f"{self.results}a_distrib.csv"
        self.b_distrib_filename = f"{self.results}b_distrib.csv"
        self.c_distrib_filename = f"{self.results}c_distrib.csv"
        self.M_distrib_filename = f"{self.results}M_distrib.csv"
        
        self.gam_min_diagnostics = f"{self.results}gam_min_diagnostics.pdf"
        self.gam_min_summary = f"{self.results}gam_min_summary.txt"
        self.gam_min_model = f"{self.results}gam_min_model.txt"
        self.gam_max_diagnostics = f"{self.results}gam_max_diagnostics.pdf"
        self.gam_max_summary = f"{self.results}gam_max_summary.txt"
        self.gam_max_model = f"{self.results}gam_max_model.txt"
        
        
