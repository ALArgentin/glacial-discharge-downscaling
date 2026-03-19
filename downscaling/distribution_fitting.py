# Standard library imports
import csv

import numpy as np
import scipy.stats as stats


def test_different_functions(meteo_df, variable):
    """
    Tests multiple probability distribution functions and selects the best fit 
    for a given variable based on the Kolmogorov-Smirnov (KS) statistic.

    This function fits four candidate distributions (normal, lognormal, gamma, 
    and beta) to the input data and evaluates their goodness-of-fit using the 
    KS test. The distribution with the lowest KS statistic is selected as the 
    best-fitting distribution.

    @param meteo_df (pandas.DataFrame)
        DataFrame containing the variable of interest.
    @param variable (str)
        Name of the column in `meteo_df` to be analyzed.

    @return (tuple)
        A tuple containing:
        - `type` (str): Name of the best-fitting distribution 
          ('normal', 'lognormal', 'gamma', or 'beta').
        - `params` (tuple): Parameters of the selected distribution.
        - `statis` (float): KS statistic associated with the best fit.

    Notes:
    -----
    - Missing values are removed prior to fitting.
    - The KS statistic measures the distance between empirical and theoretical 
      distributions; lower values indicate better fit.
    - The function prints the selected distribution and its KS statistic.
    """

    data = meteo_df[variable].dropna().values

    # Fit Distributions
    # Lognormal distribution
    sigma, loc1, scale1 = stats.lognorm.fit(data)
    # Gamma distribution
    shape, loc2, scale2 = stats.gamma.fit(data)
    # Beta distribution
    a, b, loc3, scale3 = stats.beta.fit(data)
    # Normal distribution
    loc4, scale4 = stats.norm.fit(data)

    # Evaluate Distributions
    lognormal_test = stats.kstest(data, stats.lognorm.cdf, args=(sigma, loc1, scale1))
    gamma_test = stats.kstest(data, stats.gamma.cdf, args=(shape, loc2, scale2))
    beta_test = stats.kstest(data, stats.beta.cdf, args=(a, b, loc3, scale3))
    norm_test = stats.kstest(data, stats.norm.cdf, args=(loc4, scale4))

    # Select Best Distribution
    if norm_test.statistic < lognormal_test.statistic and norm_test.statistic < gamma_test.statistic and norm_test.statistic < beta_test.statistic:
        type = 'normal'
        params = (loc4, scale4)
        statis = norm_test.statistic
    elif lognormal_test.statistic < gamma_test.statistic and lognormal_test.statistic < beta_test.statistic:
        type = 'lognormal'
        params = (sigma, loc1, scale1)
        statis = lognormal_test.statistic
    elif gamma_test.statistic < lognormal_test.statistic and gamma_test.statistic < beta_test.statistic:
        type = 'gamma'
        params = (shape, loc2, scale2)
        statis = gamma_test.statistic
    else:
        type = 'beta'
        params = (a, b, loc3, scale3)
        statis = beta_test.statistic
    print(f"{type} distribution is the best fit for {variable} with {statis:.2f}.")

    return type, params, statis


def find_and_save_best_pdf_functions(meteo_df, filename, function):
    """
    This function applies `test_different_functions` to a predefined set of 
    variables (e.g., parameters of a sigmoid model and discharge values), 
    depending on the selected function type. It identifies the best-fitting
    probability distributions and saves the results to a CSV file.

    @param meteo_df (pandas.DataFrame)
        DataFrame containing the variables to analyze.
    @param filename (str)
        Path to the output CSV file where results will be saved.
    @param function (str)
        Type of function used in the model (e.g., "Sigmoid", "Sigmoid_d"). 
        Determines which variables are included in the analysis.

    @return None

    Notes:
    -----
    - Variables analyzed include "$a$", "$b$", "$M$", "$Q_{min}$", "$Q_{max}$",
      and optionally "$c$" and "$d$" depending on `function`.
    - Output CSV contains columns: variable, distribution type, parameters, and statistic.
    """

    a_type, a_params, a_stats = test_different_functions(meteo_df, "$a$")
    b_type, b_params, b_stats = test_different_functions(meteo_df, "$b$")
    if function == "Sigmoid_d" or function == "Sigmoid":
        c_type, c_params, c_stats = test_different_functions(meteo_df, "$c$")
    if function == "Sigmoid_d":
        d_type, d_params, d_stats = test_different_functions(meteo_df, "$d$")
    M_type, M_params, M_stats = test_different_functions(meteo_df, "$M$")
    Qmin_type, Qmin_params, Qmin_stats = test_different_functions(meteo_df, "$Q_{min}$")
    Qmax_type, Qmax_params, Qmax_stats = test_different_functions(meteo_df, "$Q_{max}$")

    with open(filename, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['variable', 'type', 'params', 'stats'])
        csv_out.writerow(["$a$", a_type, a_params, a_stats])
        csv_out.writerow(["$b$", b_type, b_params, b_stats])
        if function == "Sigmoid_d" or function == "Sigmoid":
            csv_out.writerow(["$c$", c_type, c_params, c_stats])
        if function == "Sigmoid_d":
            csv_out.writerow(["$d$", d_type, d_params, d_stats])
        csv_out.writerow(["$M$", M_type, M_params, M_stats])
        csv_out.writerow(["$Q_{min}$", Qmin_type, Qmin_params, Qmin_stats])
        csv_out.writerow(["$Q_{max}$", Qmax_type, Qmax_params, Qmax_stats])


def get_KDE_model(meteo_df, variable, weather_list=None):
    """
    Computes Kernel Density Estimation (KDE) models for a given variable.

    This function estimates the probability density function of a variable using 
    Gaussian KDE. KDEs can be computed globally or separately for different 
    weather states.

    @param meteo_df (pandas.DataFrame)
        DataFrame containing the variable and optionally weather classifications.
    @param variable (str)
        Name of the variable to model.
    @param weather_list (list of str, optional)
        List of weather states used to condition the KDE estimation. 
        If provided, a KDE is computed for each weather state.

    @return (tuple)
        - If `weather_list` is provided:
            (list of scipy.stats.gaussian_kde, list of str)
        - Otherwise:
            (scipy.stats.gaussian_kde, None)

    Notes:
    -----
    - Missing values are removed before KDE computation.
    - KDEs are computed using `scipy.stats.gaussian_kde`.
    """

    if weather_list:
        kdes = []
        for state in weather_list:
            state_data = meteo_df.loc[(meteo_df["Weather"] == state), variable].dropna().values

            # KDE modelisation
            kde = stats.gaussian_kde(state_data)
            kdes.append(kde)
        return kdes, weather_list

    else:
        data = meteo_df[variable].dropna().values

        # KDE modelisation
        kde = stats.gaussian_kde(data)
        return kde, None


def add_to_kde_dict(kde_dict, kde, var, weather_list):
    """
    Adds KDE models to a dictionary with appropriate keys, optionally separating them 
    by weather state.

    @param kde_dict (dict)
        Dictionary to store KDE models.
    @param kde (scipy.stats.gaussian_kde or list)
        KDE model(s) to add.
    @param var (str)
        Name of the variable associated with the KDE.
    @param weather_list (list of str or None)
        Weather states corresponding to the KDE models. If None, a single KDE is stored.

    @return None

    Notes:
    -----
    - Keys are formatted as "variable_state" when weather-dependent.
    - Otherwise, the key is simply the variable name.
    """
    if weather_list:
        for state, k in zip(weather_list, kde):
            key = var + "_" + state
            kde_dict[key] = k
    else:
        kde_dict[var] = kde


def KDE_computations(meteo_df, function, weather_list):
    """
    Computes KDE models for multiple variables, optionally conditioned on 
    weather states, and stores them in a dictionary.

    @param meteo_df (pandas.DataFrame)
        DataFrame containing the variables to model.
    @param function (str)
        Type of function used (e.g., "Sigmoid", "Sigmoid_d"), determining which 
        variables are included.
    @param weather_list (list of str)
        List of weather states for conditional KDE estimation.

    @return (dict)
        Dictionary containing KDE models, indexed by variable names or 
        variable-weather combinations.

    Notes:
    -----
    - Always includes "$a$", "$b$", "$M$", "$Q_{min}$", "$Q_{max}$".
    - Includes "$c$" and "$d$" depending on the function type.
    """

    kde_dict = {}

    for var in ["$a$", "$b$", "$M$", "$Q_{min}$", "$Q_{max}$"]:
        kde, weather_list = get_KDE_model(meteo_df, var, weather_list)
        add_to_kde_dict(kde_dict, kde, var, weather_list)

    if function == "Sigmoid_d" or function == "Sigmoid":
        kde, weather_list = get_KDE_model(meteo_df, "$c$", weather_list)
        add_to_kde_dict(kde_dict, kde, "$c$", weather_list)
    if function == "Sigmoid_d":
        kde, weather_list = get_KDE_model(meteo_df, "$d$", weather_list)
        add_to_kde_dict(kde_dict, kde, "$d$", weather_list)

    return kde_dict


def sample_weather_distribs_dependently(meteo_df, state_of_days):
    """
    Samples parameter values from observed data conditioned on weather states.

    This function selects random observations from the dataset for each weather 
    state, preserving dependencies between variables.

    @param meteo_df (pandas.DataFrame)
        DataFrame containing variables and a "Weather" column.
    @param state_of_days (list of str)
        Sequence of weather states for which to sample values.

    @return (tuple of lists)
        - `a_distrib`: Sampled values of "$a$"
        - `b_distrib`: Sampled values of "$b$"
        - `c_distrib`: Sampled values of "$c$"
        - `M_distrib`: Sampled values of "$M$"

    Notes:
    -----
    - Sampling is done with a fixed random seed for reproducibility.
    - If a state is 'None', NaN values are returned.
    - Dependencies between variables are preserved since values are sampled from the same row.
    """
    a_distrib = []
    b_distrib = []
    c_distrib = []
    M_distrib = []
    for state in state_of_days:
        if state != 'None':
            sampled_row = meteo_df[(meteo_df['Weather'] == state) & meteo_df['$a$'].notna()].sample(1, random_state=42)
            a = sampled_row["$a$"].values
            b = sampled_row["$b$"].values
            c = sampled_row["$c$"].values
            M = sampled_row["$M$"].values
            assert not np.isnan(a)
        else:
            a, b, c, M =  np.nan, np.nan, np.nan, np.nan
        a_distrib.append(a)
        b_distrib.append(b)
        c_distrib.append(c)
        M_distrib.append(M)
    return a_distrib, b_distrib, c_distrib, M_distrib


def sample_weather_kde_independently(var, kde_dict, state_of_days):
    """
    Samples parameter values from KDE distributions for each weather state, 
    treating variables independently.

    @param var (str)
        Variable name to sample.
    @param kde_dict (dict)
        Dictionary containing KDE models indexed by variable and weather state.
    @param state_of_days (list of str)
        Sequence of weather states for which to generate samples.

    @return (list)
        List of sampled values corresponding to each weather state.

    Notes:
    -----
    - Sampling is performed using `gaussian_kde.resample`.
    - A fixed seed ensures reproducibility.
    - If a state is 'None', NaN values are returned.
    - Unlike dependent sampling, variables are sampled independently.
    """
    distrib = []
    for state in state_of_days:
        if state != 'None':
            key = var + '_' + state
            value = kde_dict[key].resample(1, seed=42)[0][0]
            assert not np.isnan(value)
        else:
            value = np.nan
        distrib.append(value)
    return distrib


