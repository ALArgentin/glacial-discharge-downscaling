import csv
import numpy as np
import pandas as pd
import rpy2.robjects as ro

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def gam_on_discharge(meteo_df, catchment, file_paths):
    
    # Activate pandas to R DataFrame conversion
    pandas2ri.activate()
    
    # Import necessary R packages
    base = importr("base")
    dplyr = importr("dplyr")
    mgcv = importr("mgcv")
    tidyverse = importr("tidyverse")
    
    sel_data = meteo_df.copy()
    
    # Inspect data
    print(sel_data.head(10))
    
    print(sel_data.columns)
    
    # Define the mapping of old column names to new column names
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
    
    # Rename columns in the DataFrame
    sel_data.rename(columns=column_mapping, inplace=True)
    
    # Remove weird values
    sel_data = sel_data[(sel_data["a"] > -50) & (sel_data["a"] < 50)]
    sel_data = sel_data[(sel_data["b"] > -10) & (sel_data["b"] < 5)]
    
     # Add the 'Weather' column using pandas' `np.select`
    conditions = [
        (sel_data["c"] > -0.25) & (sel_data["a"] >= 6),
        (sel_data["c"] > -0.25) & (sel_data["a"] < 6),
        (sel_data["c"] < -0.25) & (sel_data["a"] >= 6),
        (sel_data["c"] < -0.25) & (sel_data["a"] < 6),
    ]
    
    choices = ["1", "2", "3", "4"]
    
    sel_data["Weather"] = np.select(conditions, choices, default=np.nan)
    
    date1 = '2011-07-01'
    date2 = '2014-07-07'
    dates1 = sel_data.index.get_level_values(0)
    print("D1", sel_data["Qmean"][(dates1 >= date1) & (dates1 < date2)])
    print("D1", sel_data["Qmin"][(dates1 >= date1) & (dates1 < date2)])
    print("D1", sel_data["Qmax"][(dates1 >= date1) & (dates1 < date2)])
    
    # Load data into R environment (assume sel_data is already available in Python)
    # We will convert pandas dataframe to R dataframe and work with it in R
    sel_data_r = pandas2ri.py2rpy(sel_data)
    sel_data_r = pandas2ri.py2rpy(sel_data[(dates1 >= date1) & (dates1 < date2)])
        
    # Assign `sel_data_r` to R's environment
    ro.r.assign("sel_data_r", sel_data_r)

    for str in ["min", "max"]:
        
        if str == "min":
            gam_diagnostics = file_paths.gam_min_diagnostics
            gam_summary = file_paths.gam_min_summary
            gam_model = file_paths.gam_min_model
        elif str == "max":
            gam_diagnostics = file_paths.gam_max_diagnostics
            gam_summary = file_paths.gam_max_summary
            gam_model = file_paths.gam_max_model
        
        # Fit the GAM model using mgcv package (adjust this model to your use case)
        ro.r('set.seed(2)')
        ro.r(f"""
        res <- gam(Q{str} ~ s(Qmean, k=10) + s(Tp, k=10) + s(R, k=10) + s(P, k=10) + s(GAP, k=3)
                  + s(IM, k=10) + s(SM, k=10) + s(AS, k=10) + s(GS, k=10), 
                  data=sel_data_r, method="REML", family=tw(link="identity"))
        """)
        
        # Run the gam.check() function on the fitted model
        ro.r(f'pdf("{gam_diagnostics}")')  # Start saving to PDF
        ro.r('gam.check(res)')  # Generate diagnostic plots
        ro.r('dev.off()')  # Close the PDF device to save the file
        
        # Redirect the R console output to a text file
        ro.r(f'''
        sink("{gam_summary}")
        cat("summary(res):\n")
        print(summary(res))
        cat("\n\n")  # Add spacing
        cat("gam.check(res):\n")
        gam.check(res)
        sink()  # Stop redirection
        ''')
        
        # Save the fitted model to a file
        ro.r(f'saveRDS(res, file="{gam_model}")')
 

def linear_regression_with_scikit(q_mean, q_min_max):
    """
    Performs linear regression using scikit-learn and evaluates the model.

    This function fits a linear regression model to the provided data, computes
    performance metrics, and outputs key information such as the model's score,
    coefficients, and intercept.

    @param q_mean (numpy.ndarray or pandas.DataFrame)
        Input data for the mean discharge (predictors).
    @param q_min_max (numpy.ndarray or pandas.DataFrame)
        Input data for the minimum or maximum discharge (response).

    @return tuple
        A tuple containing the following:
        - regr : sklearn.linear_model.LinearRegression
            The fitted linear regression model.
        - score : float
            The coefficient of determination R² of the fitted model.
        - coefs : numpy.ndarray
            Coefficients of the linear regression model.
        - intercept : float
            Intercept of the linear regression model.
        - mse : float
            Mean squared error of the fitted regression model.
        - r2 : float
            Coefficient of determination R² for the fitted regression model.

    Notes:
    -----
    - Ensure that `q_mean` and `q_min_max` are appropriately preprocessed and
      compatible with scikit-learn's `LinearRegression` model.
    """

    regr = LinearRegression().fit(q_mean, q_min_max)
    score = regr.score(q_mean, q_min_max)
    coefs = regr.coef_
    intercept = regr.intercept_
    print(f"Linear regression with score = {score}, coefficients = {coefs} and intercept = {intercept}.")

    # Make predictions using the testing set and compute the fit
    mse = mean_squared_error(q_min_max, regr.predict(q_mean))
    r2 = r2_score(q_min_max, regr.predict(q_mean))
    print("Mean squared error: %.2f" % mse)
    print("Coefficient of determination R$^2$: %.2f" % r2)

    return regr, score, coefs, intercept, mse, r2

def extract_discharge_relation_to_daily_mean(meteo_df, filename, criteria):
    """
    Extracts discharge relationships to daily mean values and performs regression analysis.

    This function computes linear and multiple linear regression models to
    establish relationships between the mean daily discharge and its minimum
    and maximum values. Results are saved to a specified CSV file.

    @param meteo_df (pandas.DataFrame)
        DataFrame containing meteorological data with columns for daily
        discharge values:
        - `$Q_{mean}$`: Mean daily discharge.
        - `$Q_{min}$`: Minimum daily discharge.
        - `$Q_{max}$`: Maximum daily discharge.
    @param filename (str)
        Path to the output CSV file where regression results are written.
    @param criteria (list of str)
        List of additional features (column names in `meteo_df`) to include in
        the multiple regression analysis.

    @return tuple
        A tuple containing the following regression models:
        - regr1 : sklearn.linear_model.LinearRegression
            Linear regression model for `$Q_{mean}$` vs `$Q_{min}$`.
        - regr2 : sklearn.linear_model.LinearRegression
            Linear regression model for `$Q_{mean}$` vs `$Q_{max}$`.
        - regr3 : sklearn.linear_model.LinearRegression
            Multiple linear regression model for `$Q_{mean}$` vs `$Q_{min}$` with additional criteria.
        - regr4 : sklearn.linear_model.LinearRegression
            Multiple linear regression model for `$Q_{mean}$` vs `$Q_{max}$` with additional criteria.

    Notes:
    -----
    - The function drops rows with missing values in `meteo_df` before performing regressions.
    - For each regression, the model's score, coefficients, intercept, mean
      squared error (MSE), and R² values are computed.
    - Results are saved in the specified CSV file in two formats:
      - Without additional features (linear regression).
      - With additional features (multiple regression).
    """

    meteo_df = meteo_df.dropna()
    q_min = meteo_df['$Q_{min}$']
    q_mean = meteo_df['$Q_{mean}$'].to_frame()
    q_max = meteo_df['$Q_{max}$']

    ###### Linear regression
    # Linear regression between Qmean and Qmin.
    regr1, score1, coefs1, intercept1, \
    mse1, r21 = linear_regression_with_scikit(q_mean, q_min)
    # Linear regression between Qmean and Qmax.
    regr2, score2, coefs2, intercept2, \
    mse2, r22 = linear_regression_with_scikit(q_mean, q_max)

    ###### MULTIPLE Linear regression
    print(f"Multiple regression with {criteria}... among {meteo_df.columns}.")
    for c in criteria:
        q_mean[c] = meteo_df[c]

    # Multiple linear regression between Qmean and Qmin, with additionnal info.
    regr3, score3, coefs3, intercept3, \
    mse3, r23 = linear_regression_with_scikit(q_mean, q_min)
    # Multiple linear regression between Qmean and Qmax, with additionnal info.
    regr4, score4, coefs4, intercept4, \
    mse4, r24 = linear_regression_with_scikit(q_mean, q_max)

    with open(filename, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['variable1', 'variable2', 'score', 'coefs', 'intercept', 'MSE', 'R2', 'variable3'])
        csv_out.writerow(["$Q_{mean}$", "$Q_{min}$", score1, coefs1[0], intercept1, mse1, r21, 'None'])
        csv_out.writerow(["$Q_{mean}$", "$Q_{max}$", score2, coefs2[0], intercept2, mse2, r22, 'None'])
        csv_out.writerow(['variable1', 'variable2', 'score', 'coefs', 'intercept', 'MSE', 'R2', 'MSE', 'R2', 'variable3'])
        csv_out.writerow(["$Q_{mean}$", "$Q_{min}$", score3, coefs3[0], intercept3, mse3, r23, criteria])
        csv_out.writerow(["$Q_{mean}$", "$Q_{max}$", score4, coefs4[0], intercept4, mse4, r24, criteria])
    return regr1, regr2, regr3, regr4