import csv

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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
    mse_test = mean_squared_error(q_min_max, regr.predict(q_mean))
    mse_regr = mean_squared_error(q_min_max, regr.predict(q_mean))
    r2_test = r2_score(q_min_max, regr.predict(q_mean))
    r2_regr = r2_score(q_min_max, regr.predict(q_mean))
    print("Mean squared error: %.2f" % mse_test)
    print("Mean squared error: %.2f" % mse_regr)
    print("Coefficient of determination R$^2$: %.2f" % r2_test)
    print("Coefficient of determination R$^2$: %.2f" % r2_regr)

    return regr, score, coefs, intercept, mse_test, mse_regr, r2_test, r2_regr

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
    mse_test1, mse_regr1, r2_test1, r2_regr1 = linear_regression_with_scikit(q_mean, q_min)
    # Linear regression between Qmean and Qmax.
    regr2, score2, coefs2, intercept2, \
    mse_test2, mse_regr2, r2_test2, r2_regr2 = linear_regression_with_scikit(q_mean, q_max)

    ###### MULTIPLE Linear regression
    print(f"Multiple regression with {criteria}... among {meteo_df.columns}.")
    for c in criteria:
        q_mean[c] = meteo_df[c]

    # Multiple linear regression between Qmean and Qmin, with additionnal info.
    regr3, score3, coefs3, intercept3, \
    mse_test3, mse_regr3, r2_test3, r2_regr3 = linear_regression_with_scikit(q_mean, q_min)
    # Multiple linear regression between Qmean and Qmax, with additionnal info.
    regr4, score4, coefs4, intercept4, \
    mse_test4, mse_regr4, r2_test4, r2_regr4 = linear_regression_with_scikit(q_mean, q_max)

    with open(filename, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['variable1', 'variable2', 'score', 'coefs', 'intercept', 'MSE', 'R2', 'MSE', 'R2'])
        csv_out.writerow(["$Q_{mean}$", "$Q_{min}$", score1, coefs1[0], intercept1, mse_regr1, r2_regr1, mse_test1, r2_test1])
        csv_out.writerow(["$Q_{mean}$", "$Q_{max}$", score2, coefs2[0], intercept2, mse_regr2, r2_regr2, mse_test2, r2_test2])
        csv_out.writerow(['variable1', 'variable2', 'score', 'coefs', 'intercept', 'MSE', 'R2', 'MSE', 'R2', 'variable3'])
        csv_out.writerow(["$Q_{mean}$", "$Q_{min}$", score1, coefs1[0], intercept1, mse_regr1, r2_regr1, mse_test1, r2_test1, criteria])
        csv_out.writerow(["$Q_{mean}$", "$Q_{max}$", score2, coefs2[0], intercept2, mse_regr2, r2_regr2, mse_test2, r2_test2, criteria])
    return regr1, regr2, regr3, regr4