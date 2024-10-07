def linear_regression_with_scikit(q_mean, q_min_max):

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