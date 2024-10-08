import numpy as np
from fit_metrics import compute_r2
from scipy.optimize import curve_fit


def func_Singh2014(x, a, b):
    r"""
    Equation 18b of Singh et al. (2014).

    @param x The input variable for the equation.
    @param a Parameter 'a' of the equation.
    @param b Parameter 'b' of the equation.
    @return The result of the equation: \(1 - a \cdot x^b\)
    """
    return 1 - a * x**b

def discharge_time_equation_Singh2014(tau, a, b, q_min, q_max, M):
    """
    Equation 23 of Singh et al. (2014).
    """
    # Compute the discharge time points from the observed discharges
    numerator = np.exp(-M) - (np.exp(-M) - np.exp(-M * q_min / q_max)) * a * tau ** b
    right_side = - np.log(numerator) / M
    q = q_max * right_side

    return q

def func_Sigmoid_d(x, a, b, c, d):
    """
    Sigmoid version to replace equation 18b of Singh et al. (2014).
    """
    return d / (1 + np.exp(a * (x - b))) + c * x - d + 1

def discharge_time_equation_Sigmoid_d(tau, a, b, c, d, q_min, q_max, M):
    """
    Sigmoid version to replace equation 23 of Singh et al. (2014).
    """
    # Compute the discharge time points from the observed discharges
    numerator = np.exp(-M) - (np.exp(-M * q_min / q_max) - np.exp(-M)) * (d / (1 + np.exp(a * (tau - b))) - d / (1 + np.exp(-a * b)) + c * tau)
    right_side = - np.log(numerator) / M
    q = q_max * right_side

    return q

def func_Sigmoid(x, a, b, c):
    """
    Sigmoid version to replace equation 18b of Singh et al. (2014), with d = c + 1.
    """
    return (c + 1) / (1 + np.exp(a * (x - b))) + c * x - c

def discharge_time_equation_Sigmoid(tau, a, b, c, q_min, q_max, M):
    """
    Sigmoid version to replace equation 23 of Singh et al. (2014), with d = c + 1.
    """
    # Compute the discharge time points from the observed discharges
    numerator = np.exp(-M) - (np.exp(-M * q_min / q_max) - np.exp(-M)) * ((c + 1) / (1 + np.exp(a * (tau - b))) - (c + 1) / (1 + np.exp(-a * b)) + c * tau)
    right_side = - np.log(numerator) / M
    q = q_max * right_side

    return q

def fit_a_and_b_to_discharge_probability_curve(observed_subdaily_discharge, function="Singh2014", nb=96):

    # Order observation points from higher to lowest
    # Compute tau
    df = observed_subdaily_discharge.sort_values(by='Discharge', ascending=False)
    X2 = df['Discharge'].values
    # Normalize all values to be between 0 and 1
    x_data = (X2 - np.min(X2))/(np.max(X2) - np.min(X2))
    y_data = np.array(range(len(X2)))/float(len(X2))

    # Use the least-square method to fit the a and b coefficients to the observation points
    # Use curve_fit to find the optimal values for a and b
    if function == "Singh2014":
        # F = 1 - a * tau**b
        if np.isnan(x_data).any() or np.isnan(y_data).any():
            params = covariance = a = b = np.nan
            label = ""
        else:
            try:
                params, covariance = curve_fit(func_Singh2014, x_data, y_data)
            except RuntimeError as e:
                print(e)
                #if e.message == 'RuntimeError: pyparted requires root access':
                params, covariance = (np.nan, np.nan), np.nan
            a, b = params
            label = f'Fitted Curve: 1 - {a:.2f} * x^{b:.2f}'
    elif function == "Sigmoid_d":
        params, covariance = curve_fit(func_Sigmoid_d, y_data, x_data, p0=[15, 0.3, -0.2, 0.8], maxfev=5000)
        a, b, c, d = params
        label = f'Fitted Curve: {d:.2f}/(1 + exp({a:.2f}(x-{b:.2f}))) + {c:.2f}x - {d:.2f} + 1'
    elif function == "Sigmoid":
        if np.isnan(x_data).any() or np.isnan(y_data).any():
            params = covariance = a = b = c = np.nan
            label = ""
        else:
            try:
                params, covariance = curve_fit(func_Sigmoid, y_data, x_data, p0=[15, 0.3, -0.2], maxfev=5000)
            except RuntimeError as e:
                print(e)
                #if e.message == 'RuntimeError: pyparted requires root access':
                params, covariance = (np.nan, np.nan, np.nan), np.nan
            a, b, c = params
            label = f'Fitted Curve: ({c:.2f} + 1)/(1 + exp({a:.2f}(x-{b:.2f}))) + {c:.2f}x - {c:.2f}'

    # Plot the resulting graph (Fig. 1 of Singh et al., 2014)
    # Generate points for the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), nb)
    if function == "Singh2014":
        y_fit = func_Singh2014(x_fit, a, b)
        r2 = compute_r2(np.flip(y_data), y_fit)
    elif function == "Sigmoid_d":
        y_fit = func_Sigmoid_d(x_fit, a, b, c, d)
        r2 = compute_r2(x_data, y_fit)
    elif function == "Sigmoid":
        y_fit = func_Sigmoid(x_fit, a, b, c)
        r2 = compute_r2(x_data, y_fit)


    # Return the coefficients
    return params, covariance, x_data, y_data, x_fit, y_fit, r2



def fit_m_to_flow_duration_curve(observed_subdaily_discharge, params, q_min, q_max,
                                 function="Singh2014", nb=96):
    # Extract the values of the params
    if np.isnan(params).any():
        a = b = c = d = np.nan
    else:
        if function == "Singh2014":
            a, b = params
        elif function == "Sigmoid_d":
            a, b, c, d = params
        elif function == "Sigmoid":
            a, b, c = params

    # a, b, q_min, q_max have to be defined outside the function when calling curve_fit,
    # so we define the function to be solved inside this current function.
    def discharge_time_equation_to_solve_Singh2014(tau, M):
        """
        Equation 23 of Singh et al. (2014).
        """
        # Compute the discharge time points from the observed discharges
        numerator = np.exp(-M) - (np.exp(-M) - np.exp(-M * q_min / q_max)) * a * tau ** b
        right_side = - np.log(numerator) / M
        q = q_max * right_side

        return q
    def discharge_time_equation_to_solve_Sigmoid_d(tau, M):
        """
        Sigmoid version to replace equation 23 of Singh et al. (2014).
        """
        # Compute the discharge time points from the observed discharges
        numerator = np.exp(-M) - (np.exp(-M * q_min / q_max) - np.exp(-M)) * (d / (1 + np.exp(a * (tau - b))) - d / (1 + np.exp(-a * b)) + c * tau)
        right_side = - np.log(numerator) / M
        q = q_max * right_side

        return q
    def discharge_time_equation_to_solve_Sigmoid(tau, M):
        """
        Sigmoid version to replace equation 23 of Singh et al. (2014), with d = c + 1.
        """
        # Compute the discharge time points from the observed discharges
        numerator = np.exp(-M) - (np.exp(-M * q_min / q_max) - np.exp(-M)) * ((c + 1) / (1 + np.exp(a * (tau - b))) - (c + 1) / (1 + np.exp(-a * b)) + c * tau)
        right_side = - np.log(numerator) / M
        q = q_max * right_side

        return q

    # Order observation points from higher to lowest
    df = observed_subdaily_discharge.sort_values(by='Discharge', ascending=False)
    df = df.reset_index()

    # Use the least-square method to fit the M coefficient to the observation points
    # Compute tau
    T = len(df) - 1
    tau = df.index.values / T
    x_data = tau
    y_data = df['Discharge'].values #/ np.nanmax(df['BI'].values)

    initial_guess = [1.0]  # Initial guess for M

    # Use curve_fit to fit the parameters
    if function == "Singh2014":
        if not np.isnan(x_data).any() and not np.isnan(y_data).any():
            try:
                params2, covariance = curve_fit(discharge_time_equation_to_solve_Singh2014,
                                               x_data, y_data, p0=initial_guess, maxfev=5000)
            except RuntimeError as e:
                print(e)
                params2, covariance = [np.nan], np.nan
        else:
            return np.nan, np.nan, x_data, y_data, [], [], np.nan
    elif function == "Sigmoid_d":
        try:
            params2, covariance = curve_fit(discharge_time_equation_to_solve_Sigmoid_d,
                                           x_data, y_data, p0=initial_guess, maxfev=5000)
        except RuntimeError as e:
            print(e)
            params2, covariance = [np.nan], np.nan
    elif function == "Sigmoid":
        if not np.isnan(x_data).any() and not np.isnan(y_data).any():
            try:
                params2, covariance = curve_fit(discharge_time_equation_to_solve_Sigmoid,
                                                x_data, y_data, p0=initial_guess, maxfev=5000)
            except RuntimeError as e:
                print(e)
                params2, covariance = [np.nan], np.nan

        else:
            return np.nan, np.nan, x_data, y_data, [], [], np.nan

    # Extract the optimized parameter(s)
    print(params2)
    M = params2[0]
    if not np.isnan(covariance):
        variance = covariance[0][0]
    else:
        variance = np.nan

    # Generate fitted curve for plotting
    t_fit = np.linspace(min(x_data), max(x_data), nb)
    if function == "Singh2014":
        y_fit = discharge_time_equation_to_solve_Singh2014(t_fit, M, *params2[1:])  # Use *params[1:] to unpack the remaining parameters
    elif function == "Sigmoid_d":
        y_fit = discharge_time_equation_to_solve_Sigmoid_d(t_fit, M, *params2[1:])
    elif function == "Sigmoid":
        y_fit = discharge_time_equation_to_solve_Sigmoid(t_fit, M, *params2[1:])

    r2 = compute_r2(y_data, y_fit)

    # Return the coefficient
    return M, variance, x_data, y_data, t_fit, y_fit, r2



