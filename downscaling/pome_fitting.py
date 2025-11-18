from test.support import _1M

import numpy as np
from fit_metrics import compute_r2
from scipy.optimize import curve_fit

import warnings

class pome():
    
    def __init__(self):
        
        self.nb_exp_overflow = 0

    def exp_f128(self, array):
        # Avoiding Overflow Error in numpy.exp function with np.float128 (instead of np.float64).
        array_f128 = np.array(array, dtype=np.float128)
        original_filters = warnings.filters.copy()  # Save current warning filters
        warnings.filterwarnings("error", category=RuntimeWarning)  # Temporarily treat RuntimeWarnings as errors
        try:
            exp_in_float128 = np.exp(array_f128)
        except RuntimeWarning as e:
            print(f"Overflow error caught: {e}")
            exp_in_float128 = np.nan
            self.nb_exp_overflow += 1
            print("Number of times an overflow happened (several per days possible):", self.nb_exp_overflow)
        finally:
            warnings.filters = original_filters  # Restore original warning filters
    
        return exp_in_float128

    def func_Singh2014(self, ratio, a, b):
        r"""!
        Equation 18b of Singh et al. (2014).
    
        @param ratio The ratio 't/T', with 't' the cumulative time for which the
            discharge is less than the discharge corresponding to 't' in the
            cumulative distribution function of the discharge over the period
            considered and 'T' the period considered.
        @param a Parameter 'a' of the equation.
        @param b Parameter 'b' of the equation.
    
        @return The cumulative distribution function of discharge: \f$ \(1 - a \cdot x^b\) \f$
        """
        return 1 - a * ratio**b
    
    def discharge_time_equation_Singh2014(self, ratio, a, b, q_min, q_max, M):
        """
        Equation 23 of Singh et al. (2014).
    
        @param ratio The ratio 't/T', with 't' the cumulative time for which
            the discharge is less than the discharge corresponding to 't' in the
            cumulative distribution function of the discharge over the period
            considered and 'T' the period considered.
        @param a Parameter 'a' of the equation.
        @param b Parameter 'b' of the equation.
        @param q_min The minimum daily discharge.
        @param q_max The maximum daily discharge.
        @param M Parameter 'M' of the equation.
    
        @return The flow duration curve.
        """
        numerator = self.exp_f128(-M) - (self.exp_f128(-M) - self.exp_f128(-M * q_min / q_max)) * a * ratio ** b
        right_side = - np.log(numerator) / M
        q = q_max * right_side
        # Reestablishing the np.float64 format as np.float128 is not compatible with SciPy.
        return np.array(q, dtype=np.float64)
    
    def func_Sigmoid_d(self, ratio, a, b, c, d):
        """
        Sigmoid version to replace equation 18b of Singh et al. (2014).
    
        @param ratio The ratio 't/T', with 't' the cumulative time for which
            the discharge is less than the discharge corresponding to 't' in the
            cumulative distribution function of the discharge over the period
            considered and 'T' the period considered.
        @param a Parameter 'a' of the equation.
        @param b Parameter 'b' of the equation.
        @param c Parameter 'c' of the equation.
        @param d Parameter 'd' of the equation.
    
        @return The cumulative distribution function of discharge: \f$ \\(1 - a \\cdot x^b\\) \f$
        """
        exp_calculation = d / (1 + self.exp_f128(a * (ratio - b))) + c * ratio - d + 1
        # Reestablishing the np.float64 format as np.float128 is not compatible with SciPy.
        return np.array(exp_calculation, dtype=np.float64)
    
    def discharge_time_equation_Sigmoid_d(self, ratio, a, b, c, d, q_min, q_max, M):
        """
        Sigmoid version to replace equation 23 of Singh et al. (2014).
    
        @param ratio The ratio 't/T', with 't' the cumulative time for which
            the discharge is less than the discharge corresponding to 't' in the
            cumulative distribution function of the discharge over the period
            considered and 'T' the period considered.
        @param a Parameter 'a' of the equation.
        @param b Parameter 'b' of the equation.
        @param c Parameter 'c' of the equation.
        @param d Parameter 'd' of the equation.
        @param q_min The minimum daily discharge.
        @param q_max The maximum daily discharge.
        @param M Parameter 'M' of the equation.
    
        @return The flow duration curve.
        """
        numerator = self.exp_f128(-M) - (self.exp_f128(-M * q_min / q_max) - self.exp_f128(-M)) * (d / (1 + self.exp_f128(a * (ratio - b))) - d / (1 + self.exp_f128(-a * b)) + c * ratio)
        right_side = - np.log(numerator) / M
        q = q_max * right_side
        # Reestablishing the np.float64 format as np.float128 is not compatible with SciPy.
        return np.array(q, dtype=np.float64)
    
    def func_Sigmoid(self, ratio, a, b, c):
        """
        Sigmoid version to replace equation 18b of Singh et al. (2014), with d = c + 1.
    
        @param ratio The ratio 't/T', with 't' the cumulative time for which
            the discharge is less than the discharge corresponding to 't' in the
            cumulative distribution function of the discharge over the period
            considered and 'T' the period considered.
        @param a Parameter 'a' of the equation.
        @param b Parameter 'b' of the equation.
        @param c Parameter 'c' of the equation.
    
        @return The cumulative distribution function of discharge: \f$ \\(1 - a \\cdot x^b\\) \f$
        """
        exp_calculation = (c + 1) / (1 + self.exp_f128(a * (ratio - b))) + c * ratio - c
        # Reestablishing the np.float64 format as np.float128 is not compatible with SciPy.
        return np.array(exp_calculation, dtype=np.float64)
    
    def discharge_time_equation_Sigmoid(self, ratio, a, b, c, q_min, q_max, M):
        """
        Sigmoid version to replace equation 23 of Singh et al. (2014), with d = c + 1.
    
        @param ratio The ratio 't/T', with 't' the cumulative time for which
            the discharge is less than the discharge corresponding to 't' in the
            cumulative distribution function of the discharge over the period
            considered and 'T' the period considered.
        @param a Parameter 'a' of the equation.
        @param b Parameter 'b' of the equation.
        @param c Parameter 'c' of the equation.
        @param q_min The minimum daily discharge.
        @param q_max The maximum daily discharge.
        @param M Parameter 'M' of the equation.
    
        @return The flow duration curve.
        """
        if np.isnan(a) and np.isnan(b) and np.isnan(c) and np.isnan(M):
            return np.ones(len(ratio)) * np.nan
        elif np.isnan(a) or np.isnan(b) or np.isnan(c) or np.isnan(M):
            print("a, b, c or M is NaN (but not all).")
            return np.ones(len(ratio)) * np.nan
        elif np.isnan(q_min) or np.isnan(q_max):
            print("Qmin or Qmax is NaN.")
            return np.ones(len(ratio)) * np.nan
            
        numerator = self.exp_f128(-M) - (self.exp_f128(-M * q_min / q_max) - self.exp_f128(-M)) * ((c + 1) / (1 + self.exp_f128(a * (ratio - b))) - (c + 1) / (1 + self.exp_f128(-a * b)) + c * ratio)
        right_side = - np.log(numerator) / M
        q = q_max * right_side
        # Reestablishing the np.float64 format as np.float128 is not compatible with SciPy.
        return np.array(q, dtype=np.float64)
    
    def func_Sigmoid_ext_variables(self, input, a1, b1, c1, a2, b2, c2, a3, b3, c3):
        """
        Sigmoid version to replace equation 18b of Singh et al. (2014), with d = c + 1.
        In this version, we define a, b and c as functions of other hydrological or meteorological variables.
        """
        x = input[0]
        var1 = input[1]
        var2 = input[2]
        var3 = input[3]
        a = a1 * var1 + a2 * var2 + a3 * var3
        b = b1 * var1 + b2 * var2 + b3 * var3
        c = c1 * var1 + c2 * var2 + c3 * var3
        exp_calculation = (c + 1) / (1 + self.exp_f128(a * (x - b))) + c * x - c
        # Reestablishing the np.float64 format as np.float128 is not compatible with SciPy.
        return np.array(exp_calculation, dtype=np.float64)
    
    def fit_a_and_b_to_discharge_probability_curve(self, observed_subdaily_discharge,
                                                   day_meteo, function="Singh2014", nb=96):
        """
        First step of the calibration of the downscaling procedure.
    
        @param observed_subdaily_discharge The observed subdaily discharge on
            which we fit the first set of parameters.
        @param day_meteo The daily meteorological dataset. Only used for the
            temporary new version. TO CHECK
        @param function The function to use for the downscaling, between:
            - "Singh2014": the original function
            - "Sigmoid_d": the glacial function
            - "Sigmoid": the simplified glacial function
        @param nb The resolution of the downscaled discharge (number of points
            per day).
    
        @return params, covariance, x_data, y_data, x_fit, y_fit, r2.
        """
    
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
                    params, covariance = curve_fit(self.func_Singh2014, x_data, y_data)
                except RuntimeError as e:
                    print(e)
                    #if e.message == 'RuntimeError: pyparted requires root access':
                    params, covariance = (np.nan, np.nan), np.nan
                a, b = params
                label = f'Fitted Curve: 1 - {a:.2f} * x^{b:.2f}'
        elif function == "Sigmoid_d":
            params, covariance = curve_fit(self.func_Sigmoid_d, y_data, x_data, p0=[15, 0.3, -0.2, 0.8], maxfev=5000)
            a, b, c, d = params
            label = f'Fitted Curve: {d:.2f}/(1 + exp({a:.2f}(x-{b:.2f}))) + {c:.2f}x - {d:.2f} + 1'
        elif function == "Sigmoid":
            if np.isnan(x_data).any() or np.isnan(y_data).any():
                params = covariance = a = b = c = np.nan
                label = ""
            else:
                try:
                    params, covariance = curve_fit(self.func_Sigmoid, y_data, x_data, p0=[15, 0.3, -0.2], maxfev=5000)
                except RuntimeError as e:
                    print(e)
                    #if e.message == 'RuntimeError: pyparted requires root access':
                    params, covariance = (np.nan, np.nan, np.nan), np.nan
                a, b, c = params
                label = f'Fitted Curve: ({c:.2f} + 1)/(1 + exp({a:.2f}(x-{b:.2f}))) + {c:.2f}x - {c:.2f}'
        elif function == "Sigmoid_ext_var":
            var1 = day_meteo['Temperature'].values[0]
            var2 = day_meteo['Glacier snow'].values[0]
            var3 = day_meteo['Radiation'].values[0]
            var1_data = np.ones(nb) * var1
            var2_data = np.ones(nb) * var2
            var3_data = np.ones(nb) * var3
            if np.isnan(x_data).any() or np.isnan(y_data).any():
                params = covariance = a1 = a2 = a3 = b1 = b2 = b3 = c1 = c2 = c3 = np.nan
                label = ""
            else:
                try:
                    arr = np.vstack((y_data, var1_data, var2_data, var3_data))
                    params, covariance = curve_fit(self.func_Sigmoid_ext_variables, arr, x_data,
                                                   p0=[-0.4, -0.4, -0.4, 16, 1, 1, 0.01, 0.01, 0.01]
                                                   , maxfev=5000)
                except RuntimeError as e:
                    print(e)
                    #if e.message == 'RuntimeError: pyparted requires root access':
                    params, covariance = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), np.nan
                a1, a2, a3, b1, b2, b3, c1, c2, c3 = params
                a = a1 * var1 + a2 * var2 + a3 * var3
                b = b1 * var1 + b2 * var2 + b3 * var3
                c = c1 * var1 + c2 * var2 + c3 * var3
                label = f'Fitted Curve: ({c:.2f} + 1)/(1 + exp({a:.2f}(x-{b:.2f}))) + {c:.2f}x - {c:.2f}'
    
        # Plot the resulting graph (Fig. 1 of Singh et al., 2014)
        # Generate points for the fitted curve
        x_fit = np.linspace(min(x_data), max(x_data), nb)
        if function == "Singh2014":
            y_fit = self.func_Singh2014(x_fit, a, b)
            r2 = compute_r2(np.flip(y_data), y_fit)
        elif function == "Sigmoid_d":
            y_fit = self.func_Sigmoid_d(x_fit, a, b, c, d)
            r2 = compute_r2(x_data, y_fit)
        elif function == "Sigmoid":
            y_fit = self.func_Sigmoid(x_fit, a, b, c)
            r2 = compute_r2(x_data, y_fit)
        elif function == "Sigmoid_ext_var":
            arr = np.vstack((x_fit, var1_data, var2_data, var3_data))
            y_fit = self.func_Sigmoid_ext_variables(arr, a1, b1, c1, a2, a3, b2, b3, c2, c3)
            r2 = compute_r2(x_data, y_fit)
    
        # Return the coefficients
        return params, covariance, x_data, y_data, x_fit, y_fit, r2
    
    
    
    def fit_m_to_flow_duration_curve(self, observed_subdaily_discharge, params, q_min, q_max,
                                     day_meteo, function="Singh2014", nb=96):
        """
        Second step of the calibration of the downscaling procedure.
    
        @param observed_subdaily_discharge The observed subdaily discharge on
            which we fit the first set of parameters.
        @param params The ...
        @param q_min The minimum daily discharge.
        @param q_max The maximum daily discharge.
        @param day_meteo The daily meteorological dataset. Only used for the
            temporary new version. TO CHECK
        @param function The function to use for the downscaling, between:
            - "Singh2014": the original function
            - "Sigmoid_d": the glacial function
            - "Sigmoid": the simplified glacial function
        @param nb The resolution of the downscaled discharge (number of points
            per day).
    
        @return M, variance, x_data, y_data, t_fit, y_fit, r2.
        """
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
            elif function == "Sigmoid_ext_var":
                var1 = day_meteo['Glacier snow'].values[0]
                var2 = day_meteo['Radiation'].values[0]*0
                var3 = day_meteo['Radiation'].values[0]*0
                a1, a2, a3, b1, b2, b3, c1, c2, c3 = params
                a = a1 * var1 + a2 * var2 + a3 * var3
                b = b1 * var1 + b2 * var2 + b3 * var3
                c = c1 * var1 + c2 * var2 + c3 * var3
    
        # a, b, q_min, q_max have to be defined outside the function when calling curve_fit,
        # so we define the function to be solved inside this current function.
        def discharge_time_equation_to_solve_Singh2014(tau, M):
            """
            Equation 23 of Singh et al. (2014).
            """
            numerator = self.exp_f128(-M) - (self.exp_f128(-M) - self.exp_f128(-M * q_min / q_max)) * a * tau ** b
            right_side = - np.log(numerator) / M
            q = q_max * right_side
            # Reestablishing the np.float64 format as np.float128 is not compatible with SciPy.
            return np.array(q, dtype=np.float64)
        def discharge_time_equation_to_solve_Sigmoid_d(tau, M):
            """
            Sigmoid version to replace equation 23 of Singh et al. (2014).
            """
            numerator = self.exp_f128(-M) - (self.exp_f128(-M * q_min / q_max) - self.exp_f128(-M)) * (d / (1 + self.exp_f128(a * (tau - b))) - d / (1 + self.exp_f128(-a * b)) + c * tau)
            right_side = - np.log(numerator) / M
            q = q_max * right_side
            # Reestablishing the np.float64 format as np.float128 is not compatible with SciPy.
            return np.array(q, dtype=np.float64)
        def discharge_time_equation_to_solve_Sigmoid(tau, M):
            """
            Sigmoid version to replace equation 23 of Singh et al. (2014), with d = c + 1.
            """
            numerator = self.exp_f128(-M) - (self.exp_f128(-M * q_min / q_max) - self.exp_f128(-M)) * ((c + 1) / (1 + self.exp_f128(a * (tau - b))) - (c + 1) / (1 + self.exp_f128(-a * b)) + c * tau)
            right_side = - np.log(numerator) / M
            q = q_max * right_side
            # Reestablishing the np.float64 format as np.float128 is not compatible with SciPy.
            return np.array(q, dtype=np.float64)
    
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
        elif function == "Sigmoid_ext_var":
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
        elif function == "Sigmoid_ext_var":
            y_fit = discharge_time_equation_to_solve_Sigmoid(t_fit, M, *params2[1:])
    
        r2 = compute_r2(y_data, y_fit)
    
        # Return the coefficient
        return M, variance, x_data, y_data, t_fit, y_fit, r2



