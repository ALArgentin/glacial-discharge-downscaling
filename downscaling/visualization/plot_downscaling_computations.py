from datetime import datetime

import hydrobricks as hb
import matplotlib.colors as mcolors
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from file_paths import FilePaths
from dask.array.chunk import linspace
from matplotlib import cm, ticker
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FormatStrFormatter
from pip._vendor.pygments.unistring import Pe
from scipy.optimize import curve_fit
from scipy.stats import entropy
from sklearn.metrics import r2_score
from scipy.constants._constants import pt


def select_days(df, days):
    df = df[df.index.day.isin(days)]
    return df
def select_years(df, years):
    df = df[df.index.year.isin(years)]
    return df

def select_months(df, months):
    df = df[df.index.month.isin(months)]
    return df

def find_minima_and_maxima(filename, start='2010-08-01', end='2010-08-15'):

    # Open the discharge dataset and get all data points from the requested date
    df = pd.read_csv(filename, header=0, na_values='', usecols=[0, 1], #index_col=0,
                     parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    df.rename(columns={df.columns[1]: 'Discharge'}, inplace=True)
    # Change the index format
    #df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    end_date += pd.Timedelta(hours=23)
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Step 1: Identify Minimum Daily Discharge
    min_daily_discharge = df.groupby(df['Date'].dt.date)['Discharge'].idxmin()
    min_daily_discharge_data = df.loc[min_daily_discharge, ['Date', 'Discharge']]
    min_daily_discharge_data = min_daily_discharge_data.reset_index(drop=True)

    # Step 2: Identify Maximum Discharge between Consecutive Minimums
    max_discharge_between_min = []
    all_subsets = []

    # Set up the color map (viridis in this example)
    cmap = cm.get_cmap('viridis')
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for i in range(len(min_daily_discharge) - 1):
        start_date = df.loc[min_daily_discharge[i], 'Date']
        end_date = df.loc[min_daily_discharge[i + 1], 'Date']

        # Filter data between consecutive minimums
        subset = df[(df['Date'] > start_date) & (df['Date'] <= end_date)]

        # Find maximum discharge and its occurrence time
        max_discharge_idx = subset['Discharge'].idxmax()
        max_discharge_time = df.loc[max_discharge_idx, 'Date']
        max_discharge_value = df.loc[max_discharge_idx, 'Discharge']

        max_discharge_between_min.append({'Date': max_discharge_time,
                                          'Discharge': max_discharge_value})

        # Filter data between consecutive minimums
        subset = df[(df['Date'] >= max_discharge_time) & (df['Date'] <= end_date)]
        # Append subset of data between maximum and minimum discharge
        all_subsets.append(subset)

        day_df_1 = subset.sort_values(by='Discharge', ascending=False)
        day_df_1 = day_df_1.reset_index()

        # Compute tau
        T = len(day_df_1) - 1
        tau = day_df_1.index.values / T
        x_data_1 = tau
        day_df_1['Discharge'] = day_df_1['Discharge'] / np.nanmax(day_df_1['Discharge'].values)

        d = pd.to_datetime(subset['Date'].values, format='%Y-%m-%d') - pd.Timedelta(days=i)

        color = cmap(i / (len(min_daily_discharge) - 1))  # Scale the color based on day index
        ax1.scatter(x_data_1, day_df_1['Discharge'].values, label=f'Day {i+1}', color=color, alpha=0.7)
        ax2.scatter(d, subset['Discharge'].values, label=f'Day {i+1}', color=color, alpha=0.7)

    ax1.set_xlabel('tau')
    ax1.set_ylabel('F(Q)')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    ax1.set_title('Flow duration curve of the recession flows')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Discharge')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    ax2.set_title('Hydrograph of the recession flows')
    ax2.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(dates.HourLocator(interval=4))

    max_discharge_between_min_df = pd.DataFrame(max_discharge_between_min)

    # Concatenate all subsets into a single DataFrame
    recession_flows = pd.concat(all_subsets, ignore_index=True)

    print("Minimum Daily Discharge:")
    print(min_daily_discharge_data)
    print("Maximum Daily Discharge:")
    print(max_discharge_between_min_df)
    print("All recession flows:")
    print(recession_flows)

    return min_daily_discharge_data, max_discharge_between_min_df, recession_flows

##############################################################################################################

def plot_subdaily_discharge(filename, start='2010-08-01', end='2010-08-15'):

    # Open the discharge dataset and get all data points from the requested date
    df = pd.read_csv(filename, header=0, na_values='', usecols=[0, 1], index_col=0,
                     parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    df.rename(columns={df.columns[0]: 'Discharge'}, inplace=True)
    # Change the index format
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

    # Example DataFrame with datetime index
    date_rng = pd.date_range(start=start, end=end)

    # Set up the color map (viridis in this example)
    cmap = cm.get_cmap('viridis')
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # Loop through all the days in the month
    for i, day in enumerate(date_rng.strftime('%Y-%m-%d')):
        print(f"Processing day {day}")

        # Select the day's data
        observed_subdaily_discharge = df[df.index.strftime('%Y-%m-%d') == day]

        # Order observation points from higher to lowest
        day_df_1 = observed_subdaily_discharge.sort_values(by='Discharge', ascending=False)
        day_df_1 = day_df_1.reset_index()

        # Compute tau
        T = len(day_df_1) - 1
        tau = day_df_1.index.values / T
        x_data_1 = tau
        day_df_1['Discharge'] = day_df_1['Discharge'] / np.nanmax(day_df_1['Discharge'].values)

        day_df_2 = observed_subdaily_discharge.groupby(pd.Grouper(freq='H')).mean()
        day_df_2 = day_df_2.sort_values(by='Discharge', ascending=False)
        day_df_2 = day_df_2.reset_index()

        T = len(day_df_2) - 1
        tau = day_df_2.index.values / T
        x_data_2 = tau
        day_df_2['Discharge'] = day_df_2['Discharge'] / np.nanmax(day_df_2['Discharge'].values)

        color = cmap(i / (len(date_rng) - 1))  # Scale the color based on day index
        ax1.scatter(x_data_1, day_df_1['Discharge'].values, label=f'Day {i+1}', color=color, alpha=0.7)
        ax2.scatter(x_data_2, day_df_2['Discharge'].values, label=f'Day {i+1}', color=color, alpha=0.7)

    ax1.set_xlabel('tau')
    ax1.set_ylabel('F(Q)')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    ax1.set_title('Subdaily observations for the month of August 2010')

    ax2.set_xlabel('tau')
    ax2.set_ylabel('F(Q)')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    ax2.set_title('Hourly observations for the month of August 2010')

def inside_hydrograph_plot(df, axes):
    # Select the dates
    date1 = '2010-06-01'
    date2 = '2010-06-06'

    start_date = pd.to_datetime(date1)
    end_date = pd.to_datetime(date2)
    date_range = pd.date_range(start_date, end_date, freq='D')

    for i, day in enumerate(date_range.strftime('%Y-%m-%d')):
        print(f"Processing day {day}")
        # Select the day's data
        observed_subdaily_discharge = df[df.index == day]

        axes[i].scatter(pd.to_datetime(observed_subdaily_discharge.iloc[:,0].values),
                        observed_subdaily_discharge.iloc[:,1] / np.nanmax(observed_subdaily_discharge.iloc[:,1]), label='Measured', s=2)
        axes[i].text(.80, .1, day, fontsize=8, horizontalalignment='center',
                     verticalalignment='center', transform=axes[i].transAxes)
        axes[i].set_xticklabels([])

def retrieve_subdaily_discharge(filename, figures=''):

    df = pd.read_csv(filename, header=0, na_values='', usecols=[0, 1],
                     parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    df.rename(columns={df.columns[1]: 'Discharge'}, inplace=True)
    # Change the index format
    df.index = pd.to_datetime(df.iloc[:,0].values, format='%Y-%m-%d').strftime('%Y-%m-%d')

    fig, axes = plt.subplots(6, 1, figsize=(2,8))
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.3)

    inside_hydrograph_plot(df, axes)
       # observed_subdaily_discharge = observed_subdaily_discharge.drop(observed_subdaily_discharge.columns[0], axis=1)

    axes[0].set_title("Hydrograph")
    axes[2].set_ylabel("Discharge")
    axes[-1].set_xlabel("Time t")
    # Format the date into hours
    axes[-1].xaxis.set_major_formatter(dates.DateFormatter('%H h'))
    # Change the tick interval
    axes[-1].xaxis.set_major_locator(dates.HourLocator(byhour=[0, 12, 24]))
    fig.savefig(f"{figures}hydrographs.pdf", format="pdf", bbox_inches='tight', dpi=100)

############################################################################################################

def inside_function_fit_a_and_b_to_discharge_probability_curve(meteo_df, x_data_df, y_data_df, x_fit_df, y_fit_df, r2_df, function, axes):
    print("inside_function_fit_a_and_b_to_discharge_probability_curve")

    # Select the dates
    date1 = '2010-06-01'
    date2 = '2010-06-06'
    meteo_df = meteo_df[(meteo_df['Date'] >= date1) & (meteo_df['Date'] <= date2)]
    x_data_df = x_data_df[(x_data_df['Date'] >= date1) & (x_data_df['Date'] <= date2)]
    y_data_df = y_data_df[(y_data_df['Date'] >= date1) & (y_data_df['Date'] <= date2)]
    x_fit_df = x_fit_df[(x_fit_df['Date'] >= date1) & (x_fit_df['Date'] <= date2)]
    y_fit_df = y_fit_df[(y_fit_df['Date'] >= date1) & (y_fit_df['Date'] <= date2)]
    r2_df = r2_df[(r2_df['Date'] >= date1) & (r2_df['Date'] <= date2)]

    start_date = pd.to_datetime(date1)
    end_date = pd.to_datetime(date2)
    date_range = pd.date_range(start_date, end_date, freq='D')

    a = meteo_df["$a$"]
    b = meteo_df["$b$"]
    if function == "Singh2014":
        pass
    elif function == "Sigmoid_d":
        c = meteo_df["$c$"]
        d = meteo_df["$d$"]
    elif function == "Sigmoid":
        c = meteo_df["$c$"]

    for i, day in enumerate(date_range.strftime('%Y-%m-%d')):
        print(f"Processing day {day}")
        print(f"i: {i}")

        x_data = x_data_df.iloc[i].values[1:]
        y_data = y_data_df.iloc[i].values[1:]
        x_fit = x_fit_df.iloc[i].values[1:]
        y_fit = y_fit_df.iloc[i].values[1:]
        r2 = r2_df.iloc[i].values[1:]

        if function == "Singh2014":
            label = f"r$^2$ = {r2[0]:.2f}\na = {a.iloc[i]:.1f}\nb = {b.iloc[i]:.1f}"
            #label = f'Fitted Curve: 1 - {a.iloc[i]:.2f} * x^{b.iloc[i]:.2f}'
        elif function == "Sigmoid_d":
            label = f"r$^2$ = {r2[0]:.2f}\na = {a.iloc[i]:.1f}\nb = {b.iloc[i]:.1f}\nc = {c.iloc[i]:.1f}\nd = {d.iloc[i]:.1f}"
            #label = f'Fitted Curve: {d.iloc[i]:.2f}/(1 + exp({a.iloc[i]:.2f}(x-{b.iloc[i]:.2f}))) + {c.iloc[i]:.2f}x - {d.iloc[i]:.2f} + 1'
        elif function == "Sigmoid":
            print(meteo_df)
            print(a, b, c)
            print(r2)
            print(r2[0])
            print(a.iloc[i])
            print(b.iloc[i])
            print(c.iloc[i])
            label = f"r$^2$ = {r2[0]:.2f}\na = {a.iloc[i]:.1f}\nb = {b.iloc[i]:.1f}\nc = {c.iloc[i]:.1f}"
            #label = f'Fitted Curve: ({c.iloc[i]:.2f} + 1)/(1 + exp({a.iloc[i]:.2f}(x-{b.iloc[i]:.2f}))) + {c.iloc[i]:.2f}x - {c.iloc[i]:.2f}'

        # Plotting the observations and the fitted curve
        axes[i].set_xticklabels([])
        axes[i].scatter(x_data, y_data, label='Observations', s=2)
        axes[i].plot(y_fit, x_fit, label='Fit', color='red')
        axes[i].text(.30, .1, day, fontsize=8, horizontalalignment='center',
                     verticalalignment='center', transform=axes[i].transAxes)
        axes[i].text(0.8, .7, label, fontsize=8, horizontalalignment='center',
                     verticalalignment='center', transform=axes[i].transAxes)

def inside_fit_m_to_flow_duration_curve(meteo_df, x_data_df, y_data_df, x_fit_df, y_fit_df, r2_df, function, axes):

    # Select the dates
    date1 = '2010-06-01'
    date2 = '2010-06-06'
    meteo_df = meteo_df[(meteo_df['Date'] >= date1) & (meteo_df['Date'] <= date2)]
    x_data_df = x_data_df[(x_data_df['Date'] >= date1) & (x_data_df['Date'] <= date2)]
    y_data_df = y_data_df[(y_data_df['Date'] >= date1) & (y_data_df['Date'] <= date2)]
    x_fit_df = x_fit_df[(x_fit_df['Date'] >= date1) & (x_fit_df['Date'] <= date2)]
    y_fit_df = y_fit_df[(y_fit_df['Date'] >= date1) & (y_fit_df['Date'] <= date2)]
    r2_df = r2_df[(r2_df['Date'] >= date1) & (r2_df['Date'] <= date2)]

    start_date = pd.to_datetime(date1)
    end_date = pd.to_datetime(date2)
    date_range = pd.date_range(start_date, end_date, freq='D')

    a = meteo_df["$a$"]
    b = meteo_df["$b$"]
    M = meteo_df["$M$"]
    if function == "Singh2014":
        pass
    elif function == "Sigmoid_d":
        c = meteo_df["$c$"]
        d = meteo_df["$d$"]
    elif function == "Sigmoid":
        c = meteo_df["$c$"]

    for i, day in enumerate(date_range.strftime('%Y-%m-%d')):
        print(f"Processing day {day}")

        x_data = x_data_df.iloc[i].values[1:]
        y_data = y_data_df.iloc[i].values[1:]
        t_fit = x_fit_df.iloc[i].values[1:]
        y_fit = y_fit_df.iloc[i].values[1:]
        r2 = r2_df.iloc[i].values[1:]

        if function == "Singh2014":
            label = f"a = {a.iloc[i]:.1f}\nb = {b.iloc[i]:.1f}\nM = {M.iloc[i]:.1f}\nr$^2$ = {r2[0]:.2f}"
            #label = f'Fitted Curve: 1 - {a.iloc[i]:.2f} * x^{b.iloc[i]:.2f}'
        elif function == "Sigmoid_d":
            label = f"a = {a.iloc[i]:.1f}\nb = {b.iloc[i]:.1f}\nc = {c.iloc[i]:.1f}\nd = {d.iloc[i]:.1f}\nM = {M.iloc[i]:.1f}\nr$^2$ = {r2[0]:.2f}"
            #label = f'Fitted Curve: {d.iloc[i]:.2f}/(1 + exp({a.iloc[i]:.2f}(x-{b.iloc[i]:.2f}))) + {c.iloc[i]:.2f}x - {d.iloc[i]:.2f} + 1'
        elif function == "Sigmoid":
            label = f"a = {a.iloc[i]:.1f}\nb = {b.iloc[i]:.1f}\nc = {c.iloc[i]:.1f}\nM = {M.iloc[i]:.1f}\nr$^2$ = {r2[0]:.2f}"
            #label = f'Fitted Curve: ({c.iloc[i]:.2f} + 1)/(1 + exp({a.iloc[i]:.2f}(x-{b.iloc[i]:.2f}))) + {c.iloc[i]:.2f}x - {c.iloc[i]:.2f}'

        # Plot the resulting graph (equivalent of Fig. 1 of Singh et al., 2014)
        # Plotting the original data and the fitted curve
        axes[i].set_xticklabels([])
        axes[i].scatter(x_data, y_data / np.nanmax(y_data), label='Observations', s=2)
        axes[i].plot(t_fit, y_fit / np.nanmax(y_fit), label='Fit', color='red')
        axes[i].text(.30, .1, day, fontsize=8, horizontalalignment='center',
                     verticalalignment='center', transform=axes[i].transAxes)
        axes[i].text(0.8, .7, label, fontsize=8, horizontalalignment='center',
                     verticalalignment='center', transform=axes[i].transAxes)

def figure3(filename, fp, figures=''):

    fig, axes = plt.subplots(6, 3, figsize=(7,8))
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.6)
    axes = axes.T

    df = pd.read_csv(filename, header=0, na_values='', usecols=[0, 1],
                     parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    df.rename(columns={df.columns[1]: 'Discharge'}, inplace=True)
    df.index = pd.to_datetime(df.iloc[:,0].values, format='%Y-%m-%d').strftime('%Y-%m-%d')
    inside_hydrograph_plot(df, axes[0])

    function = "Singh2014"
    meteo_df = pd.read_csv(fp.dataframe_constrained_filename, index_col=0)
    x_data_df = pd.read_csv(fp.x_data1_filename)
    y_data_df = pd.read_csv(fp.y_data1_filename)
    x_fit_df = pd.read_csv(fp.x_fit1_filename)
    y_fit_df = pd.read_csv(fp.y_fit1_filename)
    r2_df = pd.read_csv(fp.r21_filename)
    x_data_df.columns.values[0] = "Date"
    y_data_df.columns.values[0] = "Date"
    x_fit_df.columns.values[0] = "Date"
    y_fit_df.columns.values[0] = "Date"
    r2_df.columns.values[0] = "Date"
    inside_function_fit_a_and_b_to_discharge_probability_curve(meteo_df, x_data_df, y_data_df, x_fit_df, y_fit_df, r2_df, function, axes[1])

    function = "Sigmoid"
    meteo_df = pd.read_csv(fp.dataframe_constrained_filename, index_col=0)
    x_data_df = pd.read_csv(fp.x_data1_filename)
    y_data_df = pd.read_csv(fp.y_data1_filename)
    x_fit_df = pd.read_csv(fp.x_fit1_filename)
    y_fit_df = pd.read_csv(fp.y_fit1_filename)
    r2_df = pd.read_csv(fp.r21_filename)
    x_data_df.columns.values[0] = "Date"
    y_data_df.columns.values[0] = "Date"
    x_fit_df.columns.values[0] = "Date"
    y_fit_df.columns.values[0] = "Date"
    r2_df.columns.values[0] = "Date"
    inside_function_fit_a_and_b_to_discharge_probability_curve(meteo_df, x_data_df, y_data_df, x_fit_df, y_fit_df, r2_df, function, axes[2])

    #axes[0].set_title("Hydrograph")
    axes[0][0].set_title("a)")
    axes[0][2].set_ylabel("Normalized discharge")
    axes[0][-1].set_xlabel("Time t")
    axes[0][-1].xaxis.set_major_formatter(dates.DateFormatter('%H h'))
    axes[0][-1].xaxis.set_major_locator(dates.HourLocator(byhour=[0, 12, 24]))
    #axes[0].set_title("First fit:\nFitting " + title_params + " using Least Squares")
    axes[1][0].set_title("b)")
    axes[1][2].set_ylabel('F(Q)')
    axes[1][-1].set_xlabel("t/T, T=24h")
    axes[1][-1].set_xticklabels([0, 0, 0.5, 1])
    axes[2][0].set_title("c)")
    axes[2][2].set_ylabel('F(Q)')
    axes[2][-1].set_xlabel("t/T, T=24h")
    axes[2][-1].set_xticklabels([0, 0, 0.5, 1])
    axes[1][-1].legend(loc='upper left', bbox_to_anchor=(-0.4, -0.45), frameon=False, ncols=2)
    fig.savefig(f'{figures}figure3.pdf', format="pdf", bbox_inches='tight', dpi=100)

def figure4(filename, fp, figures=''):

    fig, axes = plt.subplots(6, 3, figsize=(7,8))
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.6)
    axes = axes.T

    df = pd.read_csv(filename, header=0, na_values='', usecols=[0, 1],
                     parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    df.rename(columns={df.columns[1]: 'Discharge'}, inplace=True)
    df.index = pd.to_datetime(df.iloc[:,0].values, format='%Y-%m-%d').strftime('%Y-%m-%d')
    inside_hydrograph_plot(df, axes[0])

    function = "Singh2014"
    meteo_df = pd.read_csv(fp.dataframe_constrained_filename, index_col=0)
    x_data2_df = pd.read_csv(fp.x_data2_filename)
    y_data2_df = pd.read_csv(fp.y_data2_filename)
    t_fit2_df = pd.read_csv(fp.t_fit2_filename)
    y_fit2_df = pd.read_csv(fp.y_fit2_filename)
    r22_df = pd.read_csv(fp.r22_filename)
    x_data2_df.columns.values[0] = "Date"
    y_data2_df.columns.values[0] = "Date"
    t_fit2_df.columns.values[0] = "Date"
    y_fit2_df.columns.values[0] = "Date"
    r22_df.columns.values[0] = "Date"
    inside_fit_m_to_flow_duration_curve(meteo_df, x_data2_df, y_data2_df, t_fit2_df, y_fit2_df, r22_df, function, axes[1])

    function = "Sigmoid"
    meteo_df = pd.read_csv(fp.dataframe_constrained_filename, index_col=0)
    x_data2_df = pd.read_csv(fp.x_data2_filename)
    y_data2_df = pd.read_csv(fp.y_data2_filename)
    t_fit2_df = pd.read_csv(fp.t_fit2_filename)
    y_fit2_df = pd.read_csv(fp.y_fit2_filename)
    r22_df = pd.read_csv(fp.r22_filename)
    x_data2_df.columns.values[0] = "Date"
    y_data2_df.columns.values[0] = "Date"
    t_fit2_df.columns.values[0] = "Date"
    y_fit2_df.columns.values[0] = "Date"
    r22_df.columns.values[0] = "Date"
    inside_fit_m_to_flow_duration_curve(meteo_df, x_data2_df, y_data2_df, t_fit2_df, y_fit2_df, r22_df, function, axes[2])

    #axes[0].set_title("Hydrograph")
    axes[0][0].set_title("a)")
    axes[0][2].set_ylabel("Normalized discharge")
    axes[0][-1].set_xlabel("Time t")
    axes[0][-1].xaxis.set_major_formatter(dates.DateFormatter('%H h'))
    axes[0][-1].xaxis.set_major_locator(dates.HourLocator(byhour=[0, 12, 24]))
    #axes[0].set_title("Flow duration curve:\nFitting M using Least Squares")
    axes[1][0].set_title("b)")
    axes[1][2].set_ylabel('Normalized discharge')
    axes[1][-1].set_xlabel("t/T, T=24h")
    axes[1][-1].set_xticklabels([0, 0, 0.5, 1])
    axes[2][0].set_title("c)")
    axes[2][2].set_ylabel('Normalized discharge')
    axes[2][-1].set_xlabel("t/T, T=24h")
    axes[2][-1].set_xticklabels([0, 0, 0.5, 1])
    axes[1][-1].legend(loc='upper left', bbox_to_anchor=(-0.4, -0.45), frameon=False, ncols=2)
    fig.savefig(f'{figures}figure4.pdf', format="pdf", bbox_inches='tight', dpi=100)

def fit_a_and_b_to_discharge_probability_curve(meteo_df, x_data_df, y_data_df, x_fit_df, y_fit_df, r2_df,
                                               function="Singh2014", figures=''):

    fig, axes = plt.subplots(6, 1, figsize=(2,8), sharex=True)
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.3)

    inside_function_fit_a_and_b_to_discharge_probability_curve(meteo_df, x_data_df, y_data_df, x_fit_df, y_fit_df, r2_df, function, axes)

    if function == "Singh2014":
        title_params = "a and b"
    elif function == "Sigmoid_d":
        title_params = "a, b, c and d"
    elif function == "Sigmoid":
        title_params = "a, b and c"
    axes[0].set_title("First fit:\nFitting " + title_params + " using Least Squares")
    axes[2].set_ylabel('F(Q)')
    axes[-1].set_xlabel("t/T, T=24h")
    axes[-1].set_xticklabels([0, 0, 0.5, 1])
    axes[-1].legend(loc='upper left', bbox_to_anchor=(-0.4, -0.45), frameon=False, ncols=2)
    fig.savefig(f'{figures}fit_a_and_b_to_discharge_probability_curve_{function}.pdf', format="pdf", bbox_inches='tight', dpi=100)

def fit_m_to_flow_duration_curve(meteo_df, x_data_df, y_data_df, x_fit_df, y_fit_df, r2_df, function="Singh2014", figures=''):
    # Plot the comparison of the flow duration curves (Fig. 6 of Singh et al., 2014)

    fig, axes = plt.subplots(6, 1, figsize=(2,8), sharex=True)
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.3)

    inside_fit_m_to_flow_duration_curve(meteo_df, x_data_df, y_data_df, x_fit_df, y_fit_df, r2_df, function, axes)

    axes[0].set_title("Flow duration curve:\nFitting M using Least Squares")
    axes[2].set_ylabel('Discharge')
    axes[-1].set_xlabel('t/T, T=24h')
    axes[-1].set_xticklabels([0, 0, 0.5, 1])
    axes[-1].legend(loc='upper left', bbox_to_anchor=(-0.4, -0.45), frameon=False, ncols=2)
    fig.savefig(f'{figures}fit_m_to_flow_duration_curve_{function}.pdf', format="pdf", bbox_inches='tight', dpi=100)

def histogram(axis, x, x_label, results, density, bins=20, alpha=1, range=None, label=None, color=None):
    axis.hist(x, bins=bins, range=range, density=density, alpha=alpha, label=label, color=color)
    axis.set_xlabel(x_label)

def combined_histograms(meteo_df, results, filename, function="Sigmoid",
                        pdfs=None, kde=False):
    if function == "Singh2014":
        column_nb = 3
        labels = ["$a$", "$b$", "$M$", "$Q_{min}$", "$Q_{max}$"]
    elif function == "Sigmoid_d":
        column_nb = 4
        labels = ["$a$", "$b$", "$c$", "$d$", "$M$", "$Q_{min}$", "$Q_{max}$"]
    elif function == "Sigmoid":
        column_nb = 3
        labels = ["$a$", "$b$", "$c$", "$M$", "$Q_{min}$", "$Q_{max}$"]

    data_cleaned = []
    for l in labels:
        data_cleaned.append(meteo_df[l].dropna().to_numpy())

    fig, axes = plt.subplots(2, column_nb, figsize=(11, 6))
    if function == "Sigmoid_d" or function == "Singh2014":
        fig.delaxes(axes.flatten()[-1])
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.2)

    for i, x in enumerate(data_cleaned):
        density = False
        if pdfs:
            density = True
        histogram(axes.flatten()[i], x, labels[i], results, density=density)
        if kde:
            # KDE modelisation
            x_eval = np.linspace(np.nanmin(x) - 1, np.nanmax(x) + 1, 500)
            kde = stats.gaussian_kde(x)
            axes.flatten()[i].plot(x_eval, kde(x_eval), 'm-', label="KDE")

    # Add the fitted pdfs
    if pdfs:
        pdfs = pd.read_csv(pdfs, index_col=0)
        for i, ind in enumerate(pdfs.index):
            typ = pdfs.loc[pdfs.index==ind]["type"].values[0]
            params_str = pdfs.loc[pdfs.index==ind]["params"].values[0]
            params = tuple(float(x) for x in params_str.strip('()').split(','))
            stat = pdfs.loc[pdfs.index==ind]["stats"].values[0]
            p0 = params[0]
            p1 = params[1]
            p2 = params[2]
            if typ == "lognormal":
                print('lognormal')
                pdf = stats.lognorm(*params)
                label = f"Lognormal\nsigma = {p0:.2f}\nloc = {p1:.2f}\n"\
                        f"scale = {p2:.0f}\nKS test: {stat:.2f}"
            elif typ == "gamma":
                print('gamma')
                pdf = stats.gamma(*params)
                label = f"Gamma\nshape = {p0:.2f}\nloc = {p1:.2f}\n"\
                        f"scale = {p2:.0f}\nKS test: {stat:.2f}"
            elif typ == "beta":
                print('beta')
                pdf = stats.beta(*params)
                p3 = params[3]
                label = f"Beta\na = {p0:.2f}\nb = {p1:.2f}\n"\
                        f"loc = {p2:.2f}\nscale = {p3:.0f}\nKS test: {stat:.2f}"
            x = np.linspace(np.nanmin(data_cleaned[i]), np.nanmax(data_cleaned[i]), 100)
            axes.flatten()[i].plot(x, pdf.pdf(x), label="PDF")
            axes.flatten()[i].text(0.8, .5, label, fontsize=8, horizontalalignment='center',
                                   verticalalignment='center', transform=axes.flatten()[i].transAxes)
        filename += "_with_pdfs"
        axes[0][0].set_ylabel("Probability density")
        axes[1][0].set_ylabel("Probability density")
    else:
        axes[0][0].set_ylabel("Count")
        axes[1][0].set_ylabel("Count")
    plt.legend(frameon=False)
    plt.savefig(filename + ".pdf", format="pdf", bbox_inches='tight', dpi=100)

def plot_influence(x, y, x_label, y_label):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'{results}influence_{x_label}_on_{y_label}.pdf')

def pairplot(meteo_df, output_figure):

    meteo_df["Weather"] = 'None'
    meteo_df.loc[(meteo_df["Temperature"] <= 0), "Weather"] = 'Freezing'
    meteo_df.loc[(meteo_df["Temperature"] > 0), "Weather"] = 'Melting'
    meteo_df.loc[(meteo_df["Precipitation"] > 0) & (meteo_df["Temperature"] > 0), "Weather"] = 'Raining'
    meteo_df.loc[(meteo_df["Precipitation"] > 0) & (meteo_df["Temperature"] <= 0), "Weather"] = 'Snowing'

    grid = sns.pairplot(meteo_df, hue="Weather",
                        hue_order=['Raining', 'Melting', 'Snowing', 'Freezing'],
                        palette="Set2")
    #grid = sns.PairGrid(meteo_df, hue="Weather",
    #                    hue_order=['Raining', 'Melting', 'Snowing', 'Freezing'],
    #                    palette="Set2") #'line_kws':{'color':'red'} #"s": 6 #corner=True, plot_kws={"s": 6},kind='reg'
    #grid.map_lower(sns.scatterplot, markers=["o", "s", "D", ">"])
    #u = grid.map_upper(sns.kdeplot, levels=4, color=".2")
    grid.fig.set_size_inches(25, 25)
    plt.savefig(output_figure, format="pdf", bbox_inches='tight', dpi=100)

def prepare_station_data(station_data_filename, yearly):
        
    station_data = pd.read_csv(station_data_filename, sep=";", decimal=".", encoding='cp1252', header=0, index_col=0,
                               usecols=[1, 2], na_values='-', parse_dates=['time'], date_format='%Y%m%d')
    station_data = station_data.rename(columns={station_data.columns[0] : 'Temperature'})
    # The precipitation is read as string, convert as float
    station_data['Temperature'] = station_data['Temperature'].astype(float)
    if yearly:
        station_temp_yr = station_data.groupby(pd.Grouper(freq='1Y')).mean()
        # Because otherwise the data is located on the 31st of December and feels like it's shifted.
        station_temp_yr.index = station_temp_yr.index.map(lambda t: t.replace(month=1, day=1))
    else:
        station_temp_yr = station_data
    all_yrs_station_temp_yr = station_temp_yr[station_temp_yr.index.year.isin(list(range(1969, 2015)))]
        
    return all_yrs_station_temp_yr

def plot_discharge_variability(catchments, months_str, function, meteo_station, 
                               filename, yearly=True):

    fig, axes = plt.subplots(6, 1, figsize=(9, 11), sharex=True)
    fig.subplots_adjust(hspace=0)
    cmap = plt.get_cmap('viridis')
    normalize = mcolors.Normalize(vmin=0, vmax=len(catchments) - 1)
    markers = ['+', 'o', 'v', '^', 'p', 'd', 's']
    
    GRC_all_yr = prepare_station_data(f"{meteo_station}order_127340_data_GRC.txt", yearly)
    #GSB_all_yr = prepare_station_data(f"{meteo_station}order_127340_data_GSB.txt", yearly)
    #SIO_all_yr = prepare_station_data(f"{meteo_station}order_127340_data_SIO.txt", yearly)
    
    # If summer plot
    if not yearly:
        # Select the dates
        date1 = '2010-06-01'
        date2 = '2010-09-30'
        filename = filename + "_summer_2010"
        GRC_all_yr = GRC_all_yr[(GRC_all_yr.index >= date1) & (GRC_all_yr.index <= date2)]
        
    line1, = axes[5].plot(GRC_all_yr.index, GRC_all_yr.values, '-', color="wheat", zorder=10)
    #line2, = axes[5].plot(GSB_all_yr.index, GSB_all_yr.values, '-', color="wheat", zorder=10)
    #line3, = axes[5].plot(SIO_all_yr.index, SIO_all_yr.values, '-', color="wheat", zorder=10)

    max_range = 0
    yearly_meteo_df_list = []
    for i, catchment in enumerate(catchments):
        results = f"/home/anne-laure/Documents/Datasets/Outputs/Arolla/{catchment}/"

        dataframe_filename = f"{results}meteo_df_{function}_{months_str}_constrained.csv"
        meteo_df = pd.read_csv(dataframe_filename, index_col=0)
        meteo_df.index = pd.to_datetime(meteo_df.index)
        
        meteo_df = filtering_data(meteo_df, ['$a$', '$b$', '$c$', '$M$'])
        meteo_df.drop(['Date'], axis=1, inplace=True)

        if yearly:
            yearly_meteo_df = meteo_df.groupby(pd.Grouper(freq='Y')).mean()
            # Because otherwise the data is located on the 31st of December and feels like it's shifted.
            yearly_meteo_df.index = yearly_meteo_df.index.map(lambda t: t.replace(month=1, day=1))
            yearly_meteo_df_list.append(yearly_meteo_df)
            range = yearly_meteo_df['$Q_{max}$'].values - yearly_meteo_df['$Q_{min}$'].values
        else:
            meteo_df = meteo_df[(meteo_df.index >= date1) & (meteo_df.index <= date2)]
            yearly_meteo_df_list.append(meteo_df)
            range = meteo_df['$Q_{max}$'].values - meteo_df['$Q_{min}$'].values
            
        max = np.nanmax(range)
        if max > max_range:
            max_range = max

    for i, (catchment, yearly_meteo_df) in enumerate(zip(catchments, yearly_meteo_df_list)):
        results = f"/home/anne-laure/Documents/Datasets/Outputs/Arolla/{catchment}/"
        
        # Fill the missing values with np.nans to avoid interpolation
        idx = pd.date_range(meteo_df.index.min(), meteo_df.index.max())
        meteo_df = meteo_df.reindex(idx, fill_value=np.nan)
        
        if yearly:
            window_size = 4
        else:
            window_size = 8
        five_year_moving_window = yearly_meteo_df.rolling(window=window_size, center=True, 
                                                          closed='both', min_periods=1).mean()

        x = yearly_meteo_df.index.values
        #plt.bar(x, yearly_meteo_df['Precipitation'].values, label='Precipitation (in mm)', color='dodgerblue', alpha=0.5)
        axes[0].plot(x, yearly_meteo_df['$a$'].values, markers[i], color=cmap(normalize(i)), markersize=2, label=catchment)
        if catchment == "DB":
            print("yearly_meteo_df['$a$'].values", yearly_meteo_df['$a$'].values)
        axes[0].plot(x, five_year_moving_window['$a$'].values, color=cmap(normalize(i)), alpha=0.5, linewidth=4)
        axes[1].plot(x, yearly_meteo_df['$b$'].values, markers[i], color=cmap(normalize(i)), markersize=2)
        axes[1].plot(x, five_year_moving_window['$b$'].values, color=cmap(normalize(i)), alpha=0.5, linewidth=4)
        if function == "Sigmoid_d" or function == "Sigmoid":
            axes[2].plot(x, yearly_meteo_df['$c$'].values, markers[i], color=cmap(normalize(i)), markersize=2)
            axes[2].plot(x, five_year_moving_window['$c$'].values, color=cmap(normalize(i)), alpha=0.5, linewidth=4, label=catchment)
        axes[3].plot(x, yearly_meteo_df['$M$'].values, markers[i], color=cmap(normalize(i)), markersize=2)
        axes[3].plot(x, five_year_moving_window['$M$'].values, color=cmap(normalize(i)), alpha=0.5, linewidth=4)
        axes[4].plot(x, yearly_meteo_df['$Q_{max}$'].values, label='$Q_{max}$', color='red')
        axes[4].plot(x, yearly_meteo_df['$Q_{min}$'].values, label='$Q_{min}$', color='blue')
        #axes[4].plot(x, (yearly_meteo_df['$Q_{max}$'].values - yearly_meteo_df['$Q_{min}$'].values) / max_range, color=cmap(normalize(i)), markersize=2)
    
    if yearly:
        pass
    else:
        axes[0].set_ylim([0, 50])
        axes[2].set_ylim([-1.75, 0.25])
        axes[3].set_ylim([-1, 1.5])
        
    if False:
        

        # Plot
        if i == 5:
            label = "5-year moving window"
        else:
            label = None
        axes.plot(norm_five_year_moving_window.index, norm_five_year_moving_window.values, color=color, label=label, alpha=0.5, linewidth=4)

        #meteo_df.plot(kind='line', y='$Q_{mean}$', label='Mean discharge', color='red', ax=ax1)
        #meteo_df.plot(kind='line', y='Entropy', label='Entropy', color='chartreuse', ax=ax1)
        # plt.plot(x, meteo_df['$R^2$'].values*100, 'x', label='M', color='black')

        # METEO DATASET
        # plt.plot(x, yearly_meteo_df['Glacier snow'].values, label='Glacier snow', color='black')
        # plt.plot(x, yearly_meteo_df['Snow melt'].values, label='Glacier snow', color='gray')
        # Temperature
        # y = yearly_meteo_df['Temperature'].values
        # ax.fill_between(x, 0, y, where=y>0, facecolor='red', label='Temperature (in C°): T > 0', interpolate=True, alpha=0.5)
        # ax.fill_between(x, 0, y, where=y<=0, facecolor='blue', label='T < 0', interpolate=True, alpha=0.5)

    plt.xlabel('Time')
    axes[0].set_ylabel('$a$ [-]')
    axes[1].set_ylabel('$b$ [-]')
    axes[2].set_ylabel('$c$ [-]')
    axes[3].set_ylabel('$M$ [-]')
    axes[4].set_ylabel('Normalized\n$Q_{max} - Q_{min}$')
    axes[5].set_ylabel('Temp. [°C]')
    axes[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    axes[2].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.savefig(filename + ".pdf", format="pdf", bbox_inches='tight', dpi=100)

def plot_function_sensitivity(filename, variables, function="Sigmoid"):
    if function == "Singh2014":
        column_nb = 2
        assert len(variables) == 3
        # variables = {"$a$": [], "$b$": [], "$M$": [], "$Q_{min}$": [], "$Q_{max}$": []}
    elif function == "Sigmoid_d":
        column_nb = 3
        assert len(variables) == 5
        # variables = {"$a$": [], "$b$": [], "$c$": [], "$d$": [], "$M$": [], "$Q_{min}$": [], "$Q_{max}$": []}
    elif function == "Sigmoid":
        column_nb = 2
        assert len(variables) == 4
        # variables = {"$a$": [], "$b$": [], "$c$": [], $M$": [], "$Q_{min}$": [], "$Q_{max}$": []}

    fig, axes = plt.subplots(2, column_nb, figsize=(7, 6), sharex=True)
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.3)

    # a, b, q_min, q_max have to be defined outside the function when calling curve_fit,
    # so we define the function to be solved inside this current function.
    def discharge_time_equation_to_solve_Singh2014(tau, M, a, b, q_min, q_max):
        """
        Equation 23 of Singh et al. (2014).
        """
        # Compute the discharge time points from the observed discharges
        numerator = np.exp(-M) - (np.exp(-M) - np.exp(-M * q_min / q_max)) * a * tau ** b
        right_side = - np.log(numerator) / M
        q = q_max * right_side

        return q
    def discharge_time_equation_to_solve_Sigmoid_d(tau, M, a, b, c, d, q_min, q_max):
        """
        Sigmoid version to replace equation 23 of Singh et al. (2014).
        """
        # Compute the discharge time points from the observed discharges
        numerator = np.exp(-M) - (np.exp(-M * q_min / q_max) - np.exp(-M)) * (d / (1 + np.exp(a * (tau - b))) - d / (1 + np.exp(-a * b)) + c * tau)
        right_side = - np.log(numerator) / M
        q = q_max * right_side

        return q
    def discharge_time_equation_to_solve_Sigmoid(tau, M, a, b, c, q_min, q_max):
        """
        Sigmoid version to replace equation 23 of Singh et al. (2014), with d = c + 1.
        """
        # Compute the discharge time points from the observed discharges
        numerator = np.exp(-M) - (np.exp(-M * q_min / q_max) - np.exp(-M)) * ((c + 1) / (1 + np.exp(a * (tau - b))) - (c + 1) / (1 + np.exp(-a * b)) + c * tau)
        right_side = - np.log(numerator) / M
        q = q_max * right_side

        return q

    cmap = plt.get_cmap('viridis')
    for ax, (key, values) in zip(axes.flatten(), variables.items()):
        normalize = mcolors.Normalize(vmin=0, vmax=len(values) - 1)

        t_fit = np.linspace(0, 1.0, 100)
        M = 1
        a = 10
        b = 0.5
        c = -0.3
        q_min = 1
        q_max = 4
        label = f"Default values:\n$a$ = {a:.1f} [-],\n$b$ = {b:.1f} [-],\n$c$ = {c:.1f} [-],\n"\
                f"$M$ = {M:.1f} [-],\n" + "$Q_{min}$" + f" = {q_min:.1f} [-],\n" + "$Q_{max}$" + f" = {q_max:.1f} [-]"

        for i, val in enumerate(values):
            if key == "$a$":
                a = val
            elif key == "$b$":
                b = val
            elif key == "$c$":
                c = val
            elif key == "$M$":
                M = val
            elif key == "$Q_{min}$":
                q_min = val
            elif key == "$Q_{max}$":
                q_max = val

            # Generate fitted curve for plotting
            if function == "Singh2014":
                y_fit = discharge_time_equation_to_solve_Singh2014(t_fit, M, a, b, q_min, q_max)  # Use *params[1:] to unpack the remaining parameters
            elif function == "Sigmoid_d":
                y_fit = discharge_time_equation_to_solve_Sigmoid_d(t_fit, M, a, b, c, d, q_min, q_max)
            elif function == "Sigmoid":
                y_fit = discharge_time_equation_to_solve_Sigmoid(t_fit, M, a, b, c, q_min, q_max)

            ax.plot(t_fit, y_fit, label=f"{key} = {val:.1f}", color=cmap(normalize(i)))

        ax.legend(frameon=False)

    axes[0][0].text(0.1, 0.7, "a)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[0][0].transAxes)
    axes[0][1].text(0.1, 0.7, "b)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[0][1].transAxes)
    axes[1][0].text(0.1, 0.7, "c)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[1][0].transAxes)
    axes[1][1].text(0.1, 0.7, "d)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[1][1].transAxes)

    axes[0][0].set_title("Step slope parameter")
    axes[0][1].set_title("Step location parameter")
    axes[1][0].set_title("Curviness parameter")
    axes[1][1].set_title("Shift parameter")

    axes[0][0].text(-0.45, .5, label, fontsize=10, horizontalalignment='center',
                    verticalalignment='center', transform=axes[0][0].transAxes)
    plt.savefig(filename, format="pdf", bbox_inches='tight', dpi=100)
    
    

def plot_Qmin_qmax_figure(filename, catchments, path, months_str, pdfs=True, function="Sigmoid"):
    
    colors = ["mediumseagreen", "darkorchid", "darkorange", "gold", "crimson", "cornflowerblue", "gray"]

    variables = {"$Q_{min}$": np.linspace(0, 3, 5), 
                 "$Q_{max}$": np.linspace(2, 10, 5)}
    labels = ["$Q_{min}$", "$Q_{max}$"]
    column_nb = 2
    assert len(variables) == 2

    fig, axes = plt.subplots(2, column_nb, figsize=(7, 6)) #, sharex=True
    #if function == "Sigmoid_d" or function == "Singh2014":
    #    fig.delaxes(axes.flatten()[-1])
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    
    max_norm_discharge = 0
    data_cleaned_list = []
    for catchment in catchments:

        # Fetch the catchment data
        results = f"{path}Outputs/Arolla/{catchment}/"
        dataframe_filename = f"{results}meteo_df_{function}_{months_str}_constrained.csv"
        meteo_df = pd.read_csv(dataframe_filename, index_col=0)

        # The calibrated parameters
        data_cleaned = []
        for l in labels:
            data_cleaned.append(meteo_df[l].dropna().to_numpy())
        data_cleaned_list.append(data_cleaned)
        
        # For normalization of discharge
        max_discharge = np.nanmax(meteo_df["$Q_{max}$"].dropna().to_numpy())
        if max_discharge > max_norm_discharge:
            max_norm_discharge = max_discharge

    for catchment, color, data_cleaned in zip(catchments, colors, data_cleaned_list):
        # Fetch the catchment data
        results = f"{path}Outputs/Arolla/{catchment}/"

        # Plotting
        for i, x in enumerate(data_cleaned):
            x = x / max_norm_discharge
            density = False
            if pdfs:
                density = True
            histogram(axes[0].flatten()[i], x, "Normalized " + labels[i], results, density=density, bins=50,
                      range=[0, 1], alpha=0.5, label=catchment, color=color)
            # KDE modelisation
            x_eval = np.linspace(np.nanmin(x) - 1, np.nanmax(x) + 1, 500)
            kde = stats.gaussian_kde(x)
            axes[0].flatten()[i].plot(x_eval, kde(x_eval), color=color)
            axes[0].flatten()[i].set_xlim(0, 1)
    axes[0][1].legend(frameon=False)

    # a, b, q_min, q_max have to be defined outside the function when calling curve_fit,
    # so we define the function to be solved inside this current function.
    def discharge_time_equation_to_solve_Singh2014(tau, M, a, b, q_min, q_max):
        """
        Equation 23 of Singh et al. (2014).
        """
        # Compute the discharge time points from the observed discharges
        numerator = np.exp(-M) - (np.exp(-M) - np.exp(-M * q_min / q_max)) * a * tau ** b
        right_side = - np.log(numerator) / M
        q = q_max * right_side

        return q
    def discharge_time_equation_to_solve_Sigmoid_d(tau, M, a, b, c, d, q_min, q_max):
        """
        Sigmoid version to replace equation 23 of Singh et al. (2014).
        """
        # Compute the discharge time points from the observed discharges
        numerator = np.exp(-M) - (np.exp(-M * q_min / q_max) - np.exp(-M)) * (d / (1 + np.exp(a * (tau - b))) - d / (1 + np.exp(-a * b)) + c * tau)
        right_side = - np.log(numerator) / M
        q = q_max * right_side

        return q
    def discharge_time_equation_to_solve_Sigmoid(tau, M, a, b, c, q_min, q_max):
        """
        Sigmoid version to replace equation 23 of Singh et al. (2014), with d = c + 1.
        """
        # Compute the discharge time points from the observed discharges
        numerator = np.exp(-M) - (np.exp(-M * q_min / q_max) - np.exp(-M)) * ((c + 1) / (1 + np.exp(a * (tau - b))) - (c + 1) / (1 + np.exp(-a * b)) + c * tau)
        right_side = - np.log(numerator) / M
        q = q_max * right_side

        return q

    cmap = plt.get_cmap('viridis')
    for ax, (key, values) in zip(axes[1].flatten(), variables.items()):
        normalize = mcolors.Normalize(vmin=0, vmax=len(values) - 1)

        t_fit = np.linspace(0, 1.0, 100)
        M = 1
        a = 10
        b = 0.5
        c = -0.3
        q_min = 1
        q_max = 4
        label = f"Default values:\n$a$ = {a:.1f} [-],\n$b$ = {b:.1f} [-],\n$c$ = {c:.1f} [-],\n"\
                f"$M$ = {M:.1f} [-],\n" + "$Q_{min}$" + f" = {q_min:.1f} [-],\n" + "$Q_{max}$" + f" = {q_max:.1f} [-]"

        for i, val in enumerate(values):
            if key == "$a$":
                a = val
            elif key == "$b$":
                b = val
            elif key == "$c$":
                c = val
            elif key == "$M$":
                M = val
            elif key == "$Q_{min}$":
                q_min = val
            elif key == "$Q_{max}$":
                q_max = val

            # Generate fitted curve for plotting
            if function == "Singh2014":
                y_fit = discharge_time_equation_to_solve_Singh2014(t_fit, M, a, b, q_min, q_max)  # Use *params[1:] to unpack the remaining parameters
            elif function == "Sigmoid_d":
                y_fit = discharge_time_equation_to_solve_Sigmoid_d(t_fit, M, a, b, c, d, q_min, q_max)
            elif function == "Sigmoid":
                y_fit = discharge_time_equation_to_solve_Sigmoid(t_fit, M, a, b, c, q_min, q_max)

            ax.plot(t_fit, y_fit, label=f"{key} = {val:.1f}", color=cmap(normalize(i)))

        ax.legend(frameon=False)

    axes[0][0].text(0.1, 0.7, "a)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[0][0].transAxes)
    axes[0][1].text(0.1, 0.7, "b)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[0][1].transAxes)
    axes[1][0].text(0.1, 0.7, "c)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[1][0].transAxes)
    axes[1][1].text(0.1, 0.7, "d)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[1][1].transAxes)

    axes[0][0].set_title("")
    axes[0][1].set_title("")
    axes[1][0].set_title("Minimum FDC value")
    axes[1][1].set_title("Maximum FDC value")

    if pdfs:
        filename += "_with_pdfs"
        axes[0][0].set_ylabel("Probability density")
        axes[0][1].set_ylabel("Probability density")
    else:
        axes[0][0].set_ylabel("Count")
        axes[0][1].set_ylabel("Count")

    axes[1][0].text(-0.45, .5, label, fontsize=10, horizontalalignment='center',
                    verticalalignment='center', transform=axes[1][0].transAxes)
    plt.savefig(filename + ".pdf", format="pdf", bbox_inches='tight', dpi=100)

def plot_q_mean_q_min_q_max_regressions(meteo_df, regression_filename, filename,
                                        test_influences=True, regr=False, 
                                        influence_label='Ice melt', 
                                        start_date='2010-01-01', 
                                        end_date='2014-12-31'):
    

    meteo_df = meteo_df[(meteo_df['Date'] > start_date) & (meteo_df['Date'] <= end_date)]
    
    label = influence_label
    if test_influences:
        filename += "_" + influence_label
        # 'Date', 'Precipitation', 'Temperature', 'Snow melt', 'Ice melt', 'All snow',
        # 'Glacier snow', '$a$', '$b$', '$c$', '$M$', '$Q_{min}$', '$Q_{max}$', '$Q_{mean}$', 'Entropy',
        # 'Day of the year', 'Glacier area percentage', 'Radiation'
        cmap = plt.get_cmap('viridis')
        normalize = mcolors.Normalize(vmin=np.nanmin(meteo_df[influence_label].values),
                                      vmax=np.nanmax(meteo_df[influence_label].values))
        # Add units
        if influence_label in ['Precipitation', 'Snow melt', 'Ice melt', 'All snow', 'Glacier snow']:
            label = influence_label + " [mm]"
        elif influence_label == 'Temperature':
            label = influence_label + " [°C]"
        elif influence_label == 'Day of the year':
            label = influence_label + " [day]"
        elif influence_label == 'Radiation':
            label = influence_label + " [W m$^{-2}$]"
        else:
            label = influence_label + " [-]"

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.5))
    
    max_disch = np.nanmax(meteo_df['$Q_{max}$'].values)

    # Plot the data while testing for the influence of a variable
    if test_influences:
        cax = fig.add_axes([0.89, 0.15, 0.01, 0.5])
        for (mean, min, max, c) in zip(meteo_df['$Q_{mean}$'].values,
                                       meteo_df['$Q_{min}$'].values,
                                       meteo_df['$Q_{max}$'].values,
                                       cmap(normalize(meteo_df[influence_label].values))):
            axes[0].plot(mean / max_disch, min / max_disch, '+', color=c, markersize=5, label="$Q_{min,d}$")
            axes[0].plot(mean / max_disch, max / max_disch, 'o', color=c, markersize=3, label="$Q_{max,d}$")
            axes[1].plot(min / max_disch, max / max_disch, 's', color=c, markersize=3)
        ColorbarBase(cax, cmap=cmap, norm=normalize)
        cax.set_ylabel(label)
        cax.grid(axis='y')

        handles, labels = axes[0].get_legend_handles_labels()
        handles = handles[:2]
        labels = labels[:2]
    else: # Plot the data without testing
        axes[0].plot(meteo_df['$Q_{mean}$'].values / max_disch, 
                     meteo_df['$Q_{min}$'].values / max_disch, 
                     '+', label='$Q_{min}$', color='blue')
        axes[0].plot(meteo_df['$Q_{mean}$'].values / max_disch, 
                     meteo_df['$Q_{max}$'].values / max_disch, 
                     '+', label='$Q_{max}$', color='teal')
    
    if regr:
        # Retrieve the regression data
        regressions_df = pd.read_csv(regression_filename, header=0)
    
        q_min_data = regressions_df.loc[regressions_df["variable2"] == "$Q_{min}$"]
        q_max_data = regressions_df.loc[regressions_df["variable2"] == "$Q_{max}$"]
        if test_influences:
            q_min_data = q_min_data.loc[q_min_data["variable3"].notna()]
            q_max_data = q_max_data.loc[q_max_data["variable3"].notna()]
        else:
            q_min_data = q_min_data.loc[q_min_data["variable3"].isna()]
            q_max_data = q_max_data.loc[q_max_data["variable3"].isna()]
        
        q_min_coef = float(q_min_data["coefs"].values[0])
        q_max_coef = float(q_max_data["coefs"].values[0])
        q_min_intercept = float(q_min_data["intercept"].values[0])
        q_max_intercept = float(q_max_data["intercept"].values[0])
    
        # Create the regression line points
        x = linspace(np.nanmin(meteo_df['$Q_{mean}$'].values), np.nanmax(meteo_df['$Q_{mean}$'].values), 100)
        y_q_min = q_min_coef * x + q_min_intercept
        y_q_max = q_max_coef * x + q_max_intercept

        # Plot the regression lines
        axes.plot(x, y_q_min, '-', label='$Q_{min} regression line$', color='orange')
        axes.plot(x, y_q_max, '-', label='$Q_{max} regression line$', color='red')

    #axes[0].text(-0.45, .5, label, fontsize=10, horizontalalignment='center',
    #                verticalalignment='center', transform=axes[0].transAxes)
    axes[1].set_xlabel("Normalized $Q_{min,d}$")
    axes[1].set_ylabel("Normalized $Q_{max,d}$")
    axes[0].set_xlabel("Normalized $Q_{mean,d}$")
    axes[0].set_ylabel("Normalized $Q_{min,d}$ and $Q_{max,d}$")

    axes[0].text(0.1, 0.9, "a)", fontsize=15, horizontalalignment='center',
                 verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.1, 0.9, "b)", fontsize=15, horizontalalignment='center',
                 verticalalignment='center', transform=axes[1].transAxes)
    
    axes[0].set_aspect('equal', adjustable='box')
    axes[1].set_aspect('equal', adjustable='box')
    
    axes[0].legend(handles=handles, labels=labels, loc='lower right', frameon=False)
    
    plt.savefig(filename + ".pdf", format="pdf", bbox_inches='tight', dpi=100)

def plot_FDCs_together(observed_15min_FDCs, observed_daily_mean_FDCs, filename):

    # Observed discharge FDC
    FDCs1_df = pd.read_csv(observed_15min_FDCs, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    # Simulated FDC
    FDCs2_df = pd.read_csv(observed_daily_mean_FDCs, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')

    # Select the dates
    date1 = '2010-06-01'
    date2 = '2010-06-07'
    dates1 = FDCs1_df.index.get_level_values(0)
    FDCs1_df = FDCs1_df[(dates1 >= date1) & (dates1 < date2)]
    dates2 = FDCs2_df.index.get_level_values(0)
    FDCs2_df = FDCs2_df[(dates2 >= date1) & (dates2 < date2)]

    # Start the plotting
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(FDCs2_df.index, FDCs2_df.values, '.', label='Observed daily mean flow duration curves', color='teal')
    ax.plot(FDCs1_df.index, FDCs1_df.values, '.', label='Observed 15 min flow duration curves', color='orange')

    plt.xlabel('Days')
    plt.ylabel('Discharge')
    plt.legend(frameon=False)
    ax.xaxis.set_minor_locator(dates.HourLocator())
    ax.xaxis.set_minor_formatter(dates.DateFormatter(''))
    ax.xaxis.set_major_locator(dates.DayLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%d/%m'))
    plt.savefig(filename, format="pdf", bbox_inches='tight', dpi=100)

def plot_sampled_distributions(meteo_df, results, function, filename, pdfs=True):
    ranges = [[-50, 150], [-1, 2], [-3, 1], [-10, 10]]

    # The parameters to plot depending on the function
    if function == "Singh2014":
        column_nb = 3
        labels = ["$a$", "$b$", "$M$"]
    elif function == "Sigmoid_d":
        column_nb = 4
        labels = ["$a$", "$b$", "$c$", "$d$", "$M$"]
    elif function == "Sigmoid":
        column_nb = 3
        labels = ["$a$", "$b$", "$c$", "$M$"]

    # The calibrated parameters
    data_cleaned = []
    for l in labels:
        data_cleaned.append(meteo_df[l].dropna().to_numpy())

    # The parameters sampled from the KDE fitted to the calibrated parameters' distribution
    a_distrib = pd.read_csv(results + "a_distrib.csv", index_col=0)
    b_distrib = pd.read_csv(results + "b_distrib.csv", index_col=0)
    c_distrib = pd.read_csv(results + "c_distrib.csv", index_col=0)
    M_distrib = pd.read_csv(results + "M_distrib.csv", index_col=0)
    data_cleaned2 = []
    data_cleaned2.append(a_distrib.to_numpy())
    data_cleaned2.append(b_distrib.to_numpy())
    data_cleaned2.append(c_distrib.to_numpy())
    data_cleaned2.append(M_distrib.to_numpy())

    # Preparing the subplot setup
    fig, axes = plt.subplots(2, column_nb, figsize=(11, 6))
    if function == "Sigmoid_d" or function == "Singh2014":
        fig.delaxes(axes.flatten()[-1])
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.2)

    # Plotting
    for i, (x, x2, range) in enumerate(zip(data_cleaned, data_cleaned2, ranges)):
        density = False
        if pdfs:
            density = True
        histogram(axes.flatten()[i], x, labels[i], results, density=density, range=range, alpha=0.5, label="Calibrated")
        histogram(axes.flatten()[i], x2, labels[i], results, density=density, range=range, alpha=0.5, label="Sampled")

    if pdfs:
        filename += "_with_pdfs"
        axes[0][0].set_ylabel("Probability density")
        axes[1][0].set_ylabel("Probability density")
    else:
        axes[0][0].set_ylabel("Count")
        axes[1][0].set_ylabel("Count")
    axes[1][0].legend(frameon=False)
    plt.savefig(filename + ".pdf", format="pdf", bbox_inches='tight', dpi=100)

def plot_comparison_of_catchments_distributions(catchments, path, function, months_str, filename, pdfs=True):

    colors = ["mediumseagreen", "darkorchid", "darkorange", "gold", "crimson", "cornflowerblue", "gray"]

    # The parameters to plot depending on the function
    if function == "Singh2014":
        column_nb = 2
        labels = ["$a$", "$b$", "$M$"]
    elif function == "Sigmoid_d":
        column_nb = 3
        labels = ["$a$", "$b$", "$c$", "$d$", "$M$"]
    elif function == "Sigmoid":
        column_nb = 2
        labels = ["$a$", "$b$", "$c$", "$M$"]

    # Preparing the subplot setup
    fig, axes = plt.subplots(2, column_nb, figsize=(7, 6))
    if function == "Sigmoid_d" or function == "Singh2014":
        fig.delaxes(axes.flatten()[-1])
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.2)

    for catchment, color in zip(catchments, colors):

        # Fetch the catchment data
        results = f"{path}Outputs/Arolla/{catchment}/"
        dataframe_filename = f"{results}meteo_df_{function}_{months_str}_constrained.csv"
        meteo_df = pd.read_csv(dataframe_filename, index_col=0)

        ranges = [[-25, 100], [-0.5, 1], [-3, 1], [-7, 7]]

        # The calibrated parameters
        data_cleaned = []
        for l in labels:
            data_cleaned.append(meteo_df[l].dropna().to_numpy())

        # Plotting
        for i, (x, range) in enumerate(zip(data_cleaned, ranges)):
            density = False
            if pdfs:
                density = True
            histogram(axes.flatten()[i], x, labels[i] + " [-]", results, density=density, bins=50,
                      range=range, alpha=0.5, label=catchment, color=color)
            # KDE modelisation
            x_eval = np.linspace(np.nanmin(x) - 1, np.nanmax(x) + 1, 500)
            kde = stats.gaussian_kde(x)
            axes.flatten()[i].plot(x_eval, kde(x_eval), color=color)
            axes.flatten()[i].set_xlim(range[0], range[1])

    axes[0][0].text(0.1, 0.9, "a)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[0][0].transAxes)
    axes[0][1].text(0.1, 0.9, "b)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[0][1].transAxes)
    axes[1][0].text(0.1, 0.9, "c)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[1][0].transAxes)
    axes[1][1].text(0.1, 0.9, "d)", fontsize=15, horizontalalignment='center',
               verticalalignment='center', transform=axes[1][1].transAxes)

    if pdfs:
        filename += "_with_pdfs"
        axes[0][0].set_ylabel("Probability density")
        axes[1][0].set_ylabel("Probability density")
    else:
        axes[0][0].set_ylabel("Count")
        axes[1][0].set_ylabel("Count")
    axes[-1][-1].legend(frameon=False)
    plt.savefig(filename + ".pdf", format="pdf", bbox_inches='tight', dpi=100)

def plot_comparison_of_catchments_distributions_depending_on_weather(catchments, path, function, months_str, filename, pdfs=True):

    colors = ["mediumseagreen", "darkorchid", "darkorange", "gold", "crimson", "cornflowerblue", "gray"]

    # The parameters to plot depending on the function
    if function == "Singh2014":
        column_nb = 3
        labels = ["$a$", "$b$", "$M$"]
    elif function == "Sigmoid_d":
        column_nb = 5
        labels = ["$a$", "$b$", "$c$", "$d$", "$M$"]
    elif function == "Sigmoid":
        column_nb = 4
        labels = ["$a$", "$b$", "$c$", "$M$"]

    # Preparing the subplot setup
    fig, axes = plt.subplots(7, column_nb, figsize=(14, 12))
    if function == "Sigmoid_d" or function == "Singh2014":
        fig.delaxes(axes.flatten()[-1])
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.2)

    for j, catchment in enumerate(catchments):

        # Fetch the catchment data
        results = f"{path}Outputs/Arolla/{catchment}/"
        dataframe_filename = f"{results}meteo_df_{function}_{months_str}_constrained.csv"
        meteo_df = pd.read_csv(dataframe_filename, index_col=0)

        # Categorize according to weather
        meteo_df["Weather"] = 'None'
        meteo_df.loc[(meteo_df["Temperature"] <= 0), "Weather"] = 'Freezing'
        meteo_df.loc[(meteo_df["Temperature"] > 0), "Weather"] = 'Melting'
        meteo_df.loc[(meteo_df["Precipitation"] > 0) & (meteo_df["Temperature"] > 0), "Weather"] = 'Raining'
        meteo_df.loc[(meteo_df["Precipitation"] > 0) & (meteo_df["Temperature"] <= 0), "Weather"] = 'Snowing'
        
        # Count the number of occurrences
        nb_freezing = sum(meteo_df["Weather"] == 'Freezing')
        nb_melting = sum(meteo_df["Weather"] == 'Melting')
        nb_raining = sum(meteo_df["Weather"] == 'Raining')
        nb_snowing = sum(meteo_df["Weather"] == 'Snowing')

        ranges = [[-25, 75], [-0.5, 1], [-3, 1], [-7, 7]]

        # The calibrated parameters
        freezing = []
        melting = []
        raining = []
        snowing = []
        for l in labels:
            freezing.append(meteo_df.loc[(meteo_df["Weather"] == 'Freezing'), l].dropna().to_numpy())
            melting.append(meteo_df.loc[(meteo_df["Weather"] == 'Melting'), l].dropna().to_numpy())
            raining.append(meteo_df.loc[(meteo_df["Weather"] == 'Raining'), l].dropna().to_numpy())
            snowing.append(meteo_df.loc[(meteo_df["Weather"] == 'Snowing'), l].dropna().to_numpy())

        # Plotting
        for i, (f, m, r, s, range) in enumerate(zip(freezing, melting, raining, snowing, ranges)):
            density = False
            if pdfs:
                density = True
            axes[-1][i].set_xlabel(labels[i] + " [-]")
            # KDE modelisation
            x_eval = np.linspace(np.nanmin(range[0]) - 1, np.nanmax(range[1]) + 1, 500)
            f_kde = stats.gaussian_kde(f)
            m_kde = stats.gaussian_kde(m)
            r_kde = stats.gaussian_kde(r)
            s_kde = stats.gaussian_kde(s)
            axes[j][i].plot(x_eval, f_kde(x_eval), color="darkblue")
            axes[j][i].plot(x_eval, m_kde(x_eval), color="crimson")
            axes[j][i].plot(x_eval, r_kde(x_eval), color="gold")
            axes[j][i].plot(x_eval, s_kde(x_eval), color="lightblue")
            axes[j][i].set_xlim(range[0], range[1])
            axes[j][i].set_xlim(range[0], range[1])
            axes[j][i].set_xlim(range[0], range[1])
            axes[j][i].set_xlim(range[0], range[1])
            axes[j][i].fill_between(x_eval, f_kde(x_eval), color="darkblue", alpha=0.5)
            axes[j][i].fill_between(x_eval, m_kde(x_eval), color="crimson", alpha=0.5)
            axes[j][i].fill_between(x_eval, r_kde(x_eval), color="gold", alpha=0.5)
            axes[j][i].fill_between(x_eval, s_kde(x_eval), color="lightblue", alpha=0.5)

        axes[j][0].text(0.05, 0.7, catchment, fontsize=15, horizontalalignment='left',
                   verticalalignment='center', transform=axes[j][0].transAxes)
        
    # Add a figure-wide legend below the plots
    fig.legend(
        labels=[f"Freezing ({nb_freezing} days)", f"Melting ({nb_melting} days)", 
                f"Raining ({nb_raining} days)", f"Snowing ({nb_snowing} days)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        fontsize=12,
    )

    if pdfs:
        filename += "_with_pdfs"
        axes[3][0].set_ylabel("Probability density")
    else:
        axes[3][0].set_ylabel("Count")
    axes[0][-1].legend(frameon=False)
    plt.savefig(filename + ".pdf", format="pdf", bbox_inches='tight', dpi=100)

def plot_coefficients_of_determination(catchments, path, function, months_str, filename):

    fig, axes = plt.subplots(1, 7, figsize=(20, 4))

    for j, catchment in enumerate(catchments):

        # Fetch the catchment data
        results = f"{path}Outputs/Arolla/{catchment}/"

        r21_filename = f"{results}r21_df_{function}_{months_str}.csv"
        r22_filename = f"{results}r22_df_{function}_{months_str}.csv"

        r21_df = pd.read_csv(r21_filename, index_col=0)
        r22_df = pd.read_csv(r22_filename, index_col=0)

        axes[j].plot(r21_df.values,
                  r22_df.values, ".", color="darkblue")
        axes[j].plot(r21_df.loc[(r21_df.values >= 0.95) & (r22_df.values >= 0.95)].values,
                  r22_df.loc[(r21_df.values >= 0.95) & (r22_df.values >= 0.95)].values, ".", color="red")

        nb_good_fits = len(r21_df.loc[(r21_df.values >= 0.95) & (r22_df.values >= 0.95)].values)
        total_nb = len(r21_df.values)

        axes[j].text(0.1, 0.8, catchment, fontsize=11, horizontalalignment='left',
                   verticalalignment='center', transform=axes[j].transAxes)
        axes[j].text(0.1, 0.15, f"Nb of days: {total_nb:.0f}", fontsize=11, horizontalalignment='left',
                   verticalalignment='center', transform=axes[j].transAxes, color="darkblue", label="R² obtained for a day's calibration")
        axes[j].text(0.1, 0.1, fr"R² $\geq$ 0.95: {nb_good_fits/total_nb*100:.0f}%", fontsize=11, horizontalalignment='left',
                   verticalalignment='center', transform=axes[j].transAxes, color="red", label=r"Both R² $\geq$ 0.95")
    axes[2].set_xlabel("R² of the first calibration step")
    axes[0].set_ylabel("R² of the second calibration step")
    axes[-1].legend(frameon=False)

    plt.savefig(filename + ".pdf", format="pdf", bbox_inches='tight', dpi=100)

def plot_yearly_discharges(area_file, catchments, path, filename):

    fig, axes = plt.subplots(1, figsize=(8, 4))
    colors = ['b', 'c', 'r', 'm', 'g', 'k']

    for i, (catchment, color) in enumerate(zip(catchments, colors)):
        discharge_filename = f"{path}Outputs/ObservedDischarges/{area_file}{catchment}.csv"

        # Observed discharge
        df = pd.read_csv(discharge_filename, header=0, na_values='', usecols=[0, 1],
                         index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
        df.rename(columns={df.columns[0]: 'Discharge'}, inplace=True)

        # Compute discharges
        yearly_df = df.groupby(pd.Grouper(freq='Y')).sum()
        five_year_moving_window = yearly_df.rolling(window=5, center=True, closed='both').mean()
        norm_yearly_df = yearly_df / 70000
        norm_five_year_moving_window = five_year_moving_window / 70000

        # Plot
        axes.plot(norm_yearly_df.index, norm_yearly_df.values, color=color, label=catchment)
        if i == 5:
            label = "5-year moving window"
        else:
            label = None
        axes.plot(norm_five_year_moving_window.index, norm_five_year_moving_window.values, color=color, label=label, alpha=0.5, linewidth=4)

    axes.set_xlabel("Time")
    axes.set_ylabel("Normalized yearly discharge")
    axes.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.savefig(filename + ".pdf", format="pdf", bbox_inches='tight', dpi=100)

def plot_downscaling_improvement(FDCs, 
                                 FDCs1, label1, metric1,
                                 FDCs2, label2, metric2,
                                 FDCs3, label3, metric3,
                                 FDCs4, label4, metric4,
                                 FDCs5, metric5, downscaling_metrics, 
                                 catchment, filename):

    # Loading FDCs
    FDCs_ref_df = pd.read_csv(FDCs, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    FDCs1_df = pd.read_csv(FDCs1, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    FDCs2_df = pd.read_csv(FDCs2, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    FDCs3_df = pd.read_csv(FDCs3, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    FDCs4_df = pd.read_csv(FDCs4, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    if FDCs5:
        FDCs5_df = pd.read_csv(FDCs5, index_col=0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    metrics_df = pd.read_csv(downscaling_metrics, index_col=1)
    
    #print(FDCs4, FDCs4_df)
    #print(bou)

    # Select the dates
    date1 = '2011-07-01'
    date2 = '2011-07-07'
    dates_ref = FDCs_ref_df.index.get_level_values(0)
    dates1 = FDCs1_df.index.get_level_values(0)
    dates2 = FDCs2_df.index.get_level_values(0)
    dates3 = FDCs3_df.index.get_level_values(0)
    dates4 = FDCs4_df.index.get_level_values(0)
    if FDCs5:
        dates5 = FDCs5_df.index.get_level_values(0)
    FDCs_ref_df = FDCs_ref_df[(dates_ref >= date1) & (dates_ref < date2)]
    FDCs1_df = FDCs1_df[(dates1 >= date1) & (dates1 < date2)]
    FDCs2_df = FDCs2_df[(dates2 >= date1) & (dates2 < date2)]
    FDCs3_df = FDCs3_df[(dates3 >= date1) & (dates3 < date2)]
    FDCs4_df = FDCs4_df[(dates4 >= date1) & (dates4 < date2)]
    if FDCs5:
        FDCs5_df = FDCs5_df[(dates5 >= date1) & (dates5 < date2)]

    if FDCs5:
        nb_lines = 5
    else:
        nb_lines = 4
        
    # Normalize
    if FDCs5:
        max = np.nanmax(pd.concat([FDCs1_df, FDCs2_df, FDCs3_df, FDCs4_df, FDCs5_df]))
    else:
        max = np.nanmax(pd.concat([FDCs1_df, FDCs2_df, FDCs3_df, FDCs4_df]))

    # Start the plotting
    fig, axes = plt.subplots(nb_lines, 1, figsize=(10, 7), sharex=True)
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.3)
    line1, = axes[0].plot(FDCs_ref_df.index, FDCs_ref_df.values / max, '.', color='orange')
    axes[1].plot(FDCs_ref_df.index, FDCs_ref_df.values / max, '.', color='orange')
    axes[2].plot(FDCs_ref_df.index, FDCs_ref_df.values / max, '.', color='orange')
    if FDCs5:
        axes[3].plot(FDCs_ref_df.index, FDCs_ref_df.values / max, '.', color='orange')
    axes[-1].plot(FDCs_ref_df.index, FDCs_ref_df.values / max, '.', color='orange', zorder=10)

    line2, = axes[0].plot(FDCs1_df.index, FDCs1_df.values / max, '.', color='teal') # label=label1
    axes[1].plot(FDCs2_df.index, FDCs2_df.values / max, '.', color='teal') # label=label2
    axes[2].plot(FDCs3_df.index, FDCs3_df.values / max, '.', color='teal') # label=label3
    if FDCs5:
        axes[3].plot(FDCs4_df.index, FDCs4_df.values / max, '.', color='teal') # label=label4
        bootstrap = FDCs5_df
    else:
        bootstrap = FDCs4_df
    
    for i in range(99):
        axes[-1].plot(bootstrap.index, bootstrap[str(i)].values / max, '.', color='black', markeredgewidth=0.0, alpha=0.2)
    line3, = axes[-1].plot(bootstrap.index, bootstrap[str(99)].values / max, '.', color='black', markeredgewidth=0.0, alpha=0.2)

    axes[0].text(0.99, 0.75, f"nRMSE_R: {metrics_df[metric1][2]:.2f}\nnRMSE_IQR: {metrics_df[metric1][3]:.2f}\n"
                 f"nDist_R: {metrics_df[metric1][5]:.2f}", fontsize=10, horizontalalignment='right',
                 verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.99, 0.75, f"nRMSE_R: {metrics_df[metric2][2]:.2f}\nnRMSE_IQR: {metrics_df[metric2][3]:.2f}\n"
                 f"nDist_R: {metrics_df[metric2][5]:.2f}", fontsize=10, horizontalalignment='right',
                 verticalalignment='center', transform=axes[1].transAxes)
    axes[2].text(0.99, 0.75, f"nRMSE_R: {metrics_df[metric3][2]:.2f}\nnRMSE_IQR: {metrics_df[metric3][3]:.2f}\n"
                 f"nDist_R: {metrics_df[metric3][5]:.2f}", fontsize=10, horizontalalignment='right',
                 verticalalignment='center', transform=axes[2].transAxes)
    axes[3].text(0.99, 0.75, f"nRMSE_R: {metrics_df[metric4][2]:.2f}\nnRMSE_IQR: {metrics_df[metric4][3]:.2f}\n"
                 f"nDist_R: {metrics_df[metric4][5]:.2f}", fontsize=10, horizontalalignment='right',
                 verticalalignment='center', transform=axes[3].transAxes)
    if FDCs5:
        axes[4].text(0.99, 0.75, f"nRMSE_R: {metrics_df[metric5][2]:.2f}\nnRMSE_IQR: {metrics_df[metric5][3]:.2f}\n"
                 f"nDist_R: {metrics_df[metric5][5]:.2f}", fontsize=10, horizontalalignment='right',
                     verticalalignment='center', transform=axes[4].transAxes)
        
        
    handles, labels = axes[-1].get_legend_handles_labels()
    handles.extend([line1, line3, line2])
    labels.extend(["Empirical FDCs",  "Bootstrapped empirical FDCs", "Simulated FDCs"])
    axes[-1].legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(-0.05, -0.4), frameon=False, ncols=3)
    
    axes[0].text(0.03, 0.9, "a)", fontsize=10, horizontalalignment='center',
                 verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.03, 0.9, "b)", fontsize=10, horizontalalignment='center',
                 verticalalignment='center', transform=axes[1].transAxes)
    axes[2].text(0.03, 0.9, "c)", fontsize=10, horizontalalignment='center',
                 verticalalignment='center', transform=axes[2].transAxes)
    axes[3].text(0.03, 0.9, "d)", fontsize=10, horizontalalignment='center',
                 verticalalignment='center', transform=axes[3].transAxes)
    if FDCs5:
        axes[4].text(0.03, 0.9, "e)", fontsize=10, horizontalalignment='center',
                     verticalalignment='center', transform=axes[4].transAxes)

    plt.xlabel('Days')
    axes[2].set_ylabel('Normalized discharge')
    axes[-1].xaxis.set_minor_locator(dates.HourLocator())
    axes[-1].xaxis.set_minor_formatter(dates.DateFormatter(''))
    axes[-1].xaxis.set_major_locator(dates.DayLocator())
    axes[-1].xaxis.set_major_formatter(dates.DateFormatter('%d/%m/%y'))
    plt.savefig(filename, format="png", bbox_inches='tight', dpi=100)

def Mutzner2015_plot(subdaily_discharge, months, filename):
    # Observed discharge
    df = pd.read_csv(subdaily_discharge, header=0, na_values='', usecols=[0, 1],
                     index_col = 0, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S')
    df.rename(columns={df.columns[0]: 'Discharge'}, inplace=True)
    df = select_months(df, months)

    # Compute 24h moving window
    daily_moving_window = df.rolling(window=96, center=True, closed='both').mean()

    # Compute the "Detrended discharge" as defined in Mutzner 2015
    detrended_df = df - daily_moving_window

    # Extract all the years, months and days covered
    yearly_df = df.groupby(pd.Grouper(freq='Y')).last()
    years = yearly_df.index.year
    months = np.unique(detrended_df.index.month)
    days = np.unique(detrended_df.index.day)

    # Start the plotting
    nb_col = 6
    nb_row = 7
    fig, axes = plt.subplots(nb_row, nb_col, figsize=(10, 10))
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.3)

    # Set up the color map (viridis in this example)
    cmap = cm.get_cmap('viridis')
    normalize = mcolors.Normalize(vmin=152, vmax=273)
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.5])
    ColorbarBase(cax, cmap=cmap, norm=normalize)
    cax.set_ylabel('Day of year')
    cax.grid(axis='y')

    j = 0
    for i, yr in enumerate(years):
        #if i < nb_col * nb_row:
        if yr >= 1977 and yr < 2020:
            print(yr)
            yr_detrended_df = select_years(detrended_df, [yr])
            for m in months:
                month_detrended_df = select_months(yr_detrended_df, [m])
                for d in days:
                    day_detrended_df = select_days(month_detrended_df, [d])
                    day_dd_df = day_detrended_df.copy()
                    day_detrended_df.index = day_detrended_df.index - pd.tseries.offsets.MonthBegin()
                    day_detrended_df.index = day_detrended_df.index - pd.tseries.offsets.YearBegin()
                    maximum = day_detrended_df.max().values[0]
                    hour = day_detrended_df.loc[(day_detrended_df["Discharge"] == maximum), "Discharge"].index#.strftime('%H:%M:%S')
                    if not day_detrended_df.empty:
                        color = cmap(normalize(day_dd_df.index.dayofyear[0]))  # Scale the color based on day index
                        #axes.flatten()[j].plot(day_detrended_df.index.strftime('%H:%M:%S'), day_detrended_df.values,
                        #                       color=color, linewidth=0.4)
                        print(hour, maximum)
                        if len(hour) != 1:
                            print("More than 1 time with this maximum, discard.")
                            continue
                        axes.flatten()[j].plot(hour, maximum, 'o', color=color, linewidth=0.4, markersize=1)
            axes.flatten()[j].text(.1, .8, yr, fontsize=8, horizontalalignment='center',
                                   verticalalignment='center', transform=axes.flatten()[j].transAxes)
            axes.flatten()[j].set_ylim([0, 1])

            axes.flatten()[j].xaxis.set_major_formatter(dates.DateFormatter('%H h'))
            axes.flatten()[j].xaxis.set_major_locator(dates.HourLocator(byhour=[0, 12, 24]))
            j += 1
    axes[1][0].set_ylabel('Detrended discharge (Mutzner et al., 2015)')
    axes[-1][1].set_xlabel('Time during day (in hours)')
    plt.savefig(filename, format="png", bbox_inches='tight', dpi=100)

    # Start the plotting
    fig, axes = plt.subplots(nb_row, nb_col, figsize=(10, 10), sharex=True)
    [ax.spines['right'].set_visible(False) for ax in axes.flatten()]
    [ax.spines['top'].set_visible(False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.3)

    # Set up the color map (viridis in this example)
    cmap = cm.get_cmap('viridis')
    normalize = mcolors.Normalize(vmin=152, vmax=273)
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.5])
    ColorbarBase(cax, cmap=cmap, norm=normalize)
    cax.set_ylabel('Day of year')
    cax.grid(axis='y')

    j = 0
    for i, yr in enumerate(years):
        #if i < nb_col * nb_row:
        if yr >= 1977 and yr < 1997:
            print(yr)
            yr_daily_moving_window_df = select_years(daily_moving_window, [yr])
            for m in months:
                month_daily_moving_window_df = select_months(yr_daily_moving_window_df, [m])
                for d in days:
                    day_daily_moving_window_df = select_days(month_daily_moving_window_df, [d])
                    if not day_daily_moving_window_df.empty:
                        color = cmap(normalize(day_daily_moving_window_df.index.dayofyear[0]))  # Scale the color based on day index
                        axes.flatten()[j].plot(day_daily_moving_window_df.index.strftime('%H:%M:%S'), day_daily_moving_window_df.values,
                                               color=color, linewidth=0.4)
            axes.flatten()[j].text(.80, .1, yr, fontsize=8, horizontalalignment='center',
                                   verticalalignment='center', transform=axes.flatten()[j].transAxes)
            #axes.flatten()[j].set_ylim([-2.5, 4])
            j += 1

    axes[0][-1].xaxis.set_major_locator(dates.HourLocator(interval=96*6))
    axes[0][-1].xaxis.set_major_formatter(ticker.FixedFormatter(['0', '4', '8', '12', '16', '20', '24']))
    axes[1][0].set_ylabel('Trend of the detrended discharge (Mutzner et al., 2015)')
    axes[-1][1].set_xlabel('Time during day (in hours)')
    plt.savefig(filename + '_remainder.pdf', format="pdf", bbox_inches='tight', dpi=100)




