import numpy as np


def ROLLING_MAX_genie(indicators_data, data_lookback_window):
    """rolling maximum with a given configuration"""
    under_length = (indicators_data.shape[0] + data_lookback_window - 1) % data_lookback_window
    above_length = indicators_data.shape[0] - under_length
    max_indices = np.argmax(indicators_data, axis=-1)
    return indicators_data[max_indices[under_length: above_length]].reshape((-1, data_lookback_window))


import numba as nb


@nb.jit(nopython=True)
def rolling_max(prices, period=100):
    """Return rolling maximum of given values, using the given period window."""
    max_values = np.copy(prices)
    for i in range(period, len(prices)):
        max_values[i] = max(prices[i - period:i])
    return max_values


def ROLLING_MIN_genie(indicators_data, data_lookback_window):
    """rolling minimum with a given configuration"""
    under_length = (indicators_data.shape[0] + data_lookback_window - 1) % data_lookback_window
    above_length = indicators_data.shape[0] - under_length
    return np.minimum.reduceat(indicators_data, range(under_length, indicators_data.shape[0], data_lookback_window),
                               axis=-1)[
           :above_length].reshape((-1, data_lookback_window))


def BARSINCE_genie(expression):
    """Bars Since"""
    return np.sum(expression.astype(bool), axis=1)

# Code any python dependensies, funcitions, variables,etc..., to program a meta . Make sure it is all correct and that its speed is optimal. change anything as needed. add docstrings and comments as needed. Dont forget to indent as per PEP 8 standards. Make sure to print to screen what you are doing.


# import dependencies
import pandas as pd


# define function
def filter_columns(df, conditions):
    """
    This function filters out columns of dataframes if they do not meet the conditions passed as a human-readable string.
    """

    # print what we are doing
    print("Filtering out columns of dataframe that do not meet the conditions passed as a human-readable string...")

    # split conditions into a list
    conditions_list = conditions.split()

    # initialize empty list
    filtered_columns = []

    # loop through conditions
    for condition in conditions_list:

        # split condition into operator and value
        operator, value = condition.split("=")

        # check operator
        if operator == ">":

            # filter columns
            filtered_columns.append(df.columns[df.columns > value])

        elif operator == "<":

            # filter columns
            filtered_columns.append(df.columns[df.columns < value])

        elif operator == ">=":

            # filter columns
            filtered_columns.append(df.columns[df.columns >= value])

        elif operator == "<=":

            # filter columns
            filtered_columns.append(df.columns[df.columns <= value])

        elif operator == "==":

            # filter columns
            filtered_columns.append(df.columns[df.columns == value])

        elif operator == "!=":

            # filter columns
            filtered_columns.append(df.columns[df.columns != value])

        else:

            # print error
            print("Error: Invalid operator.")

            # break loop
            break

    # concatenate list of filtered columns
    filtered_columns = pd.concat(filtered_columns)

    # drop duplicate columns
    filtered_columns = filtered_columns.drop_duplicates()

    # print number of columns filtered
    print("Number of columns filtered: {}".format(len(filtered_columns)))

    # return filtered columns
    return filtered_columns


# Code in python 3.9 a python module to filter out Data given in data frames arbitrarily if conditions are not met. The module must handle as input natural English language commands as strings to filter the dataframe accordingly, so feel free to use any tools or libraries at your disposal. Add docstrings and comments as needed. Don't forget to indent as per PEP 8 standards. Code any required functions