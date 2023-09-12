import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

states = ['New South Wales', 'Victoria', 'Queensland', 'South Australia', 'Western Australia', 'Tasmania',
          'Northern Territory', 'Australian Capital Territory']
states_abb = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS',
              'NT', 'ACT']

# Specify the Excel file path
total_consumption_path = 'total.xlsx'

# Read the Excel file into a pandas DataFrame
total_consumption = pd.read_excel(total_consumption_path, engine='openpyxl')
total_consumption = pd.DataFrame(total_consumption)

print(total_consumption.shape)
print(total_consumption.head())


def adf_test(series, title):
    print("---------------------------------------")
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')  # .dropna() handles differenced data
    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pd.Series(result[0:4], index=labels)
    for key, val in result[4].items():
        out[f'critical value ({key})'] = val
    print(out.to_string())  # .to_string() removes the line "dtype: float64"
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
        print("---------------------------------------")
        return True
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        print("---------------------------------------")
        return False


for state in states_abb:
    time = 0
    while adf_test(total_consumption[state],state ) is False:
        time += 1
        total_consumption = total_consumption.diff().tail(-1)
        print('diff for ' + state + 'time ' )
        print(time)
        print(total_consumption)








# adjacency matrix (1 if adjacent, 0 if not)
adjacency_matrix = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0],  # New South Wales
    [1, 0, 1, 0, 0, 0, 0, 0],  # Victoria
    [1, 1, 0, 1, 1, 0, 0, 0],  # Queensland
    [0, 0, 1, 0, 1, 0, 0, 0],  # South Australia
    [0, 0, 1, 1, 0, 0, 0, 0],  # Western Australia
    [0, 0, 0, 0, 0, 0, 1, 0],  # Tasmania
    [0, 0, 0, 0, 0, 1, 0, 1],  # Northern Territory
    [0, 0, 0, 0, 0, 0, 1, 0]  # Australian Capital Territory
])

standardized_weights = np.array([
    [0, 0.5, 0.5, 0, 0, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0, 0, 0],
    [0.16666667, 0.16666667, 0, 0.16666667, 0.16666667, 0, 0, 0],
    [0, 0, 0.33333333, 0, 0.33333333, 0, 0, 0],
    [0, 0, 0.33333333, 0.33333333, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1., 0],
    [0, 0, 0, 0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0, 0, 0, 1., 0]
])

# spatial_weights = validate_positive_definitive(spatial_weights)
#
#
# # Specify the Excel file path
# total_consumption_path = 'total.xlsx'
# employment_rate_path = 'employment_rate.xlsx'
#
# # Read the Excel file into a pandas DataFrame
# total_consumption = pd.read_excel(total_consumption_path, engine='openpyxl')
#
# # Convert the DataFrame to a numpy matrix
# total_consumption = total_consumption.iloc[:, 1:].to_numpy()
#
# # population = np.random.randint(10000, 100000, size=(n_obs, n_cities))
# # print('population')
#
# # SVAR modeling
# model = sm.tsa.VARMAX(total_consumption, order=(1, 0), exog=spatial_weights, enforce_stationarity=False)
#
# results = model.fit()
# #
# # print(results.summary())










