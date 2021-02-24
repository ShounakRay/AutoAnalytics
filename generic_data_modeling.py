# @Author: Shounak Ray <Ray>
# @Date:   16-Oct-2020 13:10:14:142  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: generic_data_modeling.py
# @Last modified by:   Ray
# @Last modified time: 24-Feb-2021 00:02:20:209  GMT-0700
# @License: [Private IP]


from collections import Counter
from inspect import currentframe, getframeinfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.signal
import scipy.stats
import sklearn
from distfit import distfit
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from stringcase import snakecase

# convert from normal text to snake case for all columns in df


def util_snakify_cols(df):
    df.columns = [snakecase(col).replace('__', '_') for col in df.columns]
    return df

# Normalize list


def util_normalize(list):
    list = (list - np.min(list)) / (np.max(list) - np.min(list))
    return list

# Ensure continuity of list of tuples based on tuple[0]


def util_fill_lot(lot):
    X_zipl_func = lot
    X_zipl = X_zipl_func[:2]
    for tup_i in range(len(X_zipl_func) - 1):
        curr_tup = X_zipl_func[tup_i]
        next_tup = X_zipl_func[tup_i + 1]
        if(next_tup[0] > curr_tup[0] + 1):
            X_zipl.append((curr_tup[0] + 1, None))
        X_zipl.append(next_tup)

    X, Y = np.array(list(zip(*X_zipl)))
    return X, Y

# get skew type string


def util_skew_class(skew_score_func):
    if(skew_score_func == 0):
        skew_type_func = 'Perfect Normal'
    elif(skew_score_func > 0):
        skew_type_func = 'Right-Skewed'
    elif(skew_score_func < 0):
        skew_type_func = 'Left-Skewed'
    else:
        skew_type_func = 'ERROR'
    return skew_type_func

# returns smoothened list


def util_smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Returns best distribution and information associated
# Possible bug: resolution may cause extra index (so lens don't match)


def generate_dist_fits(df, pick, peak_max=3, smooth_index=5, top_dists=3, trim_lower='None', trim_upper='None'):
    # Data Sanitation
    if(peak_max == 0):
        raise ValueError('XXXX peak_max argument is equal to 0, a non-permissible value XXXX')
    if(smooth_index == 0):
        raise ValueError('XXXX smooth_index argument is equal to 0, a non-permissible value XXXX')
    if(top_dists == 0):
        raise ValueError('XXXX top_dists argument is equal to 0, a non-permissible value XXXX')
    top_dists = int(top_dists)
    smooth_index = int(smooth_index)
    peak_max = int(peak_max)
    df = util_snakify_cols(df)
    pick = snakecase(pick).replace('__', '_')
    if(pick not in df.columns):
        raise ValueError('XXXX pick argument is NOT a feature inside the inputted dataset XXXX')

    HYPER_ASSOC = {'distr': 'Distribution', 'RSS': 'Root Sum Squared', 'LLE': 'Locally Linear Embedding',
                   'loc': 'LOC: Mean', 'scale': 'SCALE: Standard Deviation', 'arg': 'Argument(s)'}
    TAG_NAME_ASSOC = {'beta': 'Beta', 'genextreme': 'Generalized Extreme Value', 't': 'Student’s t',
                      'norm': 'Normal', 'lognorm': 'Lognormal ', 'gamma': 'Gamma', 'dweibull': 'Double Weibull',
                      'expon': 'Exponential', 'uniform': 'Uniform', 'pareto': 'Pareto'}
    MODEL_PARAM_ASSOC = {'beta': ['a', 'b'], 'genextreme': ['c'], 't': ['° of freedom'], 'lognorm': ['s'],
                         'gamma': ['a'], 'dweibull': ['c'], 'pareto': ['b']}
    NAME_OBJECT_ASSOC = {'beta': scipy.stats.beta, 'genextreme': scipy.stats.genextreme, 't': scipy.stats.t,
                         'norm': scipy.stats.norm, 'lognorm': scipy.stats.lognorm, 'gamma': scipy.stats.gamma,
                         'dweibull': scipy.stats.dweibull, 'expon': scipy.stats.expon,
                         'uniform': scipy.stats.uniform, 'pareto': scipy.stats.pareto}
    DEC_ROUND_NUM = 3

    # getattr(scipy.stats, dir(scipy.stats)[dir(scipy.stats).index('genextreme')])

    # Distribution, Arguments,
    curve_df = pd.DataFrame()

    if(trim_lower != 'None'):
        df = df[(df[pick] > trim_lower)]
    if(trim_upper != 'None'):
        df = df[(df[pick] < trim_upper)]

    if(len(df) == 0):
        raise ValueError('XXXX Trimmed dataset, as per inputted bounds, is empty XXXX')

    df = df.sort_values(pick).reset_index().drop('index', 1)

    X = np.array(list(df[pick]))
    # X = X[(np.abs(X) > trim_lower) & (np.abs(X) < trim_upper)]
    max = np.max(X)
    min = np.min(X)
    # For exisiting univariate data
    counts = sorted(list(Counter(X).items()), key=lambda item: item[0])
    X_x = np.array([tup[0] for tup in counts])
    X_y = np.array([tup[1] for tup in counts])

    X_x_ORIGIN = X_x
    X_y_ORIGIN = X_y
    X_y = util_normalize(X_y)

    X_x, X_y = util_fill_lot(list(zip(X_x, X_y)))

    skew_score = round(scipy.stats.skew(list(filter(None.__ne__, X_y))), DEC_ROUND_NUM)
    skew_type = util_skew_class(skew_score)

    X_y_temp = X_y_ORIGIN
    peaks = [X_y_temp[i] for i in scipy.signal.find_peaks(X_y_temp)[0]]
    box = 3
    while(len(peaks) > peak_max):
        X_y_temp = util_smooth(X_y_temp, box)
        box += 5
        peaks = [(i, X_y_temp[i]) for i in scipy.signal.find_peaks(X_y_temp)[0]]
    peak = sorted(peaks, key=lambda item: item[1])[0]
    resolution = (max - min) / (len(X_x) - 1)

    dist = distfit(alpha=0.05, smooth=smooth_index)
    dist.fit_transform(X)

    top_models = dist.summary[:top_dists]
    top_models = top_models.rename(columns=HYPER_ASSOC)

    for row_i in range(len(top_models)):
        dist_OBJECT = NAME_OBJECT_ASSOC[top_models['Distribution'][row_i]]
        loc_val = top_models['LOC: Mean'][row_i]
        scale_val = top_models['SCALE: Standard Deviation'][row_i]
        name_val = top_models['Distribution'][row_i]
        params = top_models['Argument(s)'][row_i]
        params_val = ''
        for val_i in range(len(params)):
            if(val_i == len(params) - 1 - 1):
                params_val += MODEL_PARAM_ASSOC[name_val][val_i] + \
                    ': ' + str(round(params[val_i], DEC_ROUND_NUM)) + '; '
            else:
                params_val += MODEL_PARAM_ASSOC[name_val][val_i] + ': ' + str(round(params[val_i], DEC_ROUND_NUM))

        # B_dist_name, *B_p_value, B_params = util_get_best_distribution(list(df['age']))

        # For curve-fit plot
        x_top_dist = np.arange(min, max + resolution, resolution)
        y_top_dist = dist_OBJECT.pdf(x_top_dist, (*params), scale=scale_val, loc=loc_val)
        y_top_dist = util_normalize(y_top_dist)
        # Store information about models in DataFrame
        new_col_names = ['model' + "_X", 'model' + "_Y", 'model_id', 'model_params']
        new_df = pd.DataFrame(zip(x_top_dist, y_top_dist,
                                  [name_val] * len(y_top_dist),
                                  [params_val] * len(y_top_dist)),
                              columns=new_col_names)
        curve_df = pd.concat([curve_df, new_df], axis=0).reset_index().drop('index', 1)

    curve_df['model_id'] = curve_df['model_id'].replace(TAG_NAME_ASSOC)
    pick_df = pd.DataFrame(zip(X_x, X_y))
    pick_df.columns = ['pick_X', 'pick_Y']
    pick_df['pick_skew_type'] = skew_type
    pick_df['pick_skew_score'] = skew_score
    pick_df['pick_top_peak_value'] = round(peak[1], DEC_ROUND_NUM)
    pick_df['pick_top_peak_value_X'] = round(peak[0], DEC_ROUND_NUM)
    pick_df['pick_name'] = pick

    # plt.figure()
    # a = plt.plot(x_top_dist, y_top_dist)
    #b = plt.plot(X_x, X_y)
    # plt.show()
    # plt.close()

    base = pick_df
    for iter in range(top_dists - 1):
        pick_df = pd.concat([pick_df, base]).reset_index().drop('index', 1)
    final_df = pd.concat([pick_df, curve_df], axis=1)

    return final_df

# bool_execution = ''
# filter = 'job: management, technician; marital: single, married'
# for feature_filters in [i.strip() for i in filter.split(';')]:
#     feature_specs = [i.strip() for i in feature_filters.split(':')]
#     feature = feature_specs[0].strip()
#     options = [i.strip() for i in feature_specs[1].split(',')]
#     if(len(options) == 1):
#         bool_execution = ['df[\'' + feature + '\'] == \'' + feature[0] + '\'']
#     else:
#         for opt_i in range(len(options)):
#             store = ['df[\'' + feature + '\'] == \'' + options[opt_i] + '\'']
#         for item_i in range(len(store) - 1):
#             if(item_i == len(store) - 2):
#                 bool_execution += store[item_i]
#             else:
#                 bool_execution += store[item_i] + ' & '
#             a = str(bool_execution)
#     break
#
# df[feature].unique()


# Linear regression
# def linear_regression(df, x_ft, y_ft):
#     df = util_snakify_cols(df)
#     x_ft = snakecase(x_ft).replace('__', '_')
#     y_ft = snakecase(y_ft).replace('__', '_')
#
#     if(x_ft not in df.columns):
#         raise ValueError('XXXX x_ft not in data XXXX')
#     if(y_ft not in df.columns):
#         raise ValueError('XXXX y_ft not in data XXXX')
#
#     x_data = data[x_ft]
#     y_data = data[y_ft]
#     reg = LinearRegression().fit(x_data, y_data)
#
#     r2 = reg.score(x_data, y_data)
#     reg_coef = reg.coef_

# df = pd.read_csv('/Users/Ray/Documents/Python/5 - Webber/Datasets/Bike Sharing/day.csv')
# df.columns
# final = generate_dist_fits(df, 'temp')
# df['population'].unique()
# feature = 'age'
#
#
# X = np.array(list(df[feature]))
# counts = sorted(list(Counter(X).items()), key = lambda item: item[0])
# X_x = np.array([tup[0] for tup in counts])
# X_y = np.array([tup[1] for tup in counts])
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c
# popt, pcov = scipy.optimize.curve_fit(func, X_x, X_y)
# plt.plot(X_x, func(X_x, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#
# # stat_metrics(df, 'age')
# d = generate_dist_fits(df, feature, top_dists = 2, trim_lower = '0', trim_upper = '400')
# # d.to_html('all.html')
# plt.close()
# plt.plot(d['pick_X'], d['pick_Y'])
# plt.plot(d['model_X'], d['model_Y'])


#
