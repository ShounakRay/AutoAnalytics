# @Author: Shounak Ray <Ray>
# @Date:   28-Sep-2020 10:09:94:948  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: generic_data_prep.py
# @Last modified by:   Ray
# @Last modified time: 05-Mar-2021 17:03:99:998  GMT-0700
# @License: [Private IP]


from collections import Counter
from inspect import currentframe, getframeinfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import scipy.stats
from distfit import distfit
from stringcase import snakecase

# Return cleaned DataFrame


def papa_johns(df):
    df = pd.DataFrame([1, 2, 3, 4]) * 1
    return df


def reset_df_index(df):
    return df.reset_index().drop('index', 1)

# Return cleaned DataFrame


def prune(df, insert, nan=''):
    df = util_snakify_cols(df)
    df = strict_exclude_nan(df, nan=nan)
    if(insert == 'True'):
        df = util_insert_ID(df)
        return df
    elif(insert == 'False'):
        return df
    else:
        return 'Incorrect value for "insert" argument.'

# Add id/index columns to df


def util_insert_ID(df):
    df = df.reset_index(0).rename(columns={'index': 'id'})
    return df

# Replace all values with certain char (such as ?) with NaN
# Delete all rows that have any NaN values
# Delete all columns that only have NaN values


def strict_exclude_nan(df, nan=''):
    df = util_snakify_cols(df)

    df = util_clean_values(df)
    df.replace(to_replace=nan, value=np.NaN, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    return df

# Remove unnecesary spaces from column names and df content


def util_clean_values(df):
    df.columns = [df.columns[i].strip() for i in range(len(df.columns))]
    df = df.apply(np.vectorize(util_strip))
    # df = df.applymap(lambda x: x.str.strip() if x.dtype == "object" else x)
    # df = df.apply(lambda x: x.astype(str).str.strip())
    return df

# Utility function, return stripped string


def util_strip(x):
    if(isinstance(x, str)):
        return x.strip()
    else:
        return x

# Utility function, replaces content of df given list of list and previous column and index names


def util_replace_df(lol, before):
    final = pd.DataFrame(lol)
    final.columns = before.columns
    final.index = before.index
    return final

# Utility function, trims dict based on tuple of keys


def util_trim_dict(dict, keys):
    final = {k: dict[k] for k in keys}
    return final

# extract top 1-5 values from dict and insert into quant df


def util_modify_tops(type_info, _MAX, all_info):
    orig_col_count = type_info.shape[1]
    for i in range(_MAX):
        type_info['Top ' + str(i + 1)] = [1] * type_info.shape[0]
    rel_type = util_trim_dict(all_info, tuple(type_info.index))
    new_content_type = [list(type_info.values[i][:orig_col_count]) + list(rel_type[type_info.index[i]])
                        for i in range(len(type_info.index))]
    type_info = util_replace_df(new_content_type, type_info)

    return type_info

# BACKUPS: for getting best distribution


def BACKUP_get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for " + dist_name + " = " + str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: " + str(best_dist))
    print("Best p value: " + str(best_p))
    print("Parameters for the best fit: " + str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


def BACKUP_util_get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for " + dist_name + " = " + str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: " + str(best_dist))
    print("Best p value: " + str(best_p))
    print("Parameters for the best fit: " + str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

# outputs dataframe with frequency distribution of specific/random feature


def freq_dist(df, pick):
    df = util_snakify_cols(df)

    pick = snakecase(pick).replace('__', '_')

    if(pick == 'none'):
        index = np.random.choice(range(len(df.columns)))
    else:
        index = df.columns.get_loc(pick)

    if(isinstance(df[df.columns[index]][0], str)):  # order by count/value
        total_rows = float(df.shape[0])
        freq_tracker_pct = {str(k): round((v / total_rows) * 100.0, 3)
                            for k, v in sorted(dict(Counter(df[df.columns[index]])).items(),
                                               key=lambda item: item[1], reverse=False)}
        freq_tracker_count = {str(k): v for k, v in sorted(
            dict(Counter(df[df.columns[index]])).items(), key=lambda item: item[1], reverse=False)}
    else:  # order by number/key
        total_rows = float(df.shape[0])
        freq_tracker_pct = {str(k): round((v / total_rows) * 100.0, 3)
                            for k, v in sorted(dict(Counter(df[df.columns[index]])).items(),
                                               key=lambda item: item[0], reverse=False)}
        freq_tracker_count = {str(k): v for k, v in sorted(
            dict(Counter(df[df.columns[index]])).items(), key=lambda item: item[0], reverse=False)}

    df_freq = pd.DataFrame(freq_tracker_count.items())
    df_freq[2] = freq_tracker_pct.values()

    df_freq['pick'] = df.columns[index]
    df_freq.columns = ['feature', 'count', 'proportion', 'pick']

    return df_freq

# Returns various statistics KPI


def stat_metrics(df, pick, detailed=False):
    df = util_snakify_cols(df)
    pick = snakecase(pick).replace('__', '_')
    all_columns = ['mean', 'mode', 'mode_freq', 'stdev', 'minimum', 'first_Q', 'median', 'third_Q',
                   'maximum', 'variance', 'range', 'iqr', 'hmean', 'kurtosis', 'skew', 'sem', 'variation',
                   'trim_mean', 'median_abs_deviation', 'bayes_mean_lower', 'bayes_mean_upper', 'bayes_variance_lower',
                   'bayes_variance_upper', 'bayes_std_lower', 'bayes_std_upper']
    # assume that pick feature points to quantitative feature
    data = df[pick]
    mean = round(np.mean(data), 3)
    mode = list(scipy.stats.mode(data))[0][0]
    mode_freq = list(scipy.stats.mode(data))[1][0]
    stdev = round(np.std(data), 3)
    minimum = round(np.min(data), 3)
    first_Q = round(np.percentile(data, 25), 3)
    median = round(np.median(data), 3)
    third_Q = round(np.percentile(data, 75), 3)
    maximum = round(np.max(data), 3)
    variance = round(stdev * stdev, 3)
    range = round(maximum - minimum, 3)
    iqr = round(third_Q - first_Q, 3)
    if(detailed):
        hmean = scipy.stats.hmean(data)
        kurtosis = scipy.stats.kurtosis(data)
        skew = scipy.stats.skew(data)
        sem = scipy.stats.sem(data)
        variation = scipy.stats.variation(data)
        trim_mean = scipy.stats.trim_mean(data, proportiontocut=0.1)
        median_abs_deviation = scipy.stats.median_abs_deviation(data)
        bayes_mean_lower = scipy.stats.bayes_mvs(data)[0][1][0]
        bayes_mean_upper = scipy.stats.bayes_mvs(data)[0][1][1]
        bayes_variance_lower = scipy.stats.bayes_mvs(data)[1][1][0]
        bayes_variance_upper = scipy.stats.bayes_mvs(data)[1][1][1]
        bayes_std_lower = scipy.stats.bayes_mvs(data)[2][1][0]
        bayes_std_upper = scipy.stats.bayes_mvs(data)[2][1][1]

        df_metrics = pd.DataFrame([mean, mode, mode_freq, stdev, minimum, first_Q, median, third_Q, maximum, variance,
                                   range, iqr,  hmean, kurtosis, skew, sem, variation, trim_mean,
                                   median_abs_deviation, bayes_mean_lower, bayes_mean_upper, bayes_variance_lower,
                                   bayes_variance_upper, bayes_std_lower, bayes_std_upper]).transpose()
    else:
        df_metrics = pd.DataFrame([mean, mode, mode_freq, stdev, minimum, first_Q, median,
                                   third_Q, maximum, variance, range, iqr]).transpose()

    df_metrics.columns = all_columns[:df_metrics.shape[1]]

    return df_metrics, pd.DataFrame(df[pick])

# convert from normal text to snake case for all columns in df


def util_snakify_cols(df):
    df.columns = [snakecase(col).replace('__', '_') for col in df.columns]
    return df

# Describe DataFrame; ignore and specifics may be a list of strings of column names
# Proportions need to work properly


def describe(df, _MAX=5, pick='none', ignore=['-1'], specifics=['-1']):
    df = util_snakify_cols(df)
    pick = snakecase(pick).replace('__', '_')
    _MAX = int(_MAX)

    df = strict_exclude_nan(df)

    print('DEBUG 1, PRINT ALL INITIAL TRUE COLUMN NAMES: \n\t' + str(list(df.columns)))

    # Drop ID column, only perform for specific features (if specified)
    if('id' in df.columns):
        df.drop('id', 1, inplace=True)
    if('empty_header_0' in df.columns):
        df.drop('empty_header_0', 1, inplace=True)

    try:
        if('-1' not in ignore):
            ignore = [snakecase(ignore_val).replace('__', '_') for ignore_val in ignore]
            try:
                df.drop(ignore, 1, inplace=True)
            except Exception as e:  # specified ignore col names not in DataFrame
                print('#@ LINE ' + str(getframeinfo(currentframe()).lineno) +
                      ': Incorrect ignore/specifics params, Exception ' + e)
        if('-1' not in specifics):
            specifics = [snakecase(specifics_val).replace('__', '_') for specifics_val in specifics]
            try:
                df = df[specifics]
            except Exception as e:  # specified specifics col names not in DataFrame
                print('#@ LINE ' + str(getframeinfo(currentframe()).lineno) +
                      ': Incorrect ignore/specifics params, Exception ' + e)
    except Exception:
        print('>>> Incorrect argument for "ignore" and/or "specifics" parameter(s)')

    print('DEBUG 2, PRINT ALL PROCESSED TRUE COLUMN NAMES: \n\t' + str(list(df.columns)))

    print('@ LINE ' + str(getframeinfo(currentframe()).lineno) + ', PROCESSED DATFRAME SUCCESSFULLY.')
    feature_freq_dist = freq_dist(df, pick)
    print('@ LINE ' + str(getframeinfo(currentframe()).lineno) + ', CALCULATED FREQUENCY DATA SUCCESSFULLY.')

    # Gets class proportions for all features
    all_features = {}
    total_rows = float(df.shape[0])
    for col_i in range(len(df.columns)):
        # get frequency of each unique value in columns (string and numerical accepted) and delete
        freq_tracker = dict(list(Counter(df[df.columns[col_i]]).items())[:_MAX])
        # order dict by frequency
        freq_tracker = {k: str(round((v / total_rows) * 100.0, 3)) + '%' for k,
                        v in sorted(freq_tracker.items(), key=lambda item: item[1], reverse=True)}
        # re-structure dict -> list of strings, and add None keys if len < _MAX
        freq_tracker = [str(k) + ' (' + v + ')' for k, v in list(freq_tracker.items())]
        freq_tracker.extend([None] * (_MAX - len(freq_tracker)))

        # df_freq = pd.DataFrame()
        all_features[df.columns[col_i]] = freq_tracker

    print('@ LINE ' + str(getframeinfo(currentframe()).lineno) + ', CALCULATED ALL PROPORTION DATA SUCCESSFULLY.')

    # nuniques for all features
    col_freq = pd.DataFrame(df.nunique())

    # n%unique for each num feature
    quant_info = df.describe().iloc[1:].transpose()
    quant_info['# unique'] = col_freq[0].astype(int)
    quant_info = util_modify_tops(quant_info, _MAX, all_features)

    print('@ LINE ' + str(getframeinfo(currentframe()).lineno) + ', CALCULATED QUANT DATA SUCCESSFULLY.')

    # n%unique for each cat feature
    cat_info = pd.DataFrame()
    cat_info['# unique'] = col_freq[0].astype(int)
    cat_info.drop(list(quant_info.index), 0, inplace=True)

    cat_info = util_modify_tops(cat_info, _MAX, all_features)

    print('@ LINE ' + str(getframeinfo(currentframe()).lineno) + ', CALCULATED CAT DATA SUCCESSFULLY.')

    quant_info = quant_info.reset_index().rename(columns={'index': 'feature'})
    cat_info = cat_info.reset_index().rename(columns={'index': 'feature'})

    if(pick == 'none'):
        index = np.random.choice(range(len(df.columns)))
        pick = df.columns[index]

    print('')

    return quant_info, cat_info, feature_freq_dist

# import datetime
# from datetime import datetime
#
# path = '/Users/Ray/Documents/Python/5 - Webber/Unemplyment and Education.csv'
# df = pd.read_csv(path, encoding = "ISO-8859-1")
# df_link = pd.read_csv('/Users/Ray/Documents/Python/5 - Webber/Countries-Continents.csv', encoding = "ISO-8859-1")
# df.columns = ['country', 'code', 'year', 'population', 'continent', 'unemployment_rate_advanced_edu',
#               'unemployment_rate_basic_edu']
# linkage = dict(zip(df_link['Country'], df_link['Continent']))
# df.drop(['code'], 1, inplace = True)
# df['year'] = [datetime.strptime(str(val), '%Y') for val in df['year']]
# df['continent'] = [linkage.get(country) for country in df['country']]
# df = util_insert_ID(df)
#
# df.to_csv(path)

# df.columns = [col.replace("'", "").replace('"',"").strip() for col in df.columns]
# df = util_insert_ID(df)
#
# df.to_csv(path)
#
# y_min = 5000
# x_min = 0
# plt.plot(df['AF4'], df['FC5'])
# plt.plot(df[df['AF4'] > x_min]['AF4'], df[df['FC5'] > y_min]['FC5'])
# df.columns
#
