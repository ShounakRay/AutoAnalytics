import pandas as pd
import numpy as np
from collections import Counter
from inspect import currentframe, getframeinfo
from stringcase import snakecase
import scipy
import scipy.stats
import scipy.signal
from distfit import distfit

import matplotlib.pyplot as plt

# Insert df
def prune(df, insert, nan = ''):
    df = util_snakify_cols(df)

    df = strict_exclude_nan(df, nan = nan)
    if(insert == 'True'):
        df = util_insert_ID(df)
        return df
    elif(insert == 'False'):
        return df
    else:
        return 'Incorrect value for "insert" argument.'

# Add id/index columns to df
def util_insert_ID(df):
    df = df.reset_index(0).rename(columns = {'index': 'id'})
    return df

# Replace all values with certain char (such as ?) with NaN
# Delete all rows that have any NaN values
# Delete all columns that only have NaN values
def strict_exclude_nan(df, nan = ''):
    df = util_snakify_cols(df)

    df = util_clean_values(df)
    df.replace(to_replace = nan, value = np.NaN, inplace = True)
    df.dropna(axis = 0, how = 'any', inplace = True)
    df.dropna(axis = 1, how = 'all', inplace = True)
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
    new_content_type = [list(type_info.values[i][:orig_col_count]) + list(rel_type[type_info.index[i]]) for i in range(len(type_info.index))]
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
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

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
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

# outputs dataframe with frequency distribution of specific/random feature
def freq_dist(df, pick):
    df = util_snakify_cols(df)

    pick = snakecase(pick).replace('__', '_')

    if(pick == 'none'):
        index = np.random.choice(range(len(df.columns)))
    else:
        index = df.columns.get_loc(pick)

    if(isinstance(df[df.columns[index]][0], str)): # order by count/value
        total_rows = float(df.shape[0])
        freq_tracker_pct = {str(k): round((v/total_rows) * 100.0, 3) for k, v in sorted(dict(Counter(df[df.columns[index]])).items(), key = lambda item: item[1], reverse = True)}
        freq_tracker_count = {str(k): v for k,v in sorted(dict(Counter(df[df.columns[index]])).items(), key = lambda item: item[1], reverse = False)}
    else: # order by number/key
        total_rows = float(df.shape[0])
        freq_tracker_pct = {str(k): round((v/total_rows) * 100.0, 3) for k, v in sorted(dict(Counter(df[df.columns[index]])).items(), key = lambda item: item[0], reverse = True)}
        freq_tracker_count = {str(k): v for k,v in sorted(dict(Counter(df[df.columns[index]])).items(), key = lambda item: item[0], reverse = False)}

    df_freq = pd.DataFrame(freq_tracker_count.items())
    df_freq[2] = freq_tracker_pct.values()

    df_freq['pick'] = df.columns[index]
    df_freq.columns = ['feature', 'count', 'proportion', 'pick']

    return df_freq

# Normalize list
def util_normalize(list):
    list = (list - np.min(list))/(np.max(list) - np.min(list))
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

# returns smoothened list
def util_smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode = 'same')
    return y_smooth

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

# Returns best distribution and information associated
# Possible bug: resolution may cause extra index (so lens don't match)
def generate_dist_fits(df, pick, peak_max = 3, top_dists = 3):
    top_dists = int(top_dists)
    df = util_snakify_cols(df)
    pick = snakecase(pick).replace('__', '_')

    # Data Sanitation not required
    # if(isinstance(df[pick][0], str)):
    #     # BAD INPUT
    #     return -1

    HYPER_ASSOC = {'distr': 'Distribution', 'RSS': 'Root Sum Squared', 'LLE': 'Locally Linear Embedding', 'loc': 'LOC: Mean', 'scale': 'SCALE: Standard Deviation', 'arg': 'Argument(s)'}
    TAG_NAME_ASSOC = {'beta': 'Beta', 'genextreme': 'Generalized Extreme Value', 't': 'Student’s t', 'norm': 'Normal', 'lognorm': 'Lognormal ', 'gamma': 'Gamma', 'dweibull': 'Double Weibull', 'expon': 'Exponential', 'uniform': 'Uniform', 'pareto': 'Pareto'}
    MODEL_PARAM_ASSOC = {'beta': ['a', 'b'], 'genextreme': ['c'], 't': ['° of freedom'], 'lognorm': ['s'], 'gamma': ['a'], 'dweibull': ['c'], 'pareto': ['b']}
    NAME_OBJECT_ASSOC = {'beta': scipy.stats.beta, 'genextreme': scipy.stats.genextreme, 't': scipy.stats.t, 'norm': scipy.stats.norm, 'lognorm': scipy.stats.lognorm, 'gamma': scipy.stats.gamma, 'dweibull': scipy.stats.dweibull, 'expon': scipy.stats.expon, 'uniform': scipy.stats.uniform, 'pareto': scipy.stats.pareto}
    DEC_ROUND_NUM = 3


    # getattr(scipy.stats, dir(scipy.stats)[dir(scipy.stats).index(model_name)])


    # Distribution, Arguments,
    curve_df = pd.DataFrame()

    X = np.array(list(df[pick]))
    max = np.max(X)
    min = np.min(X)
    # For exisiting univariate data
    counts = sorted(list(Counter(X).items()), key = lambda item: item[0])
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
    peak = sorted(peaks, key = lambda item: item[1])[0]
    resolution = (max - min)/(len(X_x) - 1)

    dist = distfit(alpha = 0.05, smooth = 10)
    dist.fit_transform(X)

    top_models = dist.summary[:top_dists]
    top_models = top_models.rename(columns = HYPER_ASSOC)

    for row_i in range(len(top_models)):
        dist_OBJECT = NAME_OBJECT_ASSOC[top_models['Distribution'][row_i]]
        loc_val = top_models['LOC: Mean'][row_i]
        scale_val = top_models['SCALE: Standard Deviation'][row_i]
        name_val = top_models['Distribution'][row_i]
        params = top_models['Argument(s)'][row_i]
        params_val = ''
        for val_i in range(len(params)):
            if(val_i == len(params) - 1 - 1):
                params_val += MODEL_PARAM_ASSOC[name_val][val_i] + ': ' + str(round(params[val_i], DEC_ROUND_NUM)) + '; '
            else:
                params_val += MODEL_PARAM_ASSOC[name_val][val_i] + ': ' + str(round(params[val_i], DEC_ROUND_NUM))

        # B_dist_name, *B_p_value, B_params = util_get_best_distribution(list(df['age']))

        # For curve-fit plot
        x_top_dist = np.arange(min, max + resolution, resolution)
        y_top_dist = dist_OBJECT.pdf(x_top_dist, (*params), scale = scale_val, loc = loc_val)
        y_top_dist = util_normalize(y_top_dist)
        # Store information about models in DataFrame
        new_col_names = ['model' + "_X", 'model' + "_Y", 'model_id', 'model_params']
        new_df = pd.DataFrame(zip(x_top_dist, y_top_dist, [name_val] * len(y_top_dist), [params_val] * len(y_top_dist)), columns = new_col_names)
        curve_df = pd.concat([curve_df, new_df], axis = 0).reset_index().drop('index', 1)

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
    final_df = pd.concat([pick_df, curve_df], axis = 1)

    return final_df

# convert from normal text to snake case for all columns in df
def util_snakify_cols(df):
    df.columns = [snakecase(col).replace('__', '_') for col in df.columns]
    return df

# Describe DataFrame; ignore and specifics may be a list of strings of column names
# Proportions need to work properly
def describe(df, _MAX = 5, pick = 'none', ignore = ['-1'], specifics = ['-1'], top_dists = 5):
    df = util_snakify_cols(df)
    pick = snakecase(pick).replace('__', '_')

    df = strict_exclude_nan(df)

    print('DEBUG 1, PRINT ALL INITIAL TRUE COLUMN NAMES: \n\t' + str(list(df.columns)))

    # Drop ID column, only perform for specific features (if specified)
    if('id' in df.columns):
        df.drop('id', 1, inplace = True)
    if('empty_header_0' in df.columns):
        df.drop('empty_header_0', 1, inplace = True)

    try:
        if('-1' not in ignore):
            ignore = [snakecase(ignore_val).replace('__', '_') for ignore_val in ignore]
            try:
                df.drop(ignore, 1, inplace = True)
            except Exception as e: # specified ignore col names not in DataFrame
                print('#@ LINE ' + str(getframeinfo(currentframe()).lineno) + ': Incorrect ignore/specifics params, Exception ' + e)
        if('-1' not in specifics):
            specifics = [snakecase(specifics_val).replace('__', '_') for specifics_val in specifics]
            try:
                df = df[specifics]
            except Exception as e: # specified specifics col names not in DataFrame
                print('#@ LINE ' + str(getframeinfo(currentframe()).lineno) + ': Incorrect ignore/specifics params, Exception ' + e)
    except:
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
        freq_tracker = {k: str(round((v/total_rows)*100.0, 3)) + '%' for k, v in sorted(freq_tracker.items(), key = lambda item: item[1], reverse = True)}
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
    cat_info.drop(list(quant_info.index), 0, inplace = True)

    cat_info = util_modify_tops(cat_info, _MAX, all_features)

    print('@ LINE ' + str(getframeinfo(currentframe()).lineno) + ', CALCULATED CAT DATA SUCCESSFULLY.')

    quant_info = quant_info.reset_index().rename(columns = {'index': 'feature'})
    cat_info = cat_info.reset_index().rename(columns = {'index': 'feature'})

    if(pick == 'none'):
        index = np.random.choice(range(len(df.columns)))
        pick = df.columns[index]

    print('')

    return quant_info, cat_info, feature_freq_dist

df = pd.read_csv('/Users/Ray/Documents/Python/Webber/Datasets/adult.data.csv')
a, b, c = describe(df, pick = 'education_num')
d = generate_dist_fits(df, 'capital_gain')
d.to_html('all.html')




#
