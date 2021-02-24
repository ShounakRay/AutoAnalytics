# @Author: Shounak Ray <Ray>
# @Date:   14-Oct-2020 23:10:65:652  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: Regression.py
# @Last modified by:   Ray
# @Last modified time: 24-Feb-2021 00:02:04:046  GMT-0700
# @License: [Private IP]


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stringcase import snakecase


# convert from normal text to snake case for all columns in df
def util_snakify_cols(df):
    df.columns = [snakecase(col).replace('__', '_') for col in df.columns]
    return df


def reg(df, x, y, degree):
    df = util_snakify_cols(df)
    x = snakecase(x).replace('__', '_')
    y = snakecase(y).replace('__', '_')
    df_ORIG = df.copy()
    df_males = df[df['gender'] == 'Male']
    df_females = df[df['gender'] == 'Female']

    # # Scatter plots.
    # ax1 = df_males.plot(kind='scatter', x='Height', y='Weight', color='blue', alpha=0.5, figsize=(10, 7))
    # df_females.plot(kind='scatter', x='Height', y='Weight', color='magenta', alpha=0.5, figsize=(10, 7), ax=ax1)

    male_fit, male_res, rank_male, _, _, = np.polyfit(df_males[x], df_males[y], degree, full=True)
    male_corr = round(df_males.corr().iloc(0)[0][1], 3)
    female_fit, female_res, rank_female, _, _,  = np.polyfit(df_females[x], df_females[y], degree, full=True)
    female_corr = round(df_females.corr().iloc(0)[0][1], 3)

    fig, ax = plt.subplots()
    ax.plot(df_males[x], male_fit[0] * df_males[x] + male_fit[1], linewidth=2)
    ax.plot(df_females[x], female_fit[0] * df_females[x] + female_fit[1], linewidth=2)
    x_data_males = ax.lines[0].get_xdata()
    y_data_males = ax.lines[0].get_ydata()
    x_data_females = ax.lines[1].get_xdata()
    y_data_females = ax.lines[1].get_ydata()
    plt.close()

    df_output_1 = pd.DataFrame(x_data_females, y_data_females).reset_index()
    df_output_1['coeff'] = female_corr
    df_output_1['residual'] = round(male_res[0], 3)
    df_output_2 = pd.DataFrame(x_data_males, y_data_males).reset_index()
    df_output_2['coeff'] = male_corr
    df_output_2['residual'] = round(female_res[0], 3)

    df_output_1.columns = ['y_1', 'x_1', 'coeff', 'resid']
    df_output_2.columns = ['y_2', 'x_2', 'coeff', 'resid']

    # df_output_1 = pd.concat([df[df['gender'] == 'Female'], df_output_1], 1).reset_index().drop('index', 1)
    df_output_2 = pd.concat([df[df['gender'] == 'Male'], df_output_2], 1).reset_index().drop('index', 1)

    return df_output_1, df_output_2

# df = pd.read_csv('/Users/Ray/Documents/Python/5 - Webber/Datasets/weight-height.csv')
# a, b = reg(df, 'Height', 'Weight', 1)


#
