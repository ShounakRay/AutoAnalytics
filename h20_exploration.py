# @Author: Shounak Ray <Ray>
# @Date:   08-Mar-2021 13:03:97:975  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: h20_exploration.py
# @Last modified by:   Ray
# @Last modified time: 09-Mar-2021 12:03:17:173  GMT-0700
# @License: [Private IP]


import subprocess

import h2o
from h2o.automl import H2OAutoML

version = subprocess.check_output(['java', '-version'],
                                  stderr=subprocess.STDOUT).decode().split('"')[1].split('.')[0]
if not (version >= 8 and version <= 14):
    raise ValueError('STATUS: Java Version is not between 8 and 15.\n  \
                      h2o instance will not be initialized')

h2o.init()
h2o.cluster().show_status()

# Import a sample binary outcome train/test set into H2O
train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

# Identify predictors and response
x = train.columns
y = "response"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
