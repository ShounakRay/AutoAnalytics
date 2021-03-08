# @Author: Shounak Ray <Ray>
# @Date:   08-Mar-2021 13:03:97:975  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: h20_exploration.py
# @Last modified by:   Ray
# @Last modified time: 08-Mar-2021 14:03:72:724  GMT-0700
# @License: [Private IP]


from __future__ import print_function

from builtins import range

import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

dir(h2o)

print("Deep Learning Anomaly Detection MNIST")

train = h2o.import_file(pyunit_utils.locate("bigdata/laptop/mnist/train.csv.gz"))
test = h2o.import_file(pyunit_utils.locate("bigdata/laptop/mnist/test.csv.gz"))

predictors = list(range(0, 784))
resp = 784

# unsupervised -> drop the response column (digit: 0-9)
train = train[predictors]
test = test[predictors]

# 1) LEARN WHAT'S NORMAL
# train unsupervised Deep Learning autoencoder model on train_hex

ae_model = H2OAutoEncoderEstimator(activation="Tanh", hidden=[2], l1=1e-5, ignore_const_cols=False, epochs=1)
ae_model.train(x=predictors, training_frame=train)

# 2) DETECT OUTLIERS
# anomaly app computes the per-row reconstruction error for the test data set
# (passing it through the autoencoder model and computing mean square error (MSE) for each row)
test_rec_error = ae_model.anomaly(test)

# 3) VISUALIZE OUTLIERS
# Let's look at the test set points with low/median/high reconstruction errors.
# We will now visualize the original test set points and their reconstructions obtained
# by propagating them through the narrow neural net.

# Convert the test data into its autoencoded representation (pass through narrow neural net)
test_recon = ae_model.predict(test)

# In python, the visualization could be done with tools like numpy/matplotlib or numpy/PIL


# pyunit_utils.standalone_test(anomaly)
