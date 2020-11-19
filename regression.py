import requests
import pandas
import scipy
import numpy as np
import sys
- pip install -r requirements.txt
from sklearn import linear_model


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"

req = requests.get(TRAIN_DATA_URL)
url_content = req.content
train = open('downloaded.csv', 'wb')

req = requests.get(TEST_DATA_URL)
url_content = req.content
test = open('downloaded.csv', 'wb')


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    train_x = np.asarray(train[0][1:])
    train_y = np.asarray(train[1][1:])
    test = np.asarray(test[0][1:])
    regr = linear_model.LinearRegression()
    regr.fit(train_x, train_y)
    pred = regr.predict(test)
    return pred


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
