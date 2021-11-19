import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def prepare_data(data):
    data["horsepower"] = pd.to_numeric(data["horsepower"], errors='coerce')
    data=data.fillna(data.mean())
    data=data.drop(['car name'],axis=1)
    return data


def split_data(df):
    X = df.drop('mpg', axis=1)
    Y = df['mpg']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    data = {"train":{"X":x_train,"y":y_train},
            "test" :{"X":x_test, "y":y_test}}
    return data

def train_model(data,args):

    reg_model = LinearRegression(normalize=True)
    reg_model.fit(data["train"]["X"],data["train"]["y"])
    return reg_model

def get_model_metrics(reg_model, data):
    preds = reg_model.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    metrics = {"mse": mse}
    return metrics


def main():

    sample_data = pd.read_csv('auto-mpg.csv')

    df = pd.DataFrame(data=sample_data.data,
                columns=sample_data.feature_names)

    df['mpg'] = sample_data.target

    df = prepare_data(df)

    data = split_data(df)

    args = {}
    reg = train_model(data,args)

    metrics = get_model_metrics(reg,data)

    model_name = "sklearn_regression_model.pkl"

    joblib.dump(value=reg, filename=model_name)

if __name__ == '__main__':
    main()