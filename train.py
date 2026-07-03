from src.get_data import load
import pandas as pd
from src.regression import LinearRegression
from src.utils.loss_activation import mse, se, identity

def vectorize(data: pd.DataFrame):

    # replacing binary cols
    data = data.replace({
        "yes": 1,
        "no": 0
    })

    data["furnishingstatus"] = data["furnishingstatus"].replace({
        "furnished": 1,
        "semi-furnished": 0,
        "unfurnished": -1
    })

    data = data.astype(float)
    data.insert(1, "Bias Placeholder", [1] * len(data["price"]))
    data["price"] = data["price"]/1e6
    data["area"] = data["area"]/1e3

    return data

def train():
    data = load()
    data = vectorize(data)
    print(data)

    regression_model = LinearRegression(12, identity, se, data, mse)
    results = regression_model.gradient_descent()
    print([float(result) for result in results])


if __name__ == "__main__":
    train()