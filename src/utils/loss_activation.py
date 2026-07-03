from .math import se


# ReLU (Rectified Linear Unit) activation
def relu(x: int | float) -> int | float:

    if x > 0:
        return x
    return 0

# identity activation
def identity(x: int | float) -> int | float:
    return x

# MSE (Mean Squared Error) loss
def mse(predictions: list, expecteds: list):

    errors = []
    n = len(predictions)
    for i in range(n):
        errors.append(se(expecteds[i], predictions[i]))

    s = sum(errors)

    return s/n


