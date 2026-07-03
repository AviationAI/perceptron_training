from .utils.math import avg_derivative
from .utils.loss_activation import mse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


# Class for linear regression 
class LinearRegression():

    def __init__(self, inputs: int, activation, loss, data: pd.DataFrame, total_loss):
        self.weights = [0.0 for _ in range(inputs + 1)]
        self.num_weights = inputs + 1
        self.activation = activation
        self.loss = loss
        self.total_loss = total_loss
        self.data = data

    # trains model
    def gradient_descent(self, lr = 1e-3, epochs = 5000):
        
        n =  self.num_weights

        # splitting into test & train
        train = self.data.sample(frac = 0.8, random_state = 42)
        test = self.data.drop(train.index)

        train_y, train_x = train["price"], train.drop("price", axis = 1)
        test_y, test_x = test["price"], test.drop("price", axis = 1)

        expecteds = test_y.to_list()
        train_expecteds = train_y.to_list()
        train_x_list = train_x.values.tolist()

        losses = []

        for l in range(epochs):
            new = self.weights.copy()
            print("Epoch Number: ", l + 1)
            for i in range(n):
                # finding the average partial derivative of a weight with all the examples
                avg = avg_derivative(i, self.weights,  train_expecteds, train_x_list, self.loss)

                # subtracting derivative times learning rate to move closer to global minimum
                new[i] -= lr * avg

            self.weights = new

            train_predictions = []

            # finding loss
            for j in range(len(test_y)): 
                inps = train_x.iloc[j]
                guess = 0
                for k in range(n):
                    guess += self.weights[k] * inps.iloc[k]
                train_predictions.append(guess)
            loss = self.total_loss(train_predictions, train_expecteds)
            print("MSE Loss: ", loss)
            losses.append(loss)


        # plotting train losses
        
        # x values
        x = np.linspace(0, epochs, epochs)

        plt.figure(figsize=(10, 10))
        plt.plot(x, losses, label = "Losses")
        plt.grid(True)

        plt.title("Loss vs. Epochs")
        plt.show()
        

        predictions = []

        # testing regression alg
        for j in range(len(test_y)):
            inps = test_x.iloc[j]
            guess = 0
            for k in range(n):
                guess += self.weights[k] * inps.iloc[k]
            predictions.append(guess)
        
        tl = self.total_loss(predictions, expecteds)
        rmse = math.sqrt(tl)
        
        # R2 score
        res = sum((e - p) ** 2 for e, p in zip(expecteds, predictions))

        mean = sum(expecteds)/len(expecteds)
        tot =  sum((e - mean) ** 2 for e in expecteds)
        r2 = 1 - res/tot


        return tl, r2, rmse