# Calculates squared error
def se(y, py):
    return (y - py)**2

# differentiates through nudging
def approx_partial_derivative(f, i, weights, x, y, h = 1e-8):

    # w plus and minus arrays for calculating derivative
    w_plus = [w for w in weights]
    w_plus[i] += h
    w_minus = [w2 for w2 in weights]
    w_minus[i] -= h

    # calculating derivative 
    return (f(y, sum(w * inp for w, inp in zip(w_plus, x))) - f(y, sum(w2 * inp2 for w2, inp2 in zip(w_minus, x))))/(2 * h)

# finds avg derivative/partial derivative 
def avg_derivative(idx: int, weights: list, y: list, x: list, loss) -> float | int:
    n = len(x)
    total = 0

    for i in range(n):
        total += approx_partial_derivative(loss, idx, weights, x[i], y[i])
    
    return total/n