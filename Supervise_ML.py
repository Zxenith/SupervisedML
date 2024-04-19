import numpy as np
import copy
import math
import pandas as pd
from matplotlib import pyplot as plt


def compute_cost(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


def cost_func(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
        
    cost /= (2 * m)

    return cost


def gradient(x, y, w, b):
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += f_wb - y[i]

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def grad_descent(x, y, w_in, b_in, alpha, iters):
    j_history = []
    p_history = []

    w = w_in
    b = b_in
    for i in range(iters):
        dj_dw, dj_db = gradient(x, y, w, b)
        cost = cost_func(x, y, w, b)
        j_history.append(cost)
        p_history.append([w, b])

        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % math.ceil(iters / 10) == 0:
            print(f'Iteration number: {i:4}\n'
                  f'\tCost: {cost:0.2e}'
                  f'\tw: {w:0.2e}'
                  f'\tb: {b:0.2e}'
                  f'\tdj_dw: {dj_dw:0.2e}'
                  f'\tdj_db: {dj_db:0.2e}\n')

    return w, b, j_history, p_history


def predict_withw(x, w, b):
    m = x.shape[0]
    p = 0

    for i in range(m):
        p += w[i] * x[i]
    p += b

    return p


def predict(x, w, b):
    p = np.dot(x, w) + b
    return p


def multi_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        cost += ((np.dot(x[i], w) + b) - y[i]) ** 2

    cost /= 2 * m

    return cost


def multi_gradient(x, y, w, b):
    m, n = x.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        erro = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += erro * x[i, j]
        dj_db += erro

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def multi_gradient_descent(x, y, w_in, b_in, alpha, num_iters=10000):
    w = copy.deepcopy(w_in)
    b = b_in

    j_hist = []

    for i in range(num_iters):
        dj_dw, dj_db = multi_gradient(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = multi_cost(x, y, w, b)
        j_hist.append(cost)

        if i % math.ceil(num_iters / 10) == 0:
            print(f'Iteration: {i}\n'
                  f'Cost: {cost:0.3f}\t'
                  f'w: {w}\t'
                  f'b: {b:0.3f}\n')

    return w, b, j_hist


def mean_normalization(x):
    X = copy.deepcopy(x)
    m, n = X.shape

    for i in range(n):
        col = X.iloc[:, i]
        meany = col.mean()
        mining = col.max() - col.min()

        for j in range(m):
            col.iloc[j] = (col.iloc[j] - meany) / mining

    return X


def z_score_normalization(x):
    X = copy.deepcopy(x)
    m, n = X.shape

    for i in range(n):
        col = X.iloc[:, i]
        meany = col.mean()
        standy = col.std()

        for j in range(m):
            col.iloc[j] = (col.iloc[j] - meany) / standy

    return X


def mean_single(y):
    Y = copy.deepcopy(y)
    m = Y.shape[0]

    meany = Y.mean()
    mining = Y.max() - Y.min()

    for i in range(m):
        Y.iloc[i] = (Y.iloc[i] - meany) / mining

    return Y


def z_single(y):
    Y = copy.deepcopy(y)
    m = Y.shape[0]

    meany = Y.mean()
    standy = Y.std()

    for i in range(m):
        Y.iloc[i] = (Y.iloc[i] - meany) / standy

    return Y


def predict_single(x, w, b):
    p = np.zeros_like(x)
    m = x.shape[0]

    for i in range(m):
        p[i] = w * x[i] + b

    return p


def polynomial_regression(x, w, b):
    p = predict_single(x, w, b)
    return p


def sigmoid(z):
    g = 1 / (1 + (np.exp(-z)))
    return g


def loss(fx_wb, y):
    loss_i = -(y * np.log(fx_wb)) - ((1 - y) * np.log(1 - fx_wb))
    return loss_i


def cost_logarithmic(X, y, w, b):
    # X = Xp.to_numpy()
    # y = yp.to_numpy()

    m = X.shape[0]

    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += loss(f_wb_i, y[i])
    cost /= m

    return cost


def gradient_logistic(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        fx_wb = sigmoid(np.dot(X[i], w) + b)
        err = fx_wb - y[i]
        for j in range(n):
            dj_dw[j] += (err * X[i, j])
        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent_Logistic(X, y, w_in, b_in, alpha, iters):
    m, n = X.shape

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(iters):
        J_history.append(cost_logarithmic(X, y, w, b))

        dj_dw, dj_db = gradient_logistic(X, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % math.ceil(iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history


def regularised_cost_linear(X, y, w, b, lambda_=1):
    reg_cost = 0.
    m = X.shape[0]
    n = len(w)
    for j in range(n):
        reg_cost += (w[j] ** 2)
    reg_cost *= (lambda_ / (2 * m))

    total_cost = multi_cost(X, y, w, b) + reg_cost
    # total_cost = sml.multi_cost(X, y, w, b) + reg_cost

    return total_cost


def regularised_cost_logistic(X, y, w, b, lambda_=1):
    m = X.shape[0]
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)

    total_cost = cost_logarithmic(X, y, w, b) + reg_cost
    # total_cost = sml.cost_logarithmic(X, y, w, b) + reg_cost

    return total_cost


def regularised_gradient_linear(X, y, w, b, lambda_):
    dj_dw, dj_db = multi_gradient(X, y, w, b)
    # dj_dw, dj_db = sml.multi_gradient(X, y, w, b)

    m, n = X.shape

    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]

    return dj_dw, dj_db


def regularised_gradient_logistic(X, y, w, b, lambda_):
    dj_dw, dj_db = gradient_logistic(X, y, w, b)
    # dj_dw, dj_db = sml.gradient_logistic(X, y, w, b)

    m, n = X.shape

    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]

    return dj_dw, dj_db


def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5  # 12-15 min is best
    X[:, 0] = X[:, 0] * (285 - 150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1, 1))

def dense(X,W,b):
    units = W.shape[1]
    p = np.zeros(units)
    for j in range(units):
        w = W[:,j]
        z = np.dot(w,X) + b[j]
        p[j] = sigmoid(z)
    return p

def sequential(X,W1,b1,W2,b2):
    a1 = dense(X,W1,b1)
    a2 = dense(a1,W2,b2)
    return a2

def predict_neural(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros(m)
    for i in range(m):
        p[i] = sequential(X[i], W1, b1, W2, b2)
    return(p)

def manual_softmax(z):
    ez = np.exp(z)
    sumo = ez/np.sum(ez)
    return sumo
