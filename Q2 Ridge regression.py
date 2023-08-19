def gradient_descent_ridge(X, y, lambda_, alpha, max_iterations):
    m, n = X.shape
    theta = np.random.randn(n,1)
    for i in range(max_iterations):
        y_pred = np.dot(X, theta)
        error = y_pred - y
        grad = (1/m) * (np.dot(X.T, error) + lambda_*theta)
        theta = theta - alpha * grad
    return theta

X = X.transpose()
X_test = X_test.transpose()

print(X.shape)
print(y.shape)
print(X_test.shape)
print(y_test.shape)

import numpy as np
import matplotlib.pyplot as plt


X_train = X
y_train = y
X_test = X_test
y_test = y_test

def ridge_regression_gradient_descent(X, y, alpha, lambda_, n_iter):
    m, n = X.shape
    w = np.zeros((n, 1))
    
    for _ in range(n_iter):
        gradient = (2 / m) * (X.transpose() @ (X @ w - y) + lambda_ * w)
        w -= alpha * gradient
    
    return w

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Cross-validation
lambda_vals = np.logspace(-5, 5, 50)
cv_errors = []

for lambda_ in lambda_vals:
    w_ridge = ridge_regression_gradient_descent(X_train, y_train, alpha=0.001, lambda_=lambda_, n_iter=5000)
    y_pred_val = X_train @ w_ridge
    cv_error = mean_squared_error(y_train, y_pred_val)
    cv_errors.append(cv_error)

# Find best lambda and corresponding wR
best_lambda_idx = np.argmin(cv_errors)
best_lambda = lambda_vals[best_lambda_idx]
wR = ridge_regression_gradient_descent(X_train, y_train, alpha=0.001, lambda_=best_lambda, n_iter=5000)

# wML using normal equations
wML = np.linalg.inv(X_train.transpose() @ X_train) @ X_train.transpose() @ y_train

# Test errors
y_pred_ridge = X_test.transpose() @ wR
y_pred_ML = X_test.transpose() @ wML

test_error_ridge = mean_squared_error(y_test, y_pred_ridge)
test_error_ML = mean_squared_error(y_test, y_pred_ML)

print(f"Test error for wR (ridge): {test_error_ridge}")
print(f"Test error for wML (maximum likelihood): {test_error_ML}")

# Plot cross-validation error
plt.figure()
plt.plot(lambda_vals, cv_errors)
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Cross-validation error")
plt.title("Cross-validation error vs. Lambda")
plt.show()