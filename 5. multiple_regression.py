import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = 3 * np.random.rand(100,2)
y = 4 + 2 * X[:, 0] + X[:, 1] +np.random.rand(100)

model = LinearRegression()
model.fit(X, y)

coefficients = model.coef_
intercept = model.intercept_

print(f"Coeffiecients: {coefficients}")
print(f"Intercept: {intercept}")
