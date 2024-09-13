from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)
X = np.random.rand(100,1)
y = 2 * X.squeeze() + 1 + 0.1 * np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set Size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")