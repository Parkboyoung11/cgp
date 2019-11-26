import numpy as np
from sklearn.linear_model import LinearRegression

data = np.load('F1.npz')
X, y = data['x'].reshape(-1, 1), data['y'].reshape(-1,1)

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

print(y_pred.shape, y.shape)
print(np.mean(np.sqrt(np.power(y - y_pred, 2))))
print(reg.score(X, y))
