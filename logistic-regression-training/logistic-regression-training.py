import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X_arr = np.array(X)
    y_arr = np.array(y)

    N, num_features = X_arr.shape
    w= np.zeros(num_features)
    b=0.0

    for step in range(steps):
        z = np.dot(X_arr,w)+b
        p = _sigmoid(z)
        dz = p - y_arr

        dw = (1/N)* np.dot(X_arr.T,dz)
        db = (1/N) * np.sum(dz)
        w = w - lr * dw
        b = b - lr * db

    return w,b 
    pass