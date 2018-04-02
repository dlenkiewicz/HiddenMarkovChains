import numpy as np


def generate_data(X, F, B, u, Q, H, v, T):
    for i in range(T - 1):
        w = np.random.multivariate_normal(np.zeros(4), Q).T
        X[i + 1,] = np.matmul(F, X[i,]) + B * u + w
    X = np.transpose(X)
    Y = np.matmul(H, X) + v
    return X, Y


def kalman_filter(X, Y, F, B, u, Q, H, R, Sigma, T):
    X_kk = np.zeros((T, 4))
    Y_d = np.zeros((T, 2))
    Sigma_kk = np.zeros((T, 4, 4))
    for i in range(T):
        Y_d[i,] = Y[i,] - np.matmul(H, X)
        S = R + np.matmul(np.matmul(H, Sigma), np.transpose(H))
        K = np.matmul(np.matmul(Sigma, np.transpose(H)), np.linalg.inv(S))
        X_kk[i,] = X + np.matmul(K, Y_d[i,])
        Sigma_kk[i, :, :] = Sigma - np.matmul(np.matmul(K, H), Sigma)
        X = np.matmul(F, X_kk[i,]) + B * u
        Sigma = np.matmul(np.matmul(F, Sigma_kk[i, :, :]), np.transpose(F)) + Q
    return X_kk, Sigma_kk


def rts_smoothing(Xs, Sigma, F, Q):
    n, dim_x = Xs.shape

    L = np.zeros((n, dim_x, dim_x))
    x, sig = Xs.copy(), Sigma.copy()

    for k in range(n - 2, -1, -1):
        sig_pred = np.dot(F, sig[k]).dot(F.T) + Q

        L[k] = np.dot(sig[k], F.T).dot(np.linalg.inv(sig_pred))
        x[k] += np.dot(L[k], x[k + 1] - np.dot(F, x[k]))
        sig[k] += np.dot(L[k], sig[k + 1] - sig_pred).dot(L[k].T)
    return x, sig, L