import numpy as np


def generate_data(X, F, B, u, Q, H, R, T):
    v = np.random.multivariate_normal(np.array([0, 0]), R, T).T
    for i in range(T - 1):
        w = np.random.multivariate_normal(np.zeros(4), Q).T
        X[i + 1,] = np.matmul(F, X[i,]) + B * u + w
    X = np.transpose(X)
    Y = np.matmul(H, X) + v
    return X, Y

def generate_QR_EM(smooth_x,smooth_sig,L_smooth, X_kk, Y ,F, H):
    n, dim_x = X_kk.shape
    Q_est = np.zeros((dim_x, dim_x))
    R_est = np.zeros((2, 2))

    #Expectation
    Xk = smooth_x
    XkXkT = np.zeros((n, dim_x, dim_x))
    XkXk1T = np.zeros((n, dim_x, dim_x))
    for j in range(n - 1, -1, -1):
        XkXkT[j] = np.outer(smooth_x[j], smooth_x[j].T) + smooth_sig[j]
        bracket = smooth_sig[j] + np.outer((smooth_x[j] - X_kk[j]), smooth_x[j].T)
        if j>0:
            XkXk1T[j-1] = np.outer(X_kk[j - 1], smooth_x[j].T) + np.matmul(L_smooth[j - 1], bracket)
    #Maximalization
    for k in range(n - 1, -1, -1):
        if k > 0:
            Q_est += XkXkT[k] + np.matmul(np.matmul(F, XkXkT[k - 1]), F.T) - np.matmul(
                XkXk1T[k - 1].T, F.T) - np.matmul(F, XkXk1T[k - 1])
        R_est += np.outer(Y[k], Y[k].T) + np.matmul(np.matmul(H, XkXkT[k]), H.T) - np.matmul(
            np.outer(Y[k], Xk[k].T), H.T) - np.matmul(H, np.outer(Xk[k], Y[k].T))
    Q_est = Q_est / (n - 1)
    R_est = R_est / (n)
    return Q_est, R_est



def kalman_filter(X, Y, F, B, u, Q, H, R, Sigma, T):
    X_kk = np.zeros((T, 4))
    Y_d = np.zeros((T, 2))
    Sigma_kk = np.zeros((T, 4, 4))
    for i in range(T):
        Y_d[i,] = Y[i,] - np.matmul(H, X)
        S = R + np.matmul(np.matmul(H, Sigma), H.T)
        K = np.matmul(np.matmul(Sigma, H.T), np.linalg.inv(S))
        X_kk[i,] = X + np.matmul(K, Y_d[i,])
        Sigma_kk[i, :, :] = Sigma - np.matmul(np.matmul(K, H), Sigma)
        X = np.matmul(F, X_kk[i,]) + B * u
        Sigma = np.matmul(np.matmul(F, Sigma_kk[i, :, :]), F.T) + Q
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
