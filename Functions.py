import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from sklearn import metrics

def generate_data(X, F, B, u, Q, H, R, T):
    v = np.random.multivariate_normal(np.array([0, 0]), R, T).T
    for i in range(T - 1):
        w = np.random.multivariate_normal(np.zeros(4), Q).T
        X[i + 1,] = np.matmul(F, X[i,]) + B * u + w
    X = np.transpose(X)
    Y = np.matmul(H, X) + v
    return X, Y


def generate_QR_EM(smooth_x, smooth_sig, L_smooth, X_kk, Y, F, H, B_u):
    n, dim_x = X_kk.shape
    Q_est = np.zeros((dim_x, dim_x))
    R_est = np.zeros((2, 2))

    for i in range(n):
        if i >= 1:
            err = (smooth_x[i] - np.dot(F, smooth_x[i-1]) - B_u)
            Vt1t_A = np.dot(smooth_sig[i], np.dot(L_smooth[i-1].T, F.T))
            Q_est += np.outer(err, err) + np.dot(F, np.dot(smooth_sig[i-1], F.T)) + smooth_sig[i] - Vt1t_A - Vt1t_A.T
        err = (Y[i] - np.dot(H, smooth_x[i]))
        R_est += np.outer(err,err) + np.dot(H, np.dot(smooth_sig[i], H.T))

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

def count_MSE(method, args, realX, column):
    if method=='Kalman':
        X_est, Sig_est = kalman_filter(*args)
    if method=='RTS':
        X_est, Sig_est, L_est = rts_smoothing(*args)
    mse=metrics.mean_squared_error(X_est[:, column], realX[:, column])
    return mse

# def sym_MSE(method, args, realX, column, iter=1):
#     num_cores = multiprocessing.cpu_count()
#     results = Parallel(n_jobs=num_cores - 1)(delayed(count_MSE)(method, args, realX, column) for i in range(iter))
#     return np.mean(results)

def sym_MSE(method, args, realX, column, iter=1):
    results=0
    for k in range(iter):
        results+=count_MSE(method, args, realX, column)
    results=results/iter
    return np.mean(results)