import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def generate_data(X, F, B, u, Q, H, R, T):
    v = np.random.multivariate_normal(np.array([0, 0]), R, T).T
    for i in range(T - 1):
        w = np.random.multivariate_normal(np.zeros(4), Q).T
        X[i + 1,] = np.matmul(F, X[i,]) + B * u + w
    X = np.transpose(X)
    Y = np.matmul(H, X) + v
    return X, Y

class SnapshotListHolder:
    def __init__(self, snapshot):
        self.__snapshot = list(snapshot)

    def take_snapshot(self, snapshot):
        self.__snapshot = list(snapshot)

    def get_snapshot(self):
        return self.__snapshot

def evaluate_stop_condition(current_param, param_snapshot, current_iter, max_iter, threshold=0.0001):
    if current_iter >= max_iter:
        return False
    elif current_iter == 0:
        return True
    elif metrics.mean_squared_error(current_param, param_snapshot.get_snapshot()) > threshold:
        param_snapshot.take_snapshot(current_param)
        return True
    else:
        return False

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


def count_MSE(method, args, realX):
    if method=='Kalman':
        X_est, Sig_est = kalman_filter(*args)
    if method=='RTS':
        X_est, Sig_est, L_est = rts_smoothing(*args)
    mse=metrics.mean_squared_error(StandardScaler().fit_transform(X_est),
                                   StandardScaler().fit_transform(realX))
    return mse


def sym_MSE(method, args, realX, iter=1):
    results=0
    for k in range(iter):
        results+=count_MSE(method, args, realX)
    results=results/iter
    return results


def initial_state(throw, sigma):
    if throw=='flat':
        mu = [2, 1000, 10, 5]
    elif throw=='high':
        mu = [10, 10, 5, 10]
    if sigma is None:
        Sigma = np.zeros((4, 4))
    elif sigma=='small':
        Sigma = np.zeros((4, 4))+0.2 + np.eye(4)*0.1
    elif sigma=='big':
        Sigma = np.zeros((4, 4)) + 3 + np.eye(4) * 2

    initstate=np.random.multivariate_normal(mu, Sigma)

    return initstate, Sigma
