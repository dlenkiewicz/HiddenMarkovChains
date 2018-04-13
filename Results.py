import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from Functions import generate_data, kalman_filter, rts_smoothing, generate_QR_EM, sym_MSE, count_MSE, \
    SnapshotListHolder, evaluate_stop_condition, initial_state


### Data generator ###

d = 4
p = 1
m = 2
T = 500
dt = 0.1
F = np.eye((d))
F[0, 2] = dt
F[1, 3] = dt

B = np.array([0, -0.5 * dt ** 2, 0, -0.5 * dt ** 2])
u = 9.8
Q = np.eye((d)) * 0.01

R = np.eye((m)) * 100
#R = np.eye((m)) * 10


### Initial states: 'flat' throw with 'none' sigma, 'flat' with 'big' sigma, 'flat' with 'small' sigma, same for 'high' throw ###

X0, Sigma= initial_state("flat", "small")
X = np.zeros((T, 4));
X[0, :] = X0

H = np.zeros((m, d));
H[0, 0] = 1;
H[1, 1] = 1

X, Y = generate_data(X=X, F=F, B=B, u=u, Q=Q, H=H, R=R, T=T)
X = X.T
Y = Y.T

plt.scatter(X[:, 0], X[:, 1], s=1, color='crimson', label="X")
plt.scatter(Y[:, 0], Y[:, 1], s=1, color='blue', label="Y")


### Kalman filter ###

X_kk, Sigma_kk = kalman_filter(X=X0, Y=Y, F=F, B=B, u=u, Q=Q, H=H, R=R, Sigma=Sigma, T=T)

print("MSE - Kalman filter:",sym_MSE(method='Kalman',args=(X0, Y, F, B, u, Q, H, R, Sigma, T),realX=X,iter=100))

plt.scatter(X_kk[:, 0], X_kk[:, 1], s=1, color='green', label="est. X")


### RTS smoothing ###

smooth_x, smooth_sig, L_smooth = rts_smoothing(X_kk, Sigma_kk, F, Q)

print("MSE - RTS smoothing:",sym_MSE(method='RTS',args=(X_kk, Sigma_kk, F, Q),realX=X,iter=100))

plt.scatter(smooth_x[:, 0], smooth_x[:, 1], s=1, color='black', label="smoothed est. X")
plt.legend(loc='best', ncol=2, markerscale=5).get_frame().set_alpha(0.5)
plt.show()


### EM ###

Q_init = Q
R_init = R
Q_EM = np.eye((d))
R_EM = np.eye((m))
X_kk_em = X_kk
Sigma_kk_em = Sigma_kk

Q_snapshot = SnapshotListHolder(Q_EM)
R_snapshot = SnapshotListHolder(R_EM)
current_iter = 0

while evaluate_stop_condition(Q_EM, Q_snapshot, current_iter, 50, threshold=0.001) and \
        evaluate_stop_condition(R_EM, R_snapshot, current_iter, 50, threshold=0.001):
    Q = Q_EM
    R = R_EM
    X_kk_em, Sigma_kk_em = kalman_filter(X=X0, Y=Y, F=F, B=B, u=u, Q=Q, H=H, R=R, Sigma=Sigma, T=T)
    smooth_x, smooth_sig, L_smooth = rts_smoothing(X_kk_em, Sigma_kk_em, F, Q)
    Q_EM, R_EM = generate_QR_EM(smooth_x, smooth_sig, L_smooth, X_kk, Y, F, H, B*u)
    current_iter += 1

X_kk_em, Sigma_kk_em = kalman_filter(X=X0, Y=Y, F=F, B=B, u=u, Q=Q_EM, H=H, R=R_EM, Sigma=Sigma, T=T)

print("EM converged in", current_iter, "steps.")

print("MSE - EM - Q:", metrics.mean_squared_error(StandardScaler().fit_transform(Q_init), StandardScaler().fit_transform(Q_EM)))
print("MSE - EM - R:", metrics.mean_squared_error(StandardScaler().fit_transform(R_init), StandardScaler().fit_transform(R_EM)))

print("MSE - Kalman with EM Q and R:",sym_MSE(method='Kalman',args=(X0, Y, F, B, u, Q_EM, H, R_EM, Sigma, T),realX=X, iter=100))

plt.scatter(X_kk[:, 0], X_kk[:, 1], s=1, color='green', label="est. X with theor. Q and R")
plt.scatter(X_kk_em[:, 0], X_kk_em[:, 1], s=1, color='orange', label="est. X with EM Q and R")
plt.legend(loc='best', ncol=2,markerscale=5).get_frame().set_alpha(0.5)
plt.show()